"""Google matching orchestration. Retrieval scores never imply acceptance."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from category_matcher.artifacts import load_manifest
from category_matcher.decision import DecisionPolicy
from category_matcher.domain import CandidateReranker, Hit, MatchResult
from category_matcher.encoder import BGEEncoder
from category_matcher.normalization import exact_key, retrieval_terms
from category_matcher.query import (
    explicit_broad_category,
    is_attribute_only,
    is_audience_only,
    is_merchandising_label,
    is_placeholder_label,
    literal_broad_variants,
    literal_group_candidates,
    retrieval_text,
    scope_only_entailment_support,
    source_ancestry,
    source_supports_lookup,
    split_context,
    store_product_terms,
    unsupported_audience_ids,
    weak_promotion_evidence,
)
from category_matcher.retrieval import METHODS, LexicalRetriever, expand_parents, fuse
from category_matcher.storage import CategoryStore


class Matcher:
    def __init__(
        self,
        bundle: Path,
        reranker: CandidateReranker | None = None,
        reranker_metadata: dict | None = None,
    ):
        self.manifest = load_manifest(bundle)
        self.version = self.manifest["taxonomy_version"]
        self.store = CategoryStore(
            bundle / "taxonomy.sqlite3", "google", self.version, self.manifest["model"]["dimension"]
        )
        self.config = self.manifest["retrieval"]
        self.encoder: BGEEncoder | None = None
        self.reranker = reranker
        self.reranker_metadata = reranker_metadata or self.manifest.get("ranking_model")
        self.default_rerank = self.manifest.get("ranking_model") is not None
        self.policy: DecisionPolicy | None = None
        aliases = json.loads((bundle / "aliases.json").read_text())
        self.exact: dict[str, set[str]] = defaultdict(set)
        for category in self.store.categories.values():
            for text in (category.label, category.full_path):
                self.exact[exact_key(text)].add(category.category_id)
        self.aliases: dict[str, set[str]] = defaultdict(set)
        for alias in aliases:
            self.aliases[exact_key(alias["alias"])].add(alias["category_id"])
        self.lexical = LexicalRetriever(bundle / "bm25", self.store.categories, aliases)
        self._source_scope_terms: dict[str, set[str]] = {}
        self._source_label_cache: dict[str, set[str]] = {}

    def close(self) -> None:
        self.store.close()

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        if self.encoder is None:
            self.encoder = BGEEncoder(self.manifest["model"])
        return self.encoder.encode_queries(texts)

    def _record(self, category_id: str) -> dict:
        category = self.store.categories[category_id]
        return {"id": category.category_id, "label": category.label, "path": category.full_path}

    def _within(self, child: str, parent: str) -> bool:
        return child == parent or self.store.categories[child].full_path.startswith(
            self.store.categories[parent].full_path + " > "
        )

    def _source_label_ids(self, label: str) -> set[str]:
        key = exact_key(label)
        if key in self._source_label_cache:
            return self._source_label_cache[key]
        ids = (
            self.exact.get(key, set())
            or self.aliases.get(key, set())
            or set(
                literal_group_candidates(label, self.store.categories, set(self.store.categories))
            )
        )
        if len(self._source_label_cache) >= 256:
            self._source_label_cache.pop(next(iter(self._source_label_cache)))
        self._source_label_cache[key] = ids
        return ids

    def _source_domain(self, cid: str) -> str:
        parts = self.store.categories[cid].full_path.split(" > ")
        return next((part for part in parts if not is_merchandising_label(part)), parts[0])

    def _source_ancestor(self, ancestors: list[str], allowed: set[str], text: str):
        """Ground generic labels in compatible product ancestors, without neural guesses."""
        anchor = None
        domain = None
        unresolved = []
        evidence = {}
        for index, raw in enumerate(ancestors):
            try:
                label, parts = split_context(raw, "")
            except ValueError:
                return None, {"reason": "invalid_source_ancestor_context"}
            for part in [*parts, label]:
                if (
                    is_merchandising_label(part)
                    or is_placeholder_label(part)
                    or is_attribute_only(part)
                    or is_audience_only(part)
                ):
                    continue
                ids = self._source_label_ids(part)
                if not ids:
                    if anchor is not None:
                        unresolved.append(part)
                    continue
                compatible = ids & allowed
                if domain is not None:
                    compatible = {cid for cid in compatible if self._source_domain(cid) == domain}
                if len(compatible) != 1:
                    return None, {
                        "reason": "conflicting_source_ancestors"
                        if not compatible
                        else "ambiguous_source_ancestor",
                        "source_label": part,
                    }
                cid = next(iter(compatible))
                # Store hierarchy can put canonical siblings under one another;
                # retain the nearest product type within the same broad domain.
                if anchor is None or not self._within(anchor, cid):
                    anchor = cid
                    domain = self._source_domain(cid)
                    unresolved = []
                    evidence = {"category_id": cid, "source_label": part, "source_index": index}
        if anchor is None:
            return None, {"reason": "no_grounded_product_ancestor"}
        # A title naming a product type can refine the inherited product evidence.
        hints = store_product_terms(text, self.lexical.vocabulary)
        if hints:
            hint_ids = self._source_label_ids(" ".join(sorted(hints))) & allowed
            compatible = {cid for cid in hint_ids if self._source_domain(cid) == domain}
            if len(compatible) != 1:
                return None, {**evidence, "reason": "unresolved_product_type_in_store_label"}
            hint = next(iter(compatible))
            if not self._within(anchor, hint):
                anchor = hint
        if anchor not in self._source_scope_terms:
            self._source_scope_terms[anchor] = {
                word
                for cid in self.store.allowed_ids(anchor)
                for word in retrieval_terms(self.store.categories[cid].full_path).split()
            }
        scope_terms = self._source_scope_terms[anchor]
        for part in [*unresolved, text]:
            terms = store_product_terms(part, self.lexical.vocabulary)
            if terms - scope_terms:
                return None, {
                    **evidence,
                    "reason": "source_product_type_conflict",
                    "conflicting_label": part,
                }
        return anchor, {
            **evidence,
            "selected_category_id": anchor,
            "reason": "grounded_source_ancestor",
        }

    def retrieval_query(
        self, text: str, context: str = "", source_ancestors: list[str] | None = None
    ) -> str:
        label, parts = split_context(text.strip(), context.strip())
        # A resolved parent already constrains the search. Repeating its broad
        # words in the query overwhelms the actual leaf concept (e.g. hoodies).
        unresolved = [part for part in parts if len(self.exact.get(exact_key(part), set())) != 1]
        if is_merchandising_label(label) and unresolved:
            return "; ".join(unresolved)
        query = label + (". Context: " + "; ".join(unresolved) if unresolved else "")
        resolved = {
            exact_key(part) for part in parts if len(self.exact.get(exact_key(part), set())) == 1
        }
        parents = [
            part
            for part in source_ancestry(source_ancestors)
            if not is_merchandising_label(part) and exact_key(part) not in resolved
        ]
        return query + (". Store ancestry: " + " > ".join(parents) if parents else "")

    def match(
        self,
        text: str = "",
        *,
        method: str = "hybrid",
        category_id: str | None = None,
        parent_id: str | None = None,
        language: str = "en",
        context: str = "",
        source_ancestors: list[str] | None = None,
        query_vector: np.ndarray | None = None,
        force_retrieval: bool = False,
        rerank: bool | None = None,
        retrieve: bool = True,
    ) -> MatchResult:
        if method not in METHODS:
            raise ValueError(f"Unknown retrieval method: {method}")
        if rerank is None:
            rerank = self.default_rerank and method == "hybrid"
        text = text.strip()
        context = context.strip()
        ancestors = source_ancestry(source_ancestors)
        if not text and category_id is None:
            raise ValueError("Provide nonempty category text or an explicit category ID")
        if len(text) > 300 or len(context) > 1000:
            raise ValueError("Category text is limited to 300 and context to 1000 characters")
        result = MatchResult(
            "provisional", "google", self.version, self.manifest["artifact_version"], method
        )
        if language != "en":
            result.status = "unsupported_language"
            result.reason_code = "english_only_prototype"
            return result
        raw_text = text
        text, context_parts = (
            split_context(text, context) if text else ("", [context] if context else [])
        )
        context_keys = [self.exact.get(exact_key(part), set()) for part in context_parts]
        canonical_context = bool(context_parts) and all(len(ids) == 1 for ids in context_keys)
        self.store.allowed_ids(parent_id)  # Validate an explicit constraint before inferring scope.
        for ids in context_keys:
            if len(ids) != 1:
                continue
            inferred_id = next(iter(ids))
            if parent_id is None or inferred_id in self.store.allowed_ids(parent_id):
                parent_id = inferred_id
            elif parent_id not in self.store.allowed_ids(inferred_id):
                raise ValueError("Category context contradicts the parent constraint")
        allowed = self.store.allowed_ids(parent_id)
        canonical = self.exact.get(exact_key(text), set())
        if category_id is not None:
            if category_id not in allowed:
                raise ValueError("Unknown category ID or ID outside the parent constraint")
            if text and (canonical != {category_id}):
                raise ValueError("Explicit category ID and category text are inconsistent")
            canonical = {category_id}
        if canonical and not canonical & allowed:
            raise ValueError("Canonical category text contradicts the parent constraint")
        placeholder = is_placeholder_label(text) and category_id is None
        meaningful_context = any(
            not (
                is_placeholder_label(part)
                or is_merchandising_label(part)
                or is_attribute_only(part)
                or is_audience_only(part)
            )
            for part in context_parts
        )
        if placeholder and not meaningful_context:
            result.status = "needs_clarification"
            result.reason_code = "placeholder_label_requires_product_context"
            result.clarification = "Provide a product type or description; a default category name does not identify one."
            result.query_analysis["placeholder_label"] = {"retrieval_skipped": True}
            return result
        exact_ids = canonical & allowed
        exact_method = "exact"
        if not exact_ids:
            exact_ids = self.aliases.get(exact_key(text), set()) & allowed
            exact_method = "approved_alias"
        # Extra context must not turn an otherwise exact-looking term into a blind shortcut.
        safe_context = not context_parts or canonical_context or category_id is not None
        if len(exact_ids) == 1 and category_id is None:
            safe_context = safe_context and source_supports_lookup(
                ancestors, self.store.categories[next(iter(exact_ids))]
            )
        if len(exact_ids) == 1 and not force_retrieval and safe_context:
            cid = next(iter(exact_ids))
            result.status = "matched"
            result.method = exact_method
            result.category = self._record(cid)
            result.candidates = [{**result.category, "source": exact_method, "rank": 1}]
            result.reason_code = "canonical_lookup" if exact_method == "exact" else "reviewed_alias"
            return result
        if method == "exact":
            return result
        merchandising = is_merchandising_label(text) or placeholder
        context_only = merchandising or is_attribute_only(text) or is_audience_only(text)
        if (
            context_only
            and ancestors
            and (not context_parts or canonical_context)
            and not placeholder
        ):
            inherited, analysis = self._source_ancestor(ancestors, allowed, text)
            result.query_analysis["source_ancestor"] = analysis
            if inherited is not None:
                result.method = "source_ancestor"
                result.reason_code = "specific_product_ancestor_provisional"
                result.candidates = [
                    {**self._record(inherited), "rank": 1, "source": "source_ancestor"}
                ]
            else:
                result.status = "needs_clarification"
                result.reason_code = analysis["reason"]
                result.clarification = "Provide the product type; the store hierarchy is missing, ambiguous, or conflicting."
            return result
        if placeholder and canonical_context and parent_id:
            result.reason_code = "explicit_product_context_provisional"
            result.candidates = [{**self._record(parent_id), "rank": 1, "source": "context_only"}]
            return result
        if merchandising and (not context_parts or canonical_context):
            result.query_analysis["merchandising_label"] = {"requires_product_context": True}
            result.reason_code = "merchandising_label_requires_product_context"
            result.clarification = (
                "What product types are included in this collection or recommendation?"
            )
            result.status = "needs_clarification"
            if parent_id is not None:
                result.candidates = [
                    {**self._record(parent_id), "rank": 1, "source": "context_only"}
                ]
            return result
        broad = explicit_broad_category(text, self.store.categories, allowed)
        if broad is not None and (not context_parts or canonical_context):
            result.reason_code = "explicit_broad_category"
            result.candidates = [{**self._record(broad), "rank": 1, "source": "explicit_broad"}]
            result.query_analysis = {"broad_category": broad}
            if source_supports_lookup(ancestors, self.store.categories[broad]) and any(
                self.exact.get(key, set()) == {broad} for key in literal_broad_variants(text)
            ):
                result.status = "matched"
                result.method = "explicit_broad"
                result.category = self._record(broad)
                result.reason_code = "canonical_label_with_explicit_general_wrapper"
                return result
            # This uses retrieval normalization, so retain provisional status.
            if rerank:
                result.method += "+rerank"
                result.ranking_model = self.reranker_metadata
            return self.policy.apply(result) if self.policy else result
        if (is_attribute_only(text) or is_audience_only(text)) and parent_id and canonical_context:
            result.reason_code = "attribute_only_use_broad_context_provisionally"
            result.candidates = [{**self._record(parent_id), "rank": 1, "source": "context_only"}]
            if rerank:
                result.method += "+rerank"
                result.ranking_model = self.reranker_metadata
            return self.policy.apply(result) if self.policy is not None else result
        if context_only and not meaningful_context:
            result.status = "needs_clarification"
            result.reason_code = "generic_label_requires_product_context"
            result.clarification = (
                "Provide a product type, description, or meaningful product ancestry."
            )
            return result
        if not retrieve:
            result.reason_code = "retrieval_required"
            return result
        query = self.retrieval_query(raw_text, context, ancestors)
        if merchandising:
            # A store section title does not name the products. Use the supplied
            # unresolved product description while retaining resolved parent scope.
            text = "; ".join(
                part for part, ids in zip(context_parts, context_keys, strict=True) if len(ids) != 1
            )
            query = text
            result.query_analysis["merchandising_label"] = {"used_product_context": True}
        excluded = unsupported_audience_ids(
            retrieval_text(raw_text, context) + " " + " ".join(ancestors), self.store.categories
        )
        allowed -= excluded
        groups = literal_group_candidates(text, self.store.categories, allowed)
        if groups:
            result.query_analysis["literal_group"] = {
                "candidate_ids": groups,
                "ambiguous": len(groups) > 1,
                "reason": "whole_alternative_in_coordinated_canonical_label",
            }
        spelling = self.lexical.spelling(text, allowed, self.config["spelling_min_characters"])
        if spelling["candidates"]:
            spelling["interpretations"] = [self._record(cid) for cid in spelling["candidates"]]
            result.query_analysis["spelling"] = spelling
            if spelling["ambiguous"]:
                result.clarification = (
                    "Please clarify the category: "
                    + ", ".join(self.store.categories[cid].label for cid in spelling["candidates"])
                    + "."
                )
        branches: dict[str, list[Hit]] = {}
        if method in {"bm25", "hybrid"}:
            branches["bm25"] = self.lexical.bm25(query, allowed, self.config["bm25_limit"])
        if method in {"fuzzy", "hybrid"}:
            branches["fuzzy"] = self.lexical.fuzzy(
                text,
                allowed,
                self.config["fuzzy_limit"],
                self.config["fuzzy_floor"],
            )
        if method in {"semantic", "hybrid"}:
            if query_vector is None:
                query_vector = self.encode_queries([query])[0]
            branches["semantic"] = self.store.semantic_search(
                query_vector,
                self.config["semantic_limit"],
                parent_id,
                excluded_ids=excluded,
            )
        hits = (
            fuse(
                branches,
                self.config["fusion_limit"],
                self.config["rrf_k"],
                self.config["branch_weights"],
            )
            if method == "hybrid"
            else next(iter(branches.values()))[: self.config["fusion_limit"]]
        )
        if spelling["eligible"] and len(spelling["candidates"]) == 1:
            # Protect the unique edit-near category before optional neural scoring,
            # retaining the configured pool bound and the original semantic scores.
            cid = spelling["candidates"][0]
            if cid not in {hit.category_id for hit in hits}:
                hits = [Hit(cid, 0.0, "spelling", 1)] + hits
            hits.sort(key=lambda hit: hit.category_id != cid)
            hits = hits[: self.config["fusion_limit"]]
        if groups:
            included = {hit.category_id for hit in hits}
            hits.extend(
                Hit(cid, 0.0, "literal_group", len(hits) + i)
                for i, cid in enumerate(groups, 1)
                if cid not in included
            )
            hits.sort(key=lambda hit: hit.category_id not in groups)
            hits = hits[: self.config["fusion_limit"]]
        hits = expand_parents(hits, self.store.categories, allowed, self.config["parent_limit"])
        for final_rank, hit in enumerate(hits, 1):
            evidence = {
                name: {"rank": h.rank, "raw_score": h.score, "higher_is_better": True}
                for name, branch in branches.items()
                for h in branch
                if h.category_id == hit.category_id
            }
            result.candidates.append(
                {
                    **self._record(hit.category_id),
                    "rank": final_rank,
                    "source": hit.source,
                    "score": hit.score,
                    "evidence": evidence,
                }
            )
        group_order = {candidate["id"]: i for i, candidate in enumerate(result.candidates)}
        if rerank:
            if self.reranker is None:
                if self.reranker_metadata is None:
                    raise ValueError("Download and configure a reranker before enabling reranking")
                from category_matcher.reranker import CrossEncoderReranker

                self.reranker = CrossEncoderReranker(self.reranker_metadata)
            retrieval_order = {candidate["id"]: i for i, candidate in enumerate(result.candidates)}
            retrieval_top = result.candidates[0] if result.candidates else None
            retrieval_first = result.candidates[0]["id"] if result.candidates else None
            result.candidates = self.reranker.rank(query, result.candidates)
            result.method += "+rerank"
            result.reason_code = "uncalibrated_reranking"
            result.ranking_model = self.reranker_metadata
            guard = self.config.get("promotion_guard")
            if retrieval_top and result.candidates and guard:
                conflict = weak_promotion_evidence(retrieval_top, result.candidates[0], guard)
                if conflict:
                    result.query_analysis["promotion_conflict"] = conflict
                    for candidate in result.candidates:
                        candidate["model_rank"] = candidate["rank"]
                    result.candidates.sort(
                        key=lambda candidate: candidate["id"] != retrieval_top["id"]
                    )
                    for i, candidate in enumerate(result.candidates, 1):
                        candidate["rank"] = i
                    result.reason_code = "weak_model_promotion_retrieval_candidate_preserved"
                    result.clarification = (
                        "Retrieval and model ranking disagree; please clarify the product type."
                    )
            if retrieval_first != parent_id and scope_only_entailment_support(
                result.candidates, parent_id
            ):
                result.query_analysis["specificity_conflict"] = {
                    "parent_id": parent_id,
                    "retrieval_first_id": retrieval_first,
                    "reason": "model_supports_scope_but_no_scored_product_type",
                    "action": "preserve_retrieval_order_and_require_clarification",
                }
                for candidate in result.candidates:
                    candidate["model_rank"] = candidate["rank"]
                result.candidates.sort(key=lambda candidate: retrieval_order[candidate["id"]])
                for index, candidate in enumerate(result.candidates, 1):
                    candidate["rank"] = index
                result.reason_code = "scope_only_model_support_retrieval_order_preserved"
                result.clarification = (
                    "The parent category is clear, but the product type is uncertain. "
                    "Please clarify which type of product this category contains."
                )
        if spelling["eligible"] and len(spelling["candidates"]) == 1:
            cid = spelling["candidates"][0]
            result.candidates.sort(key=lambda candidate: candidate["id"] != cid)
            for index, candidate in enumerate(result.candidates, 1):
                candidate["rank"] = index
            result.reason_code = "unique_close_spelling_candidate_provisional"
        if groups:
            result.candidates.sort(
                key=lambda candidate: (
                    candidate["id"] not in groups,
                    group_order[candidate["id"]]
                    if candidate["id"] in groups
                    else candidate["rank"],
                )
            )
            for i, candidate in enumerate(result.candidates, 1):
                candidate["rank"] = i
            result.reason_code = "literal_group_candidate_provisional"
        return self.policy.apply(result) if self.policy is not None else result
