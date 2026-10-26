"""Google matching orchestration. Retrieval scores never imply acceptance."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from category_matcher.artifacts import load_manifest
from category_matcher.decision import DecisionPolicy
from category_matcher.domain import CandidateReranker, Hit, MatchResult
from category_matcher.encoder import BGEEncoder
from category_matcher.normalization import exact_key
from category_matcher.query import (
    is_attribute_only,
    retrieval_text,
    split_context,
    unsupported_audience_ids,
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
        self.reranker_metadata = reranker_metadata
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

    def close(self) -> None:
        self.store.close()

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        if self.encoder is None:
            self.encoder = BGEEncoder(self.manifest["model"])
        return self.encoder.encode_queries(texts)

    def _record(self, category_id: str) -> dict:
        category = self.store.categories[category_id]
        return {"id": category.category_id, "label": category.label, "path": category.full_path}

    def retrieval_query(self, text: str, context: str = "") -> str:
        label, parts = split_context(text.strip(), context.strip())
        # A resolved parent already constrains the search. Repeating its broad
        # words in the query overwhelms the actual leaf concept (e.g. hoodies).
        unresolved = [part for part in parts if len(self.exact.get(exact_key(part), set())) != 1]
        return label + (". Context: " + "; ".join(unresolved) if unresolved else "")

    def match(
        self,
        text: str = "",
        *,
        method: str = "hybrid",
        category_id: str | None = None,
        parent_id: str | None = None,
        language: str = "en",
        context: str = "",
        query_vector: np.ndarray | None = None,
        force_retrieval: bool = False,
        rerank: bool = False,
    ) -> MatchResult:
        if method not in METHODS:
            raise ValueError(f"Unknown retrieval method: {method}")
        text = text.strip()
        context = context.strip()
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
        exact_ids = canonical & allowed
        exact_method = "exact"
        if not exact_ids:
            exact_ids = self.aliases.get(exact_key(text), set()) & allowed
            exact_method = "approved_alias"
        # Extra context must not turn an otherwise exact-looking term into a blind shortcut.
        safe_context = not context_parts or canonical_context or category_id is not None
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
        if is_attribute_only(text) and parent_id and canonical_context:
            result.reason_code = "attribute_only_use_broad_context_provisionally"
            result.candidates = [{**self._record(parent_id), "rank": 1, "source": "context_only"}]
            if rerank:
                result.method += "+rerank"
                result.ranking_model = self.reranker_metadata
            return self.policy.apply(result) if self.policy is not None else result
        query = self.retrieval_query(raw_text, context)
        excluded = unsupported_audience_ids(
            retrieval_text(raw_text, context), self.store.categories
        )
        allowed -= excluded
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
        hits = expand_parents(hits, self.store.categories, allowed, self.config["parent_limit"])
        for hit in hits:
            evidence = {
                name: {"rank": h.rank, "raw_score": h.score, "higher_is_better": True}
                for name, branch in branches.items()
                for h in branch
                if h.category_id == hit.category_id
            }
            result.candidates.append(
                {
                    **self._record(hit.category_id),
                    "rank": hit.rank,
                    "source": hit.source,
                    "score": hit.score,
                    "evidence": evidence,
                }
            )
        if rerank:
            if self.reranker is None:
                raise ValueError("Download and configure a reranker before enabling reranking")
            result.candidates = self.reranker.rank(query, result.candidates)
            result.method += "+rerank"
            result.reason_code = "uncalibrated_reranking"
            result.ranking_model = self.reranker_metadata
        return self.policy.apply(result) if self.policy is not None else result
