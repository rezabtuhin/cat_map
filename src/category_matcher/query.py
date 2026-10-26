"""Explicit vendor-label syntax and narrow, explainable specificity checks."""

import re

from category_matcher.domain import Category
from category_matcher.normalization import exact_key, retrieval_terms, search_text

INFANT_CUES = {"baby", "babies", "toddler", "toddlers", "infant", "infants", "newborn", "newborns"}
ATTRIBUTE_WORDS = {"eco", "friendly", "sustainable", "recycled", "recyclable", "ethical"}


def split_context(text: str, context: str) -> tuple[str, list[str]]:
    parts = [part.strip() for part in text.split("|")]
    if any(not part for part in parts):
        raise ValueError("Pipe-separated categories must have a label and nonempty context")
    return parts[0], parts[1:] + ([context] if context else [])


def retrieval_text(text: str, context: str = "") -> str:
    label, parts = split_context(text.strip(), context.strip())
    return label + (". Context: " + "; ".join(parts) if parts else "")


def unsupported_audience_ids(query: str, categories: dict[str, Category]) -> set[str]:
    """A generic apparel input cannot introduce an infant demographic.

    This is deliberately limited to the observed baby/toddler apparel failure.
    It is not a general claim to understand every product qualifier.
    """
    if set(re.findall(r"\w+", exact_key(query))) & INFANT_CUES:
        return set()
    return {
        cid
        for cid, category in categories.items()
        if "baby and toddler clothing" in exact_key(category.full_path)
    }


def is_attribute_only(text: str) -> bool:
    words = set(search_text(text).split())
    return bool(words) and words <= ATTRIBUTE_WORDS


def scope_only_entailment_support(candidates: list[dict], parent_id: str | None) -> bool:
    """Detect a model that endorses only the supplied scope, not a product type.

    Zero is the boundary between entailment and the combined other NLI logits;
    it is not a calibrated category-correctness threshold. Unscored candidates
    provide no model evidence and do not count as rejected descendants.
    """
    if not candidates or parent_id is None or candidates[0]["id"] != parent_id:
        return False
    parent_support = candidates[0].get("entailment_logit_margin")
    descendants = [
        candidate["entailment_logit_margin"]
        for candidate in candidates
        if candidate["id"] != parent_id and candidate.get("entailment_logit_margin") is not None
    ]
    return (
        parent_support is not None
        and parent_support > 0
        and len(descendants) >= 2
        and all(score <= 0 for score in descendants)
    )


def explicit_broad_category(
    text: str, categories: dict[str, Category], allowed: set[str]
) -> str | None:
    """Resolve an explicitly broad label, not an inferred common ancestor."""
    normalized = search_text(text)
    explicit = normalized.endswith(" in general") or (
        normalized.startswith("general ") and normalized.endswith(" category")
    )
    if not explicit:
        return None
    for key in literal_broad_variants(text):
        exact = [
            cid
            for cid, category in categories.items()
            if cid in allowed and exact_key(category.label) == key
        ]
        if exact:
            return exact[0] if len(exact) == 1 else None
    words = set(retrieval_terms(text).split()) - {"category", "device", "equipment", "item"}
    matches = [
        c
        for cid, c in categories.items()
        if cid in allowed and set(retrieval_terms(c.label).split()) == words
    ]
    if not matches:
        return None
    shallowest = min(c.depth for c in matches)
    best = [c for c in matches if c.depth == shallowest]
    return best[0].category_id if len(best) == 1 else None


def literal_broad_variants(text: str) -> list[str]:
    normalized = exact_key(text)
    if normalized.endswith(" in general"):
        normalized = normalized.removesuffix(" in general").strip()
    elif normalized.startswith("general ") and normalized.endswith(" category"):
        normalized = normalized.removeprefix("general ").removesuffix(" category").strip()
    else:
        return []
    variants = [normalized]
    for suffix in (" products", " goods", " equipment", " devices", " items"):
        if normalized.endswith(suffix):
            variants.append(normalized.removesuffix(suffix).strip())
            break
    return variants


def literal_broad_label(text: str) -> str:
    variants = literal_broad_variants(text)
    return variants[-1] if variants else ""
