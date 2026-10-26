"""Explicit vendor-label syntax and narrow, explainable specificity checks."""

import re

from category_matcher.domain import Category
from category_matcher.normalization import exact_key, search_text

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
