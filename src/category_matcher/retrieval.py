"""Independent candidate retrieval branches and rank fusion."""

import json
from pathlib import Path

import bm25s
import numpy as np
from rapidfuzz import fuzz, process

from category_matcher.domain import Category, Hit
from category_matcher.normalization import retrieval_terms

METHODS = ("exact", "fuzzy", "bm25", "semantic", "hybrid")
RETRIEVAL_CONFIG = {
    "bm25_limit": 100,
    "semantic_limit": 100,
    "fuzzy_limit": 50,
    "fuzzy_floor": 45,
    "fusion_limit": 200,
    "parent_limit": 5,
    "rrf_k": 10,
    "branch_weights": {"bm25": 1, "fuzzy": 1, "semantic": 5},
    "semantic_query_instruction": "",
    "bm25_representation": "label_only",
}


def tokenize(texts: list[str]):
    return bm25s.tokenize(
        [retrieval_terms(text) for text in texts],
        stopwords=[],
        show_progress=False,
    )


def build_bm25(categories: list[Category], directory: Path) -> None:
    index = bm25s.BM25()
    index.index(tokenize([c.label for c in categories]), show_progress=False)
    index.save(str(directory), show_progress=False)
    (directory / "category_ids.json").write_text(json.dumps([c.category_id for c in categories]))


class LexicalRetriever:
    def __init__(self, directory: Path, categories: dict[str, Category], aliases: list[dict]):
        self.index = bm25s.BM25.load(str(directory), show_progress=False, verbosity=0)
        self.ids = json.loads((directory / "category_ids.json").read_text())
        if set(self.ids) != set(categories) or len(self.ids) != len(categories):
            raise ValueError("BM25 category mapping differs from the canonical store")
        self.choices = [(retrieval_terms(c.label), c.category_id) for c in categories.values()]
        self.choices.extend((retrieval_terms(a["alias"]), a["category_id"]) for a in aliases)

    def bm25(self, query: str, allowed: set[str], limit: int) -> list[Hit]:
        results, scores = self.index.retrieve(
            tokenize([query]),
            corpus=self.ids,
            k=len(self.ids),
            show_progress=False,
            n_threads=1,
        )
        # Filter scope before limiting, retaining global corpus IDF statistics.
        rows = [
            (str(cid), float(score))
            for cid, score in zip(results[0], scores[0], strict=True)
            if str(cid) in allowed and float(score) > 0
        ]
        rows.sort(key=lambda item: (-item[1], item[0]))
        return [Hit(cid, score, "bm25", i) for i, (cid, score) in enumerate(rows[:limit], 1)]

    def fuzzy(self, query: str, allowed: set[str], limit: int, floor: float) -> list[Hit]:
        choices = [(text, cid) for text, cid in self.choices if cid in allowed]
        matches = process.extract(
            retrieval_terms(query),
            [text for text, _ in choices],
            scorer=fuzz.WRatio,
            score_cutoff=floor,
            limit=None,
        )
        best: dict[str, float] = {}
        for _, score, index in matches:
            cid = choices[index][1]
            best[cid] = max(best.get(cid, 0), float(score))
        rows = sorted(best.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return [Hit(cid, score, "fuzzy", i) for i, (cid, score) in enumerate(rows, 1)]


def fuse(
    branches: dict[str, list[Hit]],
    limit: int,
    constant: int = 60,
    weights: dict[str, float] | None = None,
) -> list[Hit]:
    scores: dict[str, float] = {}
    for name, hits in branches.items():
        seen = set()
        for hit in hits:
            if hit.category_id not in seen:
                weight = 1 if weights is None else weights.get(name, 1)
                scores[hit.category_id] = scores.get(hit.category_id, 0) + weight / (
                    constant + hit.rank
                )
                seen.add(hit.category_id)
    rows = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [Hit(cid, score, "rrf", i) for i, (cid, score) in enumerate(rows, 1)]


def expand_parents(
    hits: list[Hit], categories: dict[str, Category], allowed: set[str], limit: int
) -> list[Hit]:
    expanded = list(hits)
    if limit <= 0:
        return expanded
    included = {hit.category_id for hit in hits}
    for hit in hits[:5]:
        parent_id = categories[hit.category_id].parent_id
        if parent_id and parent_id in allowed and parent_id not in included:
            expanded.append(Hit(parent_id, 0.0, "parent_expansion", len(expanded) + 1))
            included.add(parent_id)
            if len(expanded) - len(hits) >= limit:
                break
    return expanded


def cosine_reference(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Independent numerical oracle used by vector integration checks."""
    return (vectors @ query) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query))
