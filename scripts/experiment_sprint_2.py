"""Development-only retrieval ablation; does not tune on calibration/test data."""

import json
from pathlib import Path

import bm25s
from rapidfuzz import fuzz, process

from category_matcher.artifacts import resolve_bundle
from category_matcher.encoder import QUERY_INSTRUCTION
from category_matcher.matcher import Matcher
from category_matcher.normalization import search_text

STOP = set("a an the and or for of in on to with all general products goods used".split())


def terms(text):
    words = []
    for word in search_text(text).split():
        if word in STOP:
            continue
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 4 and word.endswith(("sses", "ches", "shes", "xes")):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
        words.append(word)
    return " ".join(words)


def main():
    matcher = Matcher(resolve_bundle(Path("artifacts/google")))
    rows = [
        json.loads(line)
        for line in Path("data/google/evaluation/development.jsonl").read_text().splitlines()
    ]
    rows += [
        json.loads(line)
        for line in Path("data/google/evaluation/user_examples.jsonl").read_text().splitlines()
    ]
    ids = list(matcher.store.categories)
    cats = matcher.store.categories
    records = [
        r
        for r in rows
        if r["expected_status"] == "matched"
        and matcher.match(r["query"], method="exact", category_id=r.get("category_id")).status
        != "matched"
    ]
    queries = [matcher.retrieval_query(r["query"]) for r in records]
    matcher.encode_queries(["warmup"])
    instructed = matcher.encoder.encode_documents([QUERY_INSTRUCTION + query for query in queries])
    plain = matcher.encoder.encode_documents(queries)
    index = bm25s.BM25()
    index.index(
        bm25s.tokenize([terms(cats[cid].label) for cid in ids], stopwords=[], show_progress=False),
        show_progress=False,
    )
    results = []
    evidence = []
    for row, q, vi, vp in zip(records, queries, instructed, plain, strict=True):
        label = row["query"].split("|")[0]
        context = row["query"].split("|")[1:]
        scope = None
        if context and len(matcher.exact.get(search_text(context[0]), set())) == 1:
            scope = next(iter(matcher.exact[search_text(context[0])]))
        allowed = matcher.store.allowed_ids(scope)
        from category_matcher.query import unsupported_audience_ids

        allowed -= unsupported_audience_ids(row["query"], cats)
        found, scores = index.retrieve(
            bm25s.tokenize([terms(q)], stopwords=[], show_progress=False),
            corpus=ids,
            k=len(ids),
            show_progress=False,
            n_threads=1,
        )
        lexical = sorted(
            [
                (cid, float(score))
                for cid, score in zip(found[0], scores[0], strict=True)
                if cid in allowed and score > 0
            ],
            key=lambda x: (-x[1], x[0]),
        )[:100]
        fuzzy_choices = [(terms(cats[cid].label), cid) for cid in allowed]
        fuzzy = process.extract(
            terms(label),
            [x[0] for x in fuzzy_choices],
            scorer=fuzz.WRatio,
            score_cutoff=45,
            limit=None,
        )
        fuzzy = sorted(
            [(fuzzy_choices[i][1], float(score)) for _, score, i in fuzzy],
            key=lambda x: (-x[1], x[0]),
        )[:50]
        evidence.append(
            {
                "row": row,
                "lexical": lexical,
                "fuzzy": fuzzy,
                "instruction": [
                    (h.category_id, h.score)
                    for h in matcher.store.semantic_search(
                        vi, 100, scope, excluded_ids=set(cats) - allowed
                    )
                ],
                "plain": [
                    (h.category_id, h.score)
                    for h in matcher.store.semantic_search(
                        vp, 100, scope, excluded_ids=set(cats) - allowed
                    )
                ],
            }
        )
    for view in ("instruction", "plain"):
        for weight in (1, 2, 3, 5):
            for constant in (10, 60):
                correct = recalled = 0
                examples = []
                for item in evidence:
                    score = {}
                    for name, multiplier in [(view, weight), ("lexical", 1), ("fuzzy", 1)]:
                        for rank, (cid, _) in enumerate(item[name], 1):
                            score[cid] = score.get(cid, 0) + multiplier / (constant + rank)
                    ranked = sorted(score, key=lambda cid: (-score[cid], cid))[:80]
                    gold = set(item["row"]["acceptable_category_ids"])
                    correct += bool(ranked and ranked[0] in gold)
                    recalled += bool(set(ranked) & gold)
                    if item["row"]["example_id"].startswith("user"):
                        examples.append((item["row"]["query"], ranked[:3]))
                results.append(
                    {
                        "query_view": view,
                        "semantic_weight": weight,
                        "rrf_constant": constant,
                        "correct": correct,
                        "total": len(evidence),
                        "recalled80": recalled,
                        "user_examples": examples,
                    }
                )
    output = Path("reports/sprint_2/development_ablation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"purpose": "draft development-only exploration", "results": results}, indent=2)
    )
    for result in sorted(results, key=lambda x: (-x["correct"], -x["recalled80"])):
        print(result)
    matcher.close()


if __name__ == "__main__":
    main()
