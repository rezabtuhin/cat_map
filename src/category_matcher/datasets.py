"""Review provenance and leakage checks shared by calibration and qualification."""

import json
from pathlib import Path

from category_matcher.normalization import exact_key


def is_reviewed(row: dict) -> bool:
    return bool(
        row.get("review_status") == "approved" and row.get("reviewer") and row.get("reviewed_at")
    )


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def check_splits(paths: dict[str, Path]) -> dict:
    seen_ids, seen_queries, seen_families, seen_targets = {}, {}, {}, {}
    counts, conflicts = {}, []
    for split, path in paths.items():
        rows = read_jsonl(path)
        counts[split] = {"examples": len(rows), "unreviewed": sum(not is_reviewed(r) for r in rows)}
        local_ids = set()
        local_queries = set()
        for row in rows:
            identifier = row["example_id"]
            if identifier in local_ids:
                conflicts.append(f"Duplicate ID {identifier} in {split}")
            local_ids.add(identifier)
            query = (
                exact_key(row["query"]),
                row.get("category_id"),
                exact_key(row.get("context", "")),
                row.get("parent_category_id"),
                row.get("language", "en"),
                tuple(row.get("source_ancestors") or []),
            )
            family = exact_key(row.get("input_family") or row["query"])
            if query in local_queries:
                conflicts.append(f"Duplicate input in {split}: {query}")
            local_queries.add(query)
            entries = [
                (seen_ids, identifier, "ID"),
                (seen_queries, query, "query"),
                (seen_families, family, "family"),
            ]
            entries += [
                (
                    seen_targets,
                    (row["taxonomy_key"], row["taxonomy_version"], cid),
                    "target category family",
                )
                for cid in row["acceptable_category_ids"]
            ]
            for seen, key, kind in entries:
                if key in seen and seen[key] != split:
                    conflicts.append(f"Overlapping {kind} between {seen[key]} and {split}: {key}")
                seen[key] = split
    return {
        "counts": counts,
        "conflicts": sorted(set(conflicts)),
        "passed": not conflicts,
        "human_review_complete": all(v["unreviewed"] == 0 for v in counts.values()),
    }
