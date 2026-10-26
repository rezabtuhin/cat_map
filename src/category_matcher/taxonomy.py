"""Google text source ingestion; matching code consumes canonical records."""

import hashlib
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

from category_matcher.domain import Category
from category_matcher.normalization import exact_key


def read_google_taxonomy(source: Path) -> tuple[list[Category], dict]:
    raw = source.read_bytes()
    records: list[Category] = []
    by_path: dict[str, str] = {}
    ids: set[str] = set()
    comments: list[str] = []
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            comments.append(line.strip())
            continue
        match = re.fullmatch(r"([1-9][0-9]*) - (.+)", line)
        if not match:
            raise ValueError(f"Malformed source record at line {line_number}")
        category_id, path = match.groups()
        segments = path.split(" > ")
        if any(not segment.strip() for segment in segments) or path != " > ".join(
            segment.strip() for segment in segments
        ):
            raise ValueError(f"Invalid hierarchy at line {line_number}")
        if category_id in ids or path in by_path:
            raise ValueError(f"Duplicate category ID or path at line {line_number}")
        ids.add(category_id)
        by_path[path] = category_id
        records.append(Category(category_id, segments[-1], path, None, len(segments), True))
    if not records:
        raise ValueError("Taxonomy contains no categories")

    children: set[str] = set()
    for i, category in enumerate(records):
        if category.depth == 1:
            continue
        parent_path = category.full_path.rsplit(" > ", 1)[0]
        if parent_path not in by_path:
            raise ValueError(f"Missing parent for category {category.category_id}: {parent_path}")
        parent_id = by_path[parent_path]
        records[i] = replace(category, parent_id=parent_id)
        children.add(parent_id)
    records = [replace(c, is_leaf=c.category_id not in children) for c in records]
    labels = Counter(exact_key(c.label) for c in records)
    audit = {
        "taxonomy_key": "google",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "record_count": len(records),
        "top_level_count": sum(c.depth == 1 for c in records),
        "leaf_count": sum(c.is_leaf for c in records),
        "depth_distribution": dict(sorted(Counter(c.depth for c in records).items())),
        "normalized_label_collisions": {key: count for key, count in labels.items() if count > 1},
        "source_comments": comments,
    }
    return records, audit
