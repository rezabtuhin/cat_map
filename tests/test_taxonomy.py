from pathlib import Path

import pytest

from category_matcher.normalization import exact_key, search_text
from category_matcher.taxonomy import read_google_taxonomy


def test_source_audit_and_all_parents():
    categories, audit = read_google_taxonomy(Path("google_category_list.txt"))
    assert audit["record_count"] == 5582
    assert audit["top_level_count"] == 21
    assert audit["leaf_count"] == 4709
    assert audit["normalized_label_collisions"] == {}
    by_id = {c.category_id: c for c in categories}
    assert all(c.parent_id in by_id for c in categories if c.parent_id)
    assert by_id["2353"].parent_id == "264"


@pytest.mark.parametrize(
    "text",
    [
        "not a record",
        "1 - Root\n1 - Another",
        "1 - Root\n2 - Root",
        "2 - Absent > Child",
        "1 - Root > ",
        "0 - Invalid",
        "",
        "1 - Root >  ",
    ],
)
def test_invalid_source_rejected(tmp_path, text):
    source = tmp_path / "bad.txt"
    source.write_text(text)
    with pytest.raises(ValueError):
        read_google_taxonomy(source)


def test_bom_header_and_no_final_newline(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("\ufeff# Version: fixture\n\n1 - Root\n2 - Root > Child")
    categories, audit = read_google_taxonomy(source)
    assert len(categories) == 2
    assert audit["source_comments"] == ["# Version: fixture"]
    assert not categories[0].is_leaf
    assert categories[1].parent_id == "1"


def test_label_collisions_detected_without_removing_meaningful_words(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("1 - Root\n2 - Root > A & B\n3 - Root > A and B")
    _, audit = read_google_taxonomy(source)
    assert audit["normalized_label_collisions"] == {"a and b": 2}
    assert exact_key("  Ａ & B  ") == "a and b"
    assert exact_key("non-prescription food") != exact_key("prescription food")
    assert exact_key("phone-cases") != exact_key("phone cases")
    assert search_text("phone-cases") == "phone cases"
