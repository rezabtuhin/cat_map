"""Synthetic ONNX contracts; no checkpoints or network are needed."""

import hashlib
import json

import numpy as np
import pytest

from category_matcher.languages import MULTILINGUAL_INPUTS, language_supported
from category_matcher.matcher import Matcher
from category_matcher.model_assets import REQUIRED_FILES, PinnedFile, verify_file
from category_matcher.normalization import exact_key, search_text
from category_matcher.onnx_backend import ONNXEncoder, verified_model_files
from category_matcher.reranker import CrossEncoderReranker


def test_e5_prefixes_masked_mean_pooling_and_normalization():
    calls = []

    class Model:
        def run(self, texts, output):
            calls.append((texts, output))
            hidden = np.zeros((len(texts), 3, 384), dtype=np.float32)
            hidden[:, 0, 0] = 3
            hidden[:, 1, 1] = 4
            hidden[:, 2, :] = 1000  # Padding must not affect the embedding.
            return hidden, np.array([[1, 1, 0]] * len(texts))

    encoder = ONNXEncoder.__new__(ONNXEncoder)
    encoder.model = Model()
    encoder.metadata = {
        "dimension": 384,
        "batch_size": 1,
        "document_instruction": "passage: ",
        "query_instruction": "query: ",
    }
    document = encoder.encode_documents(["Shirts"])
    query = encoder.encode_queries(["衬衫"])
    assert calls == [
        (["passage: Shirts"], "last_hidden_state"),
        (["query: 衬衫"], "last_hidden_state"),
    ]
    assert document.shape == query.shape == (1, 384)
    np.testing.assert_allclose(document[0, :2], [0.6, 0.8])
    assert document.dtype == np.float32 and np.count_nonzero(document) == 2
    assert encoder.encode_queries([]).shape == (0, 384)


def test_onnx_nli_uses_label_zero_and_stable_log_sum_exp():
    class Predictor:
        def predict(self, pairs, **kwargs):
            assert pairs == [("chemises", "Shirts"), ("chemises", "Shoes")]
            return [[1000.0, -1000.0, -1000.0], [-1000.0, 1000.0, 1000.0]]

    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model = Predictor()
    reranker.metadata = {
        "backend": "onnx",
        "score_profile": "category_entailment_v1",
        "entailment_index": 0,
        "premise_template": "{query}",
        "hypothesis_template": "{label}",
    }
    candidates = [
        {"id": "shirts", "label": "Shirts", "path": "Shirts", "rank": 1},
        {"id": "shoes", "label": "Shoes", "path": "Shoes", "rank": 2},
    ]
    result = reranker.rank("chemises", candidates)
    assert result[0]["id"] == "shirts"
    assert np.isfinite([row["reranker_score"] for row in result]).all()
    assert result[0]["reranker_score"] > 0 > result[1]["reranker_score"]


def test_pinned_onnx_inventory_ignores_download_cache_and_detects_tampering(tmp_path):
    files = {}
    for name in REQUIRED_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = name.encode()
        path.write_bytes(content)
        files[name] = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    metadata = {"snapshot": str(tmp_path), "revision": "a" * 40, "files": files}
    cache = tmp_path / ".cache/huggingface"
    cache.mkdir(parents=True)
    (cache / "download.lock").write_text("unrelated mutable cache metadata")
    assert verified_model_files(metadata) == {key: value["sha256"] for key, value in files.items()}
    path = tmp_path / "config.json"
    path.write_bytes(b"x" * files["config.json"]["size"])
    with pytest.raises(ValueError, match="checksum mismatch"):
        verified_model_files(metadata)


def test_pinned_onnx_files_cannot_escape_snapshot(tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("data")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").symlink_to(outside)
    files = {name: {"size": 4, "sha256": "a" * 64} for name in REQUIRED_FILES}
    # Ensure the escaping path is validated first, independent of set iteration order.
    files = {"config.json": files.pop("config.json"), **files}
    with pytest.raises(ValueError, match="escapes"):
        verified_model_files({"snapshot": str(snapshot), "revision": "a" * 40, "files": files})


def test_existing_files_are_not_replaced_on_checksum_failure(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"old")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_file(path, PinnedFile("model.onnx", 3, hashlib.sha256(b"new").hexdigest()))
    assert path.read_bytes() == b"old"


@pytest.mark.parametrize("language", ["auto", "fr", "fr-CA", "zh-CN", "zh-TW", "bn", "ja"])
def test_language_hints_accept_enabled_languages(language):
    assert language_supported(language, MULTILINGUAL_INPUTS)


def test_unicode_combining_marks_and_localized_placeholders(bundle):
    for text in ("শার্ট", "कपड़े", "ملابس", "衬衫"):
        assert search_text(text) == exact_key(text)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model"]["languages"] = list(MULTILINGUAL_INPUTS)
    manifest_path.write_text(json.dumps(manifest))
    matcher = Matcher(bundle)
    try:
        for text in ("未分类", "Sans catégorie", "অশ্রেণীবদ্ধ", "بدون تصنيف"):
            result = matcher.match(text)
            assert result.status == "needs_clarification"
            assert result.category is None and not result.candidates
        result = matcher.match("chemises", language="fr", retrieve=False)
        assert result.status == "provisional" and result.category is None
        assert matcher.match("chemises", language="zz").status == "unsupported_language"
    finally:
        matcher.close()
