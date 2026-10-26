import json

import pytest
import torch

from category_matcher.reranker import CrossEncoderReranker, align_cpu_parameters, load_profile


def test_misaligned_cpu_weight_copy_preserves_values_and_parameter_identity():
    model = torch.nn.Linear(3, 2)
    model.weight = torch.nn.Parameter(torch.arange(7, dtype=torch.float32)[1:].reshape(2, 3))
    original = model.weight
    expected = original.detach().clone()
    assert model.weight.data_ptr() % 64 != 0
    assert align_cpu_parameters(model) >= 1
    assert model.weight is original
    assert model.weight.data_ptr() % 64 == 0
    assert torch.equal(model.weight, expected)
    assert align_cpu_parameters(model) == 0


def test_entailment_profile_uses_entailment_contradiction_and_neutral_logits():
    class Model:
        def predict(self, pairs, **kwargs):
            assert pairs == [
                ("These products are smart phones.", "These products are Cases."),
                ("These products are smart phones.", "These products are Phones."),
            ]
            return [[8.0, -4.0, -2.0], [-2.0, 6.0, -1.0]]

    ranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    ranker.model = Model()
    ranker.metadata = {
        "score_profile": "category_entailment_v1",
        "entailment_index": 1,
        "premise_template": "These products are {query}.",
        "hypothesis_template": "These products are {label}.",
    }
    output = ranker.rank(
        "smart phones",
        [
            {"id": "case", "label": "Cases", "path": "Cases", "rank": 1},
            {"id": "phone", "label": "Phones", "path": "Phones", "rank": 2},
        ],
    )
    assert output[0]["id"] == "phone"
    assert output[0]["reranker_score"] > 0 and output[1]["reranker_score"] < 0


def test_fused_profile_keeps_all_candidates_and_preserves_a_common_score_scale():
    class Model:
        def predict(self, pairs, **kwargs):
            return [[8.0, -4.0, -2.0], [-2.0, 6.0, -1.0]]

    ranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    ranker.model = Model()
    ranker.metadata = {
        "score_profile": "category_entailment_v1",
        "entailment_index": 1,
        "premise_template": "These products are {query}.",
        "hypothesis_template": "These products are {label}.",
        "candidate_limit": 2,
        "rank_fusion_weight": 3,
        "rank_constant": 10,
    }
    candidates = [
        {"id": cid, "label": cid, "path": cid, "rank": i}
        for i, cid in enumerate(("case", "phone", "tail"), 1)
    ]
    output = ranker.rank("smart phones", candidates)
    assert len(output) == 3 and output[0]["id"] == "phone"
    assert output[-1]["entailment_logit_margin"] is None
    assert all(c["reranker_score"] > 0 for c in output)


def test_profile_pins_revision_and_resolves_metadata_relative_to_config(tmp_path):
    metadata = tmp_path / "model.json"
    metadata.write_text(
        json.dumps(
            {"name": "model", "revision": "pinned", "score_profile": "category_entailment_v1"}
        )
    )
    config = tmp_path / "profile.json"
    profile = {
        "model_metadata": "model.json",
        "model_name": "model",
        "model_revision": "pinned",
        "score_profile": "category_entailment_v1",
        "candidate_limit": 100,
        "rank_fusion_weight": 3,
        "rank_constant": 10,
    }
    config.write_text(json.dumps(profile))
    assert load_profile(config)["candidate_limit"] == 100
    profile["model_revision"] = "other"
    config.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="revision"):
        load_profile(config)
