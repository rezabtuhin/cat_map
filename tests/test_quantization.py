import pytest
import torch

from category_matcher.quantization import apply_profile, tensor_bytes, validate_profile


class EncoderFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Module()
        self.embeddings.word_embeddings = torch.nn.Embedding(512, 32)
        self.rel_embeddings = torch.nn.Embedding(16, 32)
        self.projection = torch.nn.Linear(32, 16)
        self.norm = torch.nn.LayerNorm(16)

    def forward(self, indices):
        # DeBERTa's direct relative-weight access must continue to work.
        return (
            self.norm(self.projection(self.embeddings.word_embeddings(indices)))
            + self.rel_embeddings.weight[0, :16]
        )


@pytest.mark.parametrize("profile", ["int8_embeddings_v1", "int8_dynamic_v1"])
def test_quantization_preserves_forward_contract_and_accounts_for_packed_weights(profile):
    if "qnnpack" not in torch.backends.quantized.supported_engines:
        pytest.skip("QNNPACK is required for this CPU inference profile")
    torch.manual_seed(19)
    model = EncoderFixture().eval()
    indices = torch.tensor([[0, 5, 127], [128, 511, 13]])
    with torch.inference_mode():
        reference = model(indices).clone()
    before = tensor_bytes(model)
    identity = id(model)
    converted = apply_profile(model, profile)
    assert id(model) == identity and converted["converted_embedding"] == 1
    assert isinstance(model.rel_embeddings, torch.nn.Embedding)
    assert isinstance(model.norm, torch.nn.LayerNorm)
    if profile == "int8_embeddings_v1":
        assert isinstance(model.projection, torch.nn.Linear)
        assert converted["converted_linear"] == 0
    else:
        assert converted["converted_linear"] == 1
    with torch.inference_mode():
        output = model(indices)
    assert output.shape == reference.shape and torch.isfinite(output).all()
    assert torch.max(torch.abs(output - reference)) < 0.15
    after = tensor_bytes(model)
    assert after["packed_weight_and_bias_bytes"] > 0
    assert (
        after["estimated_parameter_tensor_bytes"] < before["estimated_parameter_tensor_bytes"] / 2
    )
    with pytest.raises(ValueError, match="word-embedding"):
        apply_profile(model, profile)


def test_float_profile_is_unchanged_and_unknown_profiles_fail_closed():
    model = EncoderFixture().eval()
    original = model.embeddings.word_embeddings.weight.detach().clone()
    assert apply_profile(model, "float32")["converted_embedding"] == 0
    assert torch.equal(model.embeddings.word_embeddings.weight, original)
    assert tensor_bytes(None) is None
    for profile in ("int4", None, ["float32"]):
        with pytest.raises(ValueError, match="inference profile"):
            validate_profile(profile)
