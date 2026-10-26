"""Explicit CPU inference profiles; conversion never changes acceptance policy."""

PROFILES = {"float32", "int8_embeddings_v1", "int8_dynamic_v1"}


def validate_profile(profile: str) -> None:
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError("Unknown CPU inference profile")


def apply_profile(module, profile: str) -> dict:
    import torch
    from torch.ao.quantization import (
        default_dynamic_qconfig,
        float_qparams_weight_only_qconfig,
        quantize_dynamic,
    )

    validate_profile(profile)
    if profile == "float32":
        return {"profile": profile, "converted_linear": 0, "converted_embedding": 0}
    if "qnnpack" not in torch.backends.quantized.supported_engines:
        raise ValueError("This inference profile requires a PyTorch build with QNNPACK")
    names = [
        name
        for name, child in module.named_modules()
        if isinstance(child, torch.nn.Embedding) and name.endswith("embeddings.word_embeddings")
    ]
    if len(names) != 1:
        raise ValueError("Expected exactly one supported word-embedding module")
    # DeBERTa's encoder reads rel_embeddings.weight directly as a float tensor.
    # Convert only lookup-based word embeddings, leaving relative positions intact.
    config = {names[0]: float_qparams_weight_only_qconfig}
    if profile == "int8_dynamic_v1":
        config[torch.nn.Linear] = default_dynamic_qconfig
    torch.backends.quantized.engine = "qnnpack"
    module.eval()
    with torch.no_grad():
        quantize_dynamic(module, qconfig_spec=config, inplace=True)
    from torch.ao.nn.quantized import Embedding
    from torch.ao.nn.quantized.dynamic import Linear

    counts = {
        "profile": profile,
        "backend": "qnnpack",
        "converted_linear": sum(isinstance(child, Linear) for child in module.modules()),
        "converted_embedding": sum(isinstance(child, Embedding) for child in module.modules()),
    }
    if counts["converted_embedding"] != 1:
        raise ValueError("Word-embedding conversion did not produce the expected quantized module")
    return counts


def tensor_bytes(module) -> dict | None:
    """Account for packed quantized weights as well as remaining float tensors.

    This estimates tensor storage, not process memory or opaque kernel caches.
    Unpacking is diagnostic work and must stay outside timed inference requests.
    """
    if module is None or not hasattr(module, "parameters"):
        return None
    import torch
    from torch.ao.nn.quantized import Embedding
    from torch.ao.nn.quantized.dynamic import Linear

    floating = sum(p.numel() * p.element_size() for p in module.parameters())
    packed = metadata = 0
    for child in module.modules():
        if not isinstance(child, (Linear, Embedding)):
            continue
        weight = child.weight()
        packed += weight.numel() * weight.element_size()
        if weight.qscheme() in (torch.per_channel_affine, torch.per_channel_affine_float_qparams):
            for value in (weight.q_per_channel_scales(), weight.q_per_channel_zero_points()):
                metadata += value.numel() * value.element_size()
        else:
            metadata += 16  # scalar scale and zero point
        if isinstance(child, Linear) and child.bias() is not None:
            bias = child.bias()
            packed += bias.numel() * bias.element_size()
    return {
        "floating_parameter_bytes": floating,
        "packed_weight_and_bias_bytes": packed,
        "quantization_metadata_bytes": metadata,
        "estimated_parameter_tensor_bytes": floating + packed + metadata,
    }
