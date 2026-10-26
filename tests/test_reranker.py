import torch

from category_matcher.reranker import align_cpu_parameters


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
