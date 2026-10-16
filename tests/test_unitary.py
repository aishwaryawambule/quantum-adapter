import pytest
import torch

from quantum_adapter import QuantumAdapterConfig
from quantum_adapter.methods import METHOD_CLASSES


def make(i, o, **kw):
    return METHOD_CLASSES["unitary"](i, o, QuantumAdapterConfig(target_modules=["x"], method="unitary", **kw))


@pytest.mark.parametrize("shape", [(16, 32), (12, 20), (768, 2304)])
def test_frames_are_orthonormal(shape):
    m = make(*shape)
    u, v = m.frames()
    assert u.shape == (shape[0], 4) and v.shape == (shape[1], 4)
    assert torch.allclose(u.T @ u, torch.eye(4), atol=1e-4)
    assert torch.allclose(v.T @ v, torch.eye(4), atol=1e-4)


def test_parameter_count_for_gpt2_attention():
    m = make(768, 2304)
    assert (m.n_in, m.n_out) == (10, 12)
    assert sum(p.numel() for p in m.parameters()) == 11 * (10 + 12) + 4   # 246


def test_rank_larger_than_layer_is_rejected():
    with pytest.raises(ValueError, match="rank=8 is larger than the layer's smaller side \\(6\\)"):
        make(6, 20, rank=8)


def test_qubit_limit():
    with pytest.raises(ValueError, match="max_qubits=8"):
        make(1024, 1024, max_qubits=8)
