import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from controls import RandomProjectionMethod

from quantum_adapter import QuantumAdapterConfig


def test_control_matches_param_count_and_starts_neutral():
    m = RandomProjectionMethod(12, 20, QuantumAdapterConfig(target_modules=["x"], rank=2), n_params=61)
    assert sum(p.numel() for p in m.parameters()) == 61
    x = torch.randn(3, 12)
    assert torch.count_nonzero(m(x)) == 0
    assert torch.allclose(m(x), x @ m.delta_weight())
