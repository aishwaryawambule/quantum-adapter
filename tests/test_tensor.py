import copy

import pytest
import torch

from quantum_adapter import QuantumAdapterConfig
from quantum_adapter.methods import METHOD_CLASSES
from quantum_adapter.methods.tensor import balanced_factors


@pytest.mark.parametrize(
    "d, k, expected",
    [(12, 2, [3, 4]), (20, 2, [4, 5]), (12, 3, [2, 2, 3]), (20, 3, [2, 2, 5]), (768, 4, [6, 4, 4, 8])],
)
def test_balanced_factors(d, k, expected):
    assert balanced_factors(d, k) == expected


def test_prime_size_is_rejected():
    cfg = QuantumAdapterConfig(target_modules=["x"], method="tensor", tensor_axes=2)
    with pytest.raises(ValueError, match="can't split 13 into 2 factors larger than 1"):
        METHOD_CLASSES["tensor"](13, 20, cfg)


def test_gate_shapes_chain_from_input_to_output():
    cfg = QuantumAdapterConfig(target_modules=["x"], method="tensor", tensor_axes=3)
    m = METHOD_CLASSES["tensor"](12, 20, cfg)
    assert m.in_dims == [2, 2, 3] and m.out_dims == [2, 2, 5]
    assert [tuple(g.shape) for g in m.gates] == [(2, 2, 2, 2), (2, 3, 2, 5), (2, 5, 2, 5)]


def test_frozen_copy_is_saved_but_not_trained():
    cfg = QuantumAdapterConfig(target_modules=["x"], method="tensor", tensor_axes=3)
    m = METHOD_CLASSES["tensor"](12, 20, cfg)
    assert {n for n, _ in m.named_buffers()} == {"frozen_0", "frozen_1", "frozen_2"}
    assert all(not b.requires_grad for b in m.buffers())
    assert "frozen_0" in m.state_dict()


@pytest.mark.parametrize("convert", [lambda m: m.double(), lambda m: m.to("cpu")])
def test_survives_dtype_and_device_moves(convert):
    cfg = QuantumAdapterConfig(target_modules=["x"], method="tensor", tensor_axes=3)
    m = convert(METHOD_CLASSES["tensor"](12, 20, cfg))
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.1 * torch.randn_like(p))
    x = torch.randn(4, 12, dtype=next(m.parameters()).dtype)
    assert torch.allclose(m(x), x @ m.delta_weight(), atol=1e-5)


def _perturbed(d=96):
    torch.manual_seed(0)
    m = METHOD_CLASSES["tensor"](d, d, QuantumAdapterConfig(target_modules=["x"], method="tensor"))
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1e-3 * torch.randn_like(p))  # an early-training step: T(x) and S(x) nearly cancel
    return m


def _rel_err(got, ref):
    return ((got.float() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_input_matches_fp32(dtype):
    m = _perturbed()
    x = torch.randn(8, 96).to(dtype)
    y = m(x)
    assert y.dtype == dtype
    assert _rel_err(y, m(x.float())) < 1e-2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_module_matches_fp32(dtype):
    mh = _perturbed().to(dtype)
    x = torch.randn(8, 96).to(dtype)
    ref = copy.deepcopy(mh).float()(x.float())
    assert _rel_err(mh(x), ref) < 1e-2
    assert _rel_err(x @ mh.delta_weight(), ref) < 1e-2


def test_autocast_does_not_lower_precision():
    m = _perturbed()
    x = torch.randn(8, 96)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = m(x)
    assert _rel_err(y, m(x)) < 1e-2


def _cpu_autocast_enabled() -> bool:
    try:
        return torch.is_autocast_enabled("cpu")
    except TypeError:  # torch < 2.4 takes no device argument
        return torch.is_autocast_cpu_enabled()


def _mps_autocast_supported() -> bool:
    try:
        torch.autocast("mps", dtype=torch.float16)
    except RuntimeError:  # older torch (e.g. 2.3) has no autocast for MPS
        return False
    return True


@pytest.mark.skipif(not torch.backends.mps.is_available() or not _mps_autocast_supported(),
                    reason="needs MPS with autocast support")
def test_tensor_is_unchanged_by_mps_autocast():
    cfg = QuantumAdapterConfig(target_modules=["x"], method="tensor", tensor_axes=3)
    m = METHOD_CLASSES["tensor"](12, 20, cfg).to("mps")
    with torch.no_grad():
        for g in m.gates:
            g.add_(0.3 * torch.randn_like(g))
        x = torch.randn(4, 12, device="mps")
        ref = m(x)
        with torch.autocast("mps", dtype=torch.float16):
            out = m(x)
    assert ((out.float() - ref).norm() / ref.norm()).item() < 1e-2


def test_no_autocast_disables_autocast_for_any_device_type():
    from quantum_adapter.methods.tensor import _no_autocast

    with _no_autocast("cpu"):
        assert not _cpu_autocast_enabled()
    with torch.autocast("cpu", dtype=torch.bfloat16), _no_autocast("cpu"):
        assert not _cpu_autocast_enabled()
    with _no_autocast("not-a-device"):  # unsupported device types fall back to doing nothing
        pass
