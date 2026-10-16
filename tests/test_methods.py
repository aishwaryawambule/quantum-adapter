import math

import pytest
import torch
from torch.func import functional_call

from quantum_adapter import QuantumAdapterConfig
from quantum_adapter.methods import METHOD_CLASSES

IN, OUT = 12, 20
METHODS = ["amplitude", "unitary", "tensor"]


def make(method: str, **kw):
    cfg = QuantumAdapterConfig(target_modules=["x"], method=method, rank=2, depth=3, tensor_axes=3, **kw)
    return METHOD_CLASSES[method](IN, OUT, cfg)


def perturb(m):
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.3 * torch.randn_like(p))


@pytest.mark.parametrize("method", METHODS)
def test_neutral_start(method):
    m = make(method)
    x = torch.randn(5, IN)
    assert torch.count_nonzero(m(x)) == 0
    assert torch.count_nonzero(m.delta_weight()) == 0


@pytest.mark.parametrize("method", METHODS)
def test_forward_equals_x_times_delta_weight(method):
    m = make(method)
    perturb(m)
    x = torch.randn(3, 4, IN)
    assert m(x).shape == (3, 4, OUT)
    assert torch.allclose(m(x), x @ m.delta_weight(), atol=1e-5)


@pytest.mark.parametrize("method", METHODS)
def test_every_parameter_learns_after_one_step(method):
    m = make(method)
    x, target = torch.randn(8, IN), torch.randn(8, OUT)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    for _ in range(2):
        opt.zero_grad()
        ((m(x) - target) ** 2).mean().backward()
        opt.step()
    dead = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert dead == [], f"no gradient reached {dead}"


@pytest.mark.parametrize("method", METHODS)
def test_learns_a_linear_map(method):
    torch.manual_seed(0)
    m = make(method)
    x = torch.randn(64, IN)
    target = x @ (0.3 * torch.randn(IN, OUT))
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    first = None
    for _ in range(40):
        opt.zero_grad()
        loss = ((m(x) - target) ** 2).mean()
        first = first if first is not None else loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < 0.9 * first


@pytest.mark.parametrize("method", METHODS)
def test_gradients_match_finite_differences(method):
    m = make(method).double()
    perturb(m)
    names = [n for n, _ in m.named_parameters()]
    x = torch.randn(3, IN, dtype=torch.float64)

    def f(*params):
        return functional_call(m, dict(zip(names, params)), (x,))

    assert torch.autograd.gradcheck(f, tuple(p.detach().requires_grad_() for p in m.parameters()))


@pytest.mark.parametrize("method", METHODS)
def test_two_forwards_one_backward(method):
    m = make(method)
    perturb(m)
    x = torch.randn(4, IN)
    (m(x).sum() + m(x).sum()).backward()        # gradient accumulation pattern must not raise


@pytest.mark.parametrize("method", METHODS)
def test_eval_cache_invalidates_after_step(method):
    m = make(method)
    perturb(m)
    x = torch.randn(4, IN)
    with torch.no_grad():
        before = m(x).clone()
        assert torch.equal(m(x), before)        # cached, same result
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    opt.zero_grad()
    m(x).pow(2).sum().backward()
    opt.step()
    with torch.no_grad():
        assert not torch.allclose(m(x), before)


@pytest.mark.parametrize("method", METHODS)
def test_summary_keys(method):
    s = make(method).summary()
    assert set(s) == {"method", "qubits", "dials", "gates", "params"}
    assert s["params"] == sum(p.numel() for p in make(method).parameters())


def test_amplitude_sizes_for_gpt2_attention():
    cfg = QuantumAdapterConfig(target_modules=["c_attn"])
    m = METHOD_CLASSES["amplitude"](768, 2304, cfg)
    assert m.n_qubits == 14                       # 4 * (768 + 2304) = 12,288 <= 2**14
    assert sum(p.numel() for p in m.parameters()) == 141
    assert m.summary()["gates"] == 10 * (14 + 14)


def test_amplitude_rejects_too_few_qubits():
    cfg = QuantumAdapterConfig(target_modules=["c_attn"], qubits=13)
    with pytest.raises(ValueError, match="needs 12,288 amplitudes, but 13 qubits only give 8,192. Use qubits >= 14"):
        METHOD_CLASSES["amplitude"](768, 2304, cfg)


def test_amplitude_rejects_too_many_auto_qubits():
    cfg = QuantumAdapterConfig(target_modules=["x"], rank=64, max_qubits=16)
    with pytest.raises(ValueError, match="max_qubits=16"):
        METHOD_CLASSES["amplitude"](4096, 4096, cfg)


def test_scaling_both_factors_would_never_learn():
    """Pins the bug from the first experiment: scale must multiply B only."""
    m = make("amplitude")
    a, b = m.factors()
    assert torch.count_nonzero(a) > 0 and torch.count_nonzero(b) == 0
    assert math.isclose(m.scale.item(), 0.0)


@pytest.mark.parametrize("method", METHODS)
def test_no_grad_cache_sees_in_place_edits_through_data(method):
    m = make(method)
    perturb(m)
    with torch.no_grad():
        before = m.delta_weight().clone()
        next(m.parameters()).data.add_(1.0)  # bypasses the parameter's own version counter
        assert not torch.allclose(m.delta_weight(), before)
