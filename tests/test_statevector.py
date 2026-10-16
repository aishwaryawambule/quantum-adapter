import math

import pytest
import torch

from quantum_adapter.sim import (
    apply_cnot,
    apply_cz,
    apply_ry,
    basis_states,
    entangling_pairs,
    run_ansatz,
    zero_state,
)

S = 1 / math.sqrt(2)


def test_ry_on_zero():
    out = apply_ry(zero_state(1, 1), 0, torch.tensor(1.0), 1)
    assert torch.allclose(out, torch.tensor([[math.cos(0.5), math.sin(0.5)]]))


def test_qubit_zero_is_most_significant_bit():
    out = apply_ry(zero_state(1, 2), 0, torch.tensor(math.pi), 2)   # |00> -> |10> = index 2
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, 1.0, 0.0]]), atol=1e-6)


def test_bell_state():
    s = apply_cnot(apply_ry(zero_state(1, 2), 0, torch.tensor(math.pi / 2), 2), 0, 1, 2)
    assert torch.allclose(s, torch.tensor([[S, 0.0, 0.0, S]]), atol=1e-6)


def test_cnot_respects_control_and_target():
    s = basis_states(4, 2)                      # rows |00>, |01>, |10>, |11>
    out = apply_cnot(s, 1, 0, 2)                # control q1, target q0: |01> -> |11>, |11> -> |01>
    assert out.argmax(1).tolist() == [0, 3, 2, 1]


def test_cz_flips_sign_of_11_only():
    s = apply_ry(apply_ry(zero_state(1, 2), 0, torch.tensor(math.pi / 2), 2), 1, torch.tensor(math.pi / 2), 2)
    assert torch.allclose(apply_cz(s, 0, 1, 2), torch.tensor([[0.5, 0.5, 0.5, -0.5]]), atol=1e-6)


def test_entangling_pairs():
    assert entangling_pairs(1, "ring") == []
    assert entangling_pairs(2, "ring") == [(0, 1)]
    assert entangling_pairs(3, "ring") == [(0, 1), (1, 2), (2, 0)]
    assert entangling_pairs(3, "line") == [(0, 1), (1, 2)]
    assert entangling_pairs(3, "full") == [(0, 1), (0, 2), (1, 2)]
    with pytest.raises(ValueError):
        entangling_pairs(3, "star")


@pytest.mark.parametrize("pattern", ["ring", "line", "full"])
@pytest.mark.parametrize("gate", ["cx", "cz"])
@pytest.mark.parametrize("final", [False, True])
def test_norm_is_preserved(pattern, gate, final):
    theta = torch.rand(3, 6) * 2 * math.pi
    out = run_ansatz(basis_states(5, 6), theta, 6, entanglement=pattern, gate=gate, final_rotation=final)
    assert torch.allclose(out.norm(dim=1), torch.ones(5), atol=1e-5)


def test_final_rotation_skips_last_entangler():
    theta = torch.zeros(1, 2)                     # identity rotations
    s = basis_states(4, 2)
    assert torch.allclose(run_ansatz(s, theta, 2, gate="cx", final_rotation=True), s)
    assert not torch.allclose(run_ansatz(s, theta, 2, gate="cx", final_rotation=False), s)


def test_gradients_match_finite_differences():
    theta = (torch.rand(2, 3, dtype=torch.float64) * 2 * math.pi).requires_grad_()
    start = zero_state(1, 3, dtype=torch.float64)
    weights = torch.randn(8, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda t: (run_ansatz(start, t, 3) * weights).sum(), (theta,))


def test_rejects_wrong_theta_width_and_gate():
    with pytest.raises(ValueError, match="theta has 4 columns but n=3"):
        run_ansatz(zero_state(1, 3), torch.zeros(2, 4), 3)
    with pytest.raises(ValueError, match="gate must be"):
        run_ansatz(zero_state(1, 3), torch.zeros(2, 3), 3, gate="swap")


def test_basis_states_match_identity_rows():
    assert torch.equal(basis_states(3, 4), torch.eye(16)[:3])
    assert basis_states(2, 3, dtype=torch.float64).dtype == torch.float64


def test_basis_states_do_not_build_the_full_identity():
    # The square identity at 20 qubits would be 2^40 floats (4 TiB); the k rows are 16 MiB.
    s = basis_states(4, 20)
    assert s.shape == (4, 2**20) and s.sum() == 4 and all(s[j, j] == 1 for j in range(4))


def test_index_caches_are_bounded():
    from quantum_adapter.sim.statevector import _cnot_index, _cz_sign

    assert _cnot_index.cache_info().maxsize is not None and _cz_sign.cache_info().maxsize is not None
