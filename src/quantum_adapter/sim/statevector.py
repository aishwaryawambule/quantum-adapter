"""Batched real state-vector simulator for RY, CNOT and CZ circuits.

Qubit q is tensor axis q, most significant first: basis index = sum(bit_q << (n - 1 - q)).
Every gate here is real, so states stay real and autograd gives exact gradients.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor


def zero_state(batch: int, n: int, *, dtype: torch.dtype = torch.float32, device=None) -> Tensor:
    state = torch.zeros(batch, 2**n, dtype=dtype, device=device)
    state[:, 0] = 1.0
    return state


def basis_states(k: int, n: int, *, dtype: torch.dtype = torch.float32, device=None) -> Tensor:
    """The first k basis states |0>, |1>, ..., |k-1> as rows of shape (k, 2**n), without building the identity."""
    states = torch.zeros(k, 2**n, dtype=dtype, device=device)
    rows = torch.arange(k, device=device)
    states[rows, rows] = 1.0
    return states


def apply_ry(state: Tensor, q: int, angle: Tensor, n: int) -> Tensor:
    b = state.shape[0]
    s = state.reshape(b, 2**q, 2, 2 ** (n - q - 1))
    c, si = torch.cos(angle / 2), torch.sin(angle / 2)
    a0, a1 = s[:, :, 0], s[:, :, 1]
    return torch.stack((c * a0 - si * a1, si * a0 + c * a1), dim=2).reshape(b, -1)


@lru_cache(maxsize=256)
def _cnot_index(n: int, control: int, target: int, device: str) -> Tensor:
    idx = torch.arange(2**n)
    ctrl = (idx >> (n - 1 - control)) & 1
    return (idx ^ (ctrl << (n - 1 - target))).to(device)


def apply_cnot(state: Tensor, control: int, target: int, n: int) -> Tensor:
    return state[:, _cnot_index(n, control, target, str(state.device))]


@lru_cache(maxsize=256)
def _cz_sign(n: int, a: int, b: int, device: str, dtype: torch.dtype) -> Tensor:
    idx = torch.arange(2**n)
    both = ((idx >> (n - 1 - a)) & 1) & ((idx >> (n - 1 - b)) & 1)
    return (1 - 2 * both).to(device=device, dtype=dtype)


def apply_cz(state: Tensor, a: int, b: int, n: int) -> Tensor:
    return state * _cz_sign(n, a, b, str(state.device), state.dtype)


def entangling_pairs(n: int, pattern: str) -> list[tuple[int, int]]:
    if pattern not in ("ring", "line", "full"):
        raise ValueError(f"entanglement must be 'ring', 'line' or 'full', got {pattern!r}.")
    if n < 2:
        return []
    if pattern == "line" or n == 2:
        return [(q, q + 1) for q in range(n - 1)]
    if pattern == "ring":
        return [(q, (q + 1) % n) for q in range(n)]
    return [(a, b) for a in range(n) for b in range(a + 1, n)]


def run_ansatz(
    state: Tensor,
    theta: Tensor,
    n: int,
    *,
    entanglement: str = "ring",
    gate: str = "cx",
    final_rotation: bool = False,
) -> Tensor:
    """Apply RY layers theta[k] (one angle per qubit), each followed by an entangling layer.

    With final_rotation=True the last row of theta is a closing RY layer with no entangler after it.
    """
    if theta.shape[-1] != n:
        raise ValueError(f"theta has {theta.shape[-1]} columns but n={n} qubits.")
    if gate not in ("cx", "cz"):
        raise ValueError(f"gate must be 'cx' or 'cz', got {gate!r}.")
    entangle = apply_cnot if gate == "cx" else apply_cz
    pairs = entangling_pairs(n, entanglement)
    layers = theta.shape[0]
    for k in range(layers):
        for q in range(n):
            state = apply_ry(state, q, theta[k, q], n)
        if not (final_rotation and k == layers - 1):
            for a, b in pairs:
                state = entangle(state, a, b, n)
    return state
