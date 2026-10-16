"""Tensor method (QuanTA, arXiv 2406.00132): the hidden vector's axes act like qudits and trainable
tensors act on pairs of axes, like two-qubit gates. Quantum-inspired: the gates are not unitary."""

from __future__ import annotations

import contextlib
import math

import torch
from torch import Tensor, nn

from quantum_adapter.config import QuantumAdapterConfig
from quantum_adapter.methods.base import AdapterMethod

_AXES = "abcdefgh"


def _no_autocast(device_type: str):
    try:
        return torch.autocast(device_type, enabled=False)
    except RuntimeError:  # e.g. torch 2.3 has no autocast for MPS: nothing to switch off
        return contextlib.nullcontext()


def balanced_factors(d: int, k: int) -> list[int]:
    """Split d into k integer factors as close to equal as possible; their product is d."""
    factors, rest = [], d
    for left in range(k, 1, -1):
        target = rest ** (1 / left)
        divisors = [f for f in range(1, rest + 1) if rest % f == 0]
        f = min(divisors, key=lambda f: (abs(f - target), f))
        factors.append(f)
        rest //= f
    factors.append(rest)
    return factors


class TensorMethod(AdapterMethod):
    name = "tensor"
    quantum_gates = False

    def __init__(self, in_features: int, out_features: int, config: QuantumAdapterConfig) -> None:
        super().__init__(in_features, out_features, config)
        k = config.tensor_axes
        self.in_dims, self.out_dims = balanced_factors(in_features, k), balanced_factors(out_features, k)
        for d, dims in ((in_features, self.in_dims), (out_features, self.out_dims)):
            if 1 in dims:
                raise ValueError(f"can't split {d} into {k} factors larger than 1; lower tensor_axes.")
        self.pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
        cur, shapes = list(self.in_dims), []
        for a, b in self.pairs:
            shapes.append((cur[a], cur[b], self.out_dims[a], self.out_dims[b]))
            cur[a], cur[b] = self.out_dims[a], self.out_dims[b]
        self.gates = nn.ParameterList(nn.Parameter(torch.randn(s) / math.sqrt(s[0] * s[1])) for s in shapes)
        for i, g in enumerate(self.gates):
            self.register_buffer(f"frozen_{i}", g.detach().clone())   # y = T(x) − S(x): zero at start

    def _frozen(self) -> list[Tensor]:
        return [getattr(self, f"frozen_{i}") for i in range(len(self.pairs))]

    def _contract(self, gates: list[Tensor], x: Tensor) -> Tensor:
        lead = x.shape[:-1]
        t = x.reshape(-1, *self.in_dims)
        axes = _AXES[: len(self.in_dims)]
        for (a, b), g in zip(self.pairs, gates):
            src = "z" + axes
            dst = src.replace(axes[a], "x").replace(axes[b], "y")
            t = torch.einsum(f"{src},{axes[a]}{axes[b]}xy->{dst}", t, g.to(t.dtype))
        return t.reshape(*lead, self.out_features)

    def forward(self, x: Tensor) -> Tensor:
        # T(x) − S(x) is a small difference of two large terms, so half precision (or autocast) would cancel
        # it away. Contract in at least float32, then cast back; float32 and float64 inputs are unchanged.
        out_dtype = x.dtype
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        with _no_autocast(x.device.type):
            y = self._contract(list(self.gates), x) - self._contract(self._frozen(), x)
        return y.to(out_dtype)

    def delta_weight(self) -> Tensor:
        g = self.gates[0]
        return self.forward(torch.eye(self.in_features, dtype=g.dtype, device=g.device))

    def summary(self) -> dict:
        params = sum(p.numel() for p in self.parameters())
        return {"method": self.name, "qubits": "-", "dials": params, "gates": len(self.pairs), "params": params}
