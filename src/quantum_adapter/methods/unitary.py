"""Unitary method (Quantum-PEFT, arXiv 2503.05431): ΔW = U diag(λ) Vᵀ, with U and V taken from RY + CZ circuits."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from quantum_adapter.config import QuantumAdapterConfig, too_many_qubits_message
from quantum_adapter.methods.base import AdapterMethod, CircuitSpec
from quantum_adapter.sim import basis_states, entangling_pairs


class UnitaryMethod(AdapterMethod):
    name = "unitary"
    quantum_gates = True

    def __init__(self, in_features: int, out_features: int, config: QuantumAdapterConfig) -> None:
        super().__init__(in_features, out_features, config)
        self.rank, self.depth = config.rank, config.depth
        smaller = min(in_features, out_features)
        if self.rank > smaller:
            raise ValueError(
                f"rank={self.rank} is larger than the layer's smaller side ({smaller}); "
                f"a unitary adapter's rank must be <= {smaller}."
            )
        self.n_in = max(1, math.ceil(math.log2(in_features)))
        self.n_out = max(1, math.ceil(math.log2(out_features)))
        for n in (self.n_in, self.n_out):
            if n > config.max_qubits:
                raise ValueError(too_many_qubits_message(n, config.max_qubits))
        self.theta_u = nn.Parameter(torch.rand(self.depth + 1, self.n_in) * 2 * math.pi)
        self.theta_v = nn.Parameter(torch.rand(self.depth + 1, self.n_out) * 2 * math.pi)
        self.lam = nn.Parameter(torch.zeros(self.rank))   # neutral start; U, V learn once λ moves

    def _frame(self, theta: Tensor, n: int, d: int) -> Tensor:
        # Row j of the output is the circuit applied to |j>, i.e. column j of the circuit's matrix.
        cols = self.simulate(basis_states(self.rank, n), theta, n, "cz", True)
        q, r = torch.linalg.qr(cols.T[:d])          # keep the first d rows, restore orthonormal columns
        sign = torch.sign(torch.diagonal(r))
        return q * torch.where(sign == 0, torch.ones_like(sign), sign)

    def frames(self) -> tuple[Tensor, Tensor]:
        def compute() -> tuple[Tensor, Tensor]:
            u = self._frame(self.theta_u, self.n_in, self.in_features)
            v = self._frame(self.theta_v, self.n_out, self.out_features)
            return u.to(self.lam.device), v.to(self.lam.device)

        return self._cached(compute)

    def forward(self, x: Tensor) -> Tensor:
        u, v = self.frames()
        return ((x @ u.to(x.dtype)) * self.lam.to(x.dtype)) @ v.T.to(x.dtype)

    def delta_weight(self) -> Tensor:
        u, v = self.frames()
        return (u * self.lam) @ v.T

    def summary(self) -> dict:
        pairs = len(entangling_pairs(self.n_in, self.entanglement)) + len(
            entangling_pairs(self.n_out, self.entanglement)
        )
        dials = self.theta_u.numel() + self.theta_v.numel()
        return {
            "method": self.name,
            "qubits": f"{self.n_in}+{self.n_out}",
            "dials": dials,
            "gates": dials + self.depth * pairs,
            "params": sum(p.numel() for p in self.parameters()),
        }

    def circuits(self) -> list[CircuitSpec]:
        return [
            CircuitSpec("U", self.n_in, self.theta_u.detach().cpu(), self.entanglement, "cz", True),
            CircuitSpec("V", self.n_out, self.theta_v.detach().cpu(), self.entanglement, "cz", True),
        ]
