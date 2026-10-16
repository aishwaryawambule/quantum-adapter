"""Amplitude method: a circuit's state-vector amplitudes become LoRA-shaped factors A and B."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from quantum_adapter.config import QuantumAdapterConfig, too_many_qubits_message
from quantum_adapter.methods.base import AdapterMethod, CircuitSpec
from quantum_adapter.sim import entangling_pairs, zero_state


class AmplitudeMethod(AdapterMethod):
    name = "amplitude"
    quantum_gates = True

    def __init__(self, in_features: int, out_features: int, config: QuantumAdapterConfig) -> None:
        super().__init__(in_features, out_features, config)
        self.rank, self.depth = config.rank, config.depth
        self.need = self.rank * (in_features + out_features)
        minimum = max(1, math.ceil(math.log2(self.need)))
        n = minimum if config.qubits == "auto" else config.qubits
        if n < minimum:
            raise ValueError(
                f"A rank-{self.rank} amplitude adapter on a {in_features}x{out_features} layer needs "
                f"{self.need:,} amplitudes, but {n} qubits only give {2**n:,}. "
                f"Use qubits >= {minimum}, or qubits='auto'."
            )
        if n > config.max_qubits:
            raise ValueError(too_many_qubits_message(n, config.max_qubits))
        self.n_qubits = n
        self.theta = nn.Parameter(torch.rand(self.depth, n) * 2 * math.pi)
        self.scale = nn.Parameter(torch.zeros(()))   # multiplies B only: zero output, non-zero gradient

    def factors(self) -> tuple[Tensor, Tensor]:
        def compute() -> Tensor:
            amps = self.simulate(zero_state(1, self.n_qubits), self.theta, self.n_qubits, "cx", False)[0]
            return (amps[: self.need] * math.sqrt(2**self.n_qubits)).to(self.theta.device)

        v = self._cached(compute)
        split = self.in_features * self.rank
        a = v[:split].view(self.in_features, self.rank) / math.sqrt(self.in_features)
        b = v[split:].view(self.rank, self.out_features) * (self.scale / math.sqrt(self.rank))
        return a, b

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.factors()
        return (x @ a.to(x.dtype)) @ b.to(x.dtype)

    def delta_weight(self) -> Tensor:
        a, b = self.factors()
        return a @ b

    def summary(self) -> dict:
        pairs = len(entangling_pairs(self.n_qubits, self.entanglement))
        return {
            "method": self.name,
            "qubits": self.n_qubits,
            "dials": self.theta.numel(),
            "gates": self.depth * (self.n_qubits + pairs),
            "params": sum(p.numel() for p in self.parameters()),
        }

    def circuits(self) -> list[CircuitSpec]:
        return [CircuitSpec("amplitude", self.n_qubits, self.theta.detach().cpu(), self.entanglement, "cx", False)]
