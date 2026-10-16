"""Classical control: the same number of trainable values as a quantum method, LoRA-shaped A and B
produced by a fixed random projection instead of a circuit. If a quantum method can't beat this,
its gain comes from having few parameters, not from the circuit."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from quantum_adapter import QuantumAdapterConfig
from quantum_adapter.methods import AdapterMethod


class RandomProjectionMethod(AdapterMethod):
    name = "control"

    def __init__(self, in_features: int, out_features: int, config: QuantumAdapterConfig, n_params: int) -> None:
        super().__init__(in_features, out_features, config)
        self.rank = config.rank
        need = self.rank * (in_features + out_features)
        self.theta = nn.Parameter(torch.randn(n_params - 1))
        self.scale = nn.Parameter(torch.zeros(()))
        g = torch.Generator().manual_seed(123)
        self.register_buffer("proj", torch.randn(n_params - 1, need, generator=g) / math.sqrt(n_params - 1),
                             persistent=False)

    def factors(self) -> tuple[Tensor, Tensor]:
        v = self.theta @ self.proj
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
        p = sum(q.numel() for q in self.parameters())
        return {"method": self.name, "qubits": "-", "dials": p, "gates": 0, "params": p}
