"""Shared base for adapter methods: one instance computes the correction for one layer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from quantum_adapter.config import QuantumAdapterConfig
from quantum_adapter.sim import run_ansatz


@dataclass(frozen=True)
class CircuitSpec:
    name: str
    n_qubits: int
    theta: Tensor
    entanglement: str
    gate: str
    final_rotation: bool


class AdapterMethod(nn.Module):
    """forward(x) returns x @ delta_weight(), shape (..., out_features)."""

    name: str = ""
    quantum_gates: bool = False

    def __init__(self, in_features: int, out_features: int, config: QuantumAdapterConfig) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.entanglement = config.entanglement
        self.simulator_device = torch.device(config.simulator_device)
        self._cache_key: tuple | None = None
        self._cache = None
        self._cache_values: list[Tensor] = []

    def delta_weight(self) -> Tensor:
        raise NotImplementedError

    def summary(self) -> dict:
        raise NotImplementedError

    def circuits(self) -> list[CircuitSpec]:
        return []

    def simulate(self, initial: Tensor, theta: Tensor, n: int, gate: str, final_rotation: bool) -> Tensor:
        dtype = torch.float64 if theta.dtype == torch.float64 else torch.float32
        dev = self.simulator_device
        return run_ansatz(
            initial.to(dev, dtype), theta.to(dev, dtype), n,
            entanglement=self.entanglement, gate=gate, final_rotation=final_rotation,
        )

    def _cached(self, compute: Callable):
        """Recompute under autograd (fresh graph per forward); reuse under no_grad until a parameter changes.

        Edits through `p.data` do not bump the parameter's version counter, so the key also holds a copy of
        the (small) parameters and compares their values.
        """
        if torch.is_grad_enabled():
            return compute()
        params = list(self.parameters())
        key = tuple((p.data_ptr(), p._version, p.dtype) for p in params)
        if key != self._cache_key or not all(torch.equal(p, s) for p, s in zip(params, self._cache_values)):
            self._cache, self._cache_key = compute(), key
            self._cache_values = [p.detach().clone() for p in params]
        return self._cache
