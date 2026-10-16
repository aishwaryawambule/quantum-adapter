"""Wraps one nn.Linear or transformers Conv1D layer with an adapter method."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers.pytorch_utils import Conv1D

from quantum_adapter.methods import AdapterMethod

SUPPORTED_LAYERS = (nn.Linear, Conv1D)


def layer_shape(module: nn.Module) -> tuple[int, int]:
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features
    if isinstance(module, Conv1D):          # y = x @ W + b, W stored as (in, out)
        return module.weight.shape[0], module.weight.shape[1]
    raise TypeError(f"{type(module).__name__} is not supported; use nn.Linear or transformers Conv1D layers.")


class AdapterLayer(nn.Module):
    def __init__(self, base: nn.Module, adapter: AdapterMethod) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter

    def forward(self, x: Tensor) -> Tensor:
        y = self.base(x)
        return y + self.adapter(x).to(y.dtype)

    @torch.no_grad()
    def merge(self) -> nn.Module:
        delta = self.adapter.delta_weight().to(self.base.weight.device, self.base.weight.dtype)
        self.base.weight += delta.T if isinstance(self.base, nn.Linear) else delta
        return self.base
