from __future__ import annotations

from quantum_adapter.methods.amplitude import AmplitudeMethod
from quantum_adapter.methods.base import AdapterMethod, CircuitSpec
from quantum_adapter.methods.tensor import TensorMethod
from quantum_adapter.methods.unitary import UnitaryMethod

METHOD_CLASSES: dict[str, type[AdapterMethod]] = {
    "amplitude": AmplitudeMethod,
    "unitary": UnitaryMethod,
    "tensor": TensorMethod,
}

# Methods that need experimental=True (see docs/benchmarks.md): unitary was about 3 points behind LoRA
# at 3 epochs on SST-2; tensor trains more parameters than LoRA r=4 on GPT-2-sized layers, so it
# cannot meet the 5x-fewer-parameters rule.
EXPERIMENTAL: frozenset[str] = frozenset({"unitary", "tensor"})

__all__ = [
    "EXPERIMENTAL",
    "METHOD_CLASSES",
    "AdapterMethod",
    "AmplitudeMethod",
    "CircuitSpec",
    "TensorMethod",
    "UnitaryMethod",
]
