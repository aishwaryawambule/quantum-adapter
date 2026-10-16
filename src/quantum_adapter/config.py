"""Adapter settings, validated on creation and stored as JSON next to the weights."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

METHODS = ("amplitude", "unitary", "tensor")
ENTANGLEMENTS = ("ring", "line", "full")


def too_many_qubits_message(n: int, limit: int) -> str:
    state_mib = 2**n * 4 / 2**20
    return (
        f"{n} qubits means a state of {2**n:,} numbers per circuit ({state_mib:,.0f} MiB, several times that "
        f"during training), above max_qubits={limit}. Lower rank, adapt smaller layers, or raise max_qubits "
        "if you have the memory."
    )


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


@dataclass
class QuantumAdapterConfig:
    """Settings for get_quantum_model, validated on creation and saved as adapter_config.json.

    Attributes:
        target_modules: layers to adapt: a list of names, each matching a module's full name or its last
            parts (["c_attn"] matches "transformer.h.0.attn.c_attn"), or one regex matched against full names.
            Matched layers must be nn.Linear or transformers Conv1D.
        method: "amplitude", "unitary" or "tensor" (see the README's method table).
        rank: rank of ΔW for amplitude and unitary (ignored by tensor).
        qubits: amplitude only: circuit width, or "auto" for the fewest qubits giving rank·(in+out) amplitudes.
            unitary always uses ⌈log2 in⌉ and ⌈log2 out⌉ qubits.
        depth: number of RY + entangler blocks per circuit (amplitude and unitary).
        entanglement: which qubit pairs each entangling layer links: "ring", "line" or "full".
        tensor_axes: tensor only: how many axes each layer's input and output sizes are split into (2 to 8).
        modules_to_save: extra modules (same name matching as target_modules) trained in full and saved with
            the adapter, e.g. ["score"] or ["classifier"] for a classification head. They may not be or
            contain a target layer.
        max_qubits: refuse circuits wider than this (the simulator holds 2**qubits numbers per circuit).
        simulator_device: device the circuit simulation runs on (default "cpu", usually fastest).
        experimental: must be True to use a method that has not passed the release benchmark.
        base_model_name_or_path: recorded in the saved config; filled from model.config.name_or_path if None.
    """

    target_modules: list[str] | str
    method: str = "amplitude"
    rank: int = 4
    qubits: int | str = "auto"
    depth: int = 10
    entanglement: str = "ring"
    tensor_axes: int = 4
    modules_to_save: list[str] = field(default_factory=list)
    max_qubits: int = 20
    simulator_device: str = "cpu"
    experimental: bool = False
    base_model_name_or_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target_modules, str):
            self.target_modules = list(self.target_modules)
        if not self.target_modules:
            raise ValueError(
                "target_modules must name at least one layer, e.g. ['c_attn'] for GPT-2 "
                "or ['q_proj', 'v_proj'] for Llama-style models."
            )
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}.")
        if self.entanglement not in ENTANGLEMENTS:
            raise ValueError(f"entanglement must be one of {ENTANGLEMENTS}, got {self.entanglement!r}.")
        for name in ("rank", "depth", "max_qubits"):
            if not _positive_int(getattr(self, name)):
                raise ValueError(f"{name} must be a positive integer, got {getattr(self, name)!r}.")
        if not (isinstance(self.tensor_axes, int) and 2 <= self.tensor_axes <= 8):
            raise ValueError(f"tensor_axes must be between 2 and 8, got {self.tensor_axes!r}.")
        if self.qubits != "auto":
            if not _positive_int(self.qubits):
                raise ValueError(f"qubits must be 'auto' or a positive integer, got {self.qubits!r}.")
            if self.qubits > self.max_qubits:
                raise ValueError(too_many_qubits_message(self.qubits, self.max_qubits))
        self.modules_to_save = list(self.modules_to_save)

    def to_dict(self) -> dict:
        """The settings as a JSON-serialisable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> QuantumAdapterConfig:
        """Rebuild a config from to_dict output; unknown keys raise (they mean a newer library saved it)."""
        data = dict(data)
        data.pop("library_version", None)
        unknown = sorted(set(data) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(
                f"Unknown config keys {unknown}. This adapter may have been saved by a newer quantum-adapter; "
                "upgrade with `pip install -U quantum-adapter`."
            )
        return cls(**data)
