"""Finds target layers, freezes the base model, and adds save / load / merge / hub."""

from __future__ import annotations

import dataclasses
import functools
import inspect
import json
import os
import re
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from quantum_adapter.config import QuantumAdapterConfig
from quantum_adapter.layers import SUPPORTED_LAYERS, AdapterLayer, layer_shape
from quantum_adapter.methods import EXPERIMENTAL, METHOD_CLASSES, AdapterMethod

CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "adapter_model.safetensors"
# Version of the saved files' layout and parameterisation. Bump it when a saved adapter would load wrongly;
# from_pretrained refuses versions newer than this one.
FORMAT_VERSION = 1


def _matches(name: str, targets: list[str] | str) -> bool:
    if isinstance(targets, str):
        return re.fullmatch(targets, name) is not None
    return any(name == t or name.endswith("." + t) for t in targets)


def _parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, child = name.rpartition(".")
    return (model.get_submodule(parent_name) if parent_name else model), child


def _target_modules(model: nn.Module, target_modules: list[str] | str) -> list[tuple[str, nn.Module]]:
    return [(n, m) for n, m in model.named_modules() if n and _matches(n, target_modules)]


def _reject_adapted(model: nn.Module) -> None:
    if any(isinstance(m, AdapterLayer) for m in model.modules()):
        raise ValueError(
            "This model already has quantum adapters. Call get_quantum_model on the original model, "
            "or load saved adapters with QuantumModel.from_pretrained onto a fresh base model."
        )


def inject_adapters(
    model: nn.Module, target_modules: list[str] | str, make_method: Callable[[int, int], AdapterMethod]
) -> list[str]:
    """Wrap every layer matching `target_modules` in an AdapterLayer; returns the wrapped layer names.

    All adapters are built before the model is touched, so a failure leaves it unchanged.
    """
    _reject_adapted(model)
    targets = _target_modules(model, target_modules)
    if not targets:
        available = sorted({n.rsplit(".", 1)[-1] for n, m in model.named_modules() if isinstance(m, SUPPORTED_LAYERS)})
        raise ValueError(
            f"No layers matched target_modules={target_modules!r}. "
            f"This model's adaptable layers include: {', '.join(available)}."
        )
    for name, module in targets:
        if not isinstance(module, SUPPORTED_LAYERS):
            raise TypeError(
                f"Layer {name!r} is a {type(module).__name__}; quantum adapters support "
                "nn.Linear and transformers Conv1D layers."
            )
    # Build every adapter before touching the model, so a failure leaves it unchanged.
    methods = [make_method(*layer_shape(m)).to(m.weight.device) for _, m in targets]
    for (name, module), method in zip(targets, methods):
        parent, child = _parent(model, name)
        setattr(parent, child, AdapterLayer(module, method))
    return [n for n, _ in targets]


def _planned_layer_shapes(model: nn.Module, target_modules: list[str] | str) -> dict[str, list[int]]:
    """Shapes of the layers get_quantum_model would adapt, computed without touching the model."""
    return {n: list(layer_shape(m)) for n, m in _target_modules(model, target_modules) if isinstance(m, SUPPORTED_LAYERS)}


def _restore(model: nn.Module, requires_grad: dict[int, bool], created: list[str]) -> None:
    """Undo get_quantum_model: unwrap the layers it created and put back the original requires_grad flags."""
    for name in created:
        layer = model.get_submodule(name)
        if isinstance(layer, AdapterLayer):
            parent, child = _parent(model, name)
            setattr(parent, child, layer.base)
    for p in model.parameters():
        p.requires_grad_(requires_grad[id(p)])


def _find_modules_to_save(model: nn.Module, names: list[str]) -> list[tuple[str, nn.Module]]:
    """The modules matched by modules_to_save, skipping anything inside an adapter layer."""
    if not names:
        return []
    adapters = [n for n, m in model.named_modules() if isinstance(m, AdapterLayer)]
    found = [(n, m) for n, m in model.named_modules()
             if n and _matches(n, names) and not any(n == a or n.startswith(a + ".") for a in adapters)]
    if not found:
        raise ValueError(f"modules_to_save={names!r} matched no module in this model.")
    return found


def _check_no_overlap(keep: list[tuple[str, nn.Module]], targets: list[str]) -> None:
    """A module that is (or contains) a target layer can't also be saved whole: its base weights would train."""
    for k, _ in keep:
        for t in targets:
            if t == k or t.startswith(k + ".") or k.startswith(t + "."):
                raise ValueError(
                    f"modules_to_save entry {k!r} overlaps the target layer {t!r}. A layer can't be both "
                    "adapted and fully trained; remove it from modules_to_save or from target_modules."
                )


def get_quantum_model(model: nn.Module, config: QuantumAdapterConfig) -> QuantumModel:
    """Add quantum adapters to `model` in place, freeze everything else, and wrap it.

    Every layer matching `config.target_modules` (nn.Linear or transformers Conv1D) is wrapped in an
    AdapterLayer that adds the method's correction `x @ ΔW`; ΔW is exactly zero at the start, so the model's
    outputs are unchanged until training moves it. All other parameters are frozen except those of the
    `config.modules_to_save` modules, which train in full and are saved with the adapter.

    Args:
        model: a transformers model (or any nn.Module). It is modified in place; keep using the returned wrapper.
        config: the adapter settings. It is copied, never modified.

    Returns:
        A QuantumModel wrapping `model`. Its `forward` has the wrapped model's signature, so the Trainer works.

    Raises:
        ValueError: no layer matched, a modules_to_save entry matched nothing or overlaps a target layer, the
            model already has adapters, the method is experimental without `experimental=True`, or the layer
            sizes need more qubits than `config.max_qubits`.
        TypeError: a matched layer isn't nn.Linear or Conv1D.

    A failure leaves `model` exactly as it was.
    """
    config = dataclasses.replace(config)
    if config.method in EXPERIMENTAL:
        if not config.experimental:
            raise ValueError(
                f"method={config.method!r} is experimental: it did not pass the release benchmark "
                "(see docs/benchmarks.md). Set experimental=True in the config (or pass experimental=True to "
                "QuantumModel.from_pretrained) to use it anyway."
            )
        warnings.warn(f"method={config.method!r} is experimental; see docs/benchmarks.md.", UserWarning)
    if config.base_model_name_or_path is None:
        config.base_model_name_or_path = getattr(getattr(model, "config", None), "name_or_path", None) or None
    keep = _find_modules_to_save(model, config.modules_to_save)  # validate before mutating anything
    _check_no_overlap(keep, [n for n, _ in _target_modules(model, config.target_modules)])
    method_cls = METHOD_CLASSES[config.method]
    adapted = inject_adapters(model, config.target_modules, lambda i, o: method_cls(i, o, config))
    adapter_ids = {id(p) for _, m in model.named_modules() if isinstance(m, AdapterLayer) for p in m.adapter.parameters()}
    for p in model.parameters():
        p.requires_grad_(id(p) in adapter_ids)
    for _, m in keep:
        m.requires_grad_(True)
    return QuantumModel(model, config, adapted)


@functools.cache
def _signature_class(base_cls: type) -> type:
    """A QuantumModel subclass whose forward carries `base_cls.forward`'s signature.

    The Trainer reads the forward signature, on the instance (which dataset columns to keep) and on the
    class (label names, whether the model returns a loss), so (*args, **kwargs) would make it drop every column.
    """
    def forward(self, *args, **kwargs):
        return QuantumModel.forward(self, *args, **kwargs)

    try:
        forward.__signature__ = inspect.signature(base_cls.forward)
    except (TypeError, ValueError):  # no introspectable forward: keep (*args, **kwargs)
        pass
    return type(f"Quantum{base_cls.__name__}", (QuantumModel,), {"forward": forward})


def _rebuild_quantum_model(state: dict) -> QuantumModel:
    """Unpickle a QuantumModel: the per-model subclass is recreated from the wrapped model's class."""
    obj = QuantumModel.__new__(QuantumModel, state["_modules"]["base_model"])
    obj.__setstate__(state)
    return obj


def _looks_like_local_path(name: str) -> bool:
    """True for strings that can't be a Hub "user/repo" id: relative or absolute paths, or existing files."""
    separators = name.count("/") + (name.count(os.sep) if os.sep != "/" else 0)
    return name.startswith((".", "/", "~", os.sep)) or separators > 1 or os.path.exists(name)


class QuantumModel(nn.Module):
    """A model with quantum adapters: create one with get_quantum_model or QuantumModel.from_pretrained.

    It is an nn.Module that calls the wrapped model (`base_model`) and forwards unknown attributes to it, so
    `model.config`, `model.generate(...)` and the Trainer work as on the original. Attributes:
    `base_model` (the wrapped model, adapted in place), `quantum_config` (a QuantumAdapterConfig) and
    `adapted_modules` (names of the wrapped layers, in base_model's naming). It can be deep-copied, pickled
    and saved with torch.save.
    """

    def __new__(cls, base_model: nn.Module | None = None, *args, **kwargs):
        # base_model is None when copy / pickle rebuild an instance of an already-specialised subclass.
        target = _signature_class(type(base_model)) if cls is QuantumModel and base_model is not None else cls
        return super().__new__(target)

    def __init__(self, base_model: nn.Module, config: QuantumAdapterConfig, adapted_modules: list[str]) -> None:
        super().__init__()
        self.base_model = base_model
        self.quantum_config = config
        self.adapted_modules = adapted_modules

    def __reduce_ex__(self, protocol):
        # The per-model subclass made by _signature_class can't be found by name, so pickle (torch.save,
        # multiprocessing) rebuilds it through a module-level function instead.
        if type(self) is _signature_class(type(self.base_model)):
            return _rebuild_quantum_model, (self.__dict__.copy(),)
        return super().__reduce_ex__(protocol)

    def forward(self, *args, **kwargs):
        """Call the wrapped model; takes and returns exactly what `base_model.forward` does."""
        return self.base_model(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "base_model":
                raise
            return getattr(self.base_model, name)

    def adapter_layers(self) -> list[tuple[str, AdapterLayer]]:
        """(name, AdapterLayer) for each adapted layer; empty after merge_and_unload."""
        return [(n, m) for n, m in self.base_model.named_modules() if isinstance(m, AdapterLayer)]

    def trainable_state_dict(self) -> dict[str, Tensor]:
        """Adapter parameters and buffers, plus modules_to_save, keyed as in base_model.state_dict().

        This is exactly what save_pretrained writes and from_pretrained expects.
        """
        out: dict[str, Tensor] = {}
        for name, layer in self.adapter_layers():
            out.update(layer.adapter.state_dict(prefix=f"{name}.adapter."))
        if self.quantum_config.modules_to_save:
            for name, module in _find_modules_to_save(self.base_model, self.quantum_config.modules_to_save):
                out.update(module.state_dict(prefix=f"{name}."))
        return out

    def _load_trainable_state(self, state: dict[str, Tensor]) -> None:
        """Copy saved adapter (and modules_to_save) weights in, after checking every key and shape."""
        expected = self.trainable_state_dict()
        if set(state) != set(expected):
            raise ValueError(
                f"Adapter weights don't match this config: missing {sorted(set(expected) - set(state))[:5]}, "
                f"unexpected {sorted(set(state) - set(expected))[:5]}."
            )
        for k, v in state.items():  # check sizes first so a bad tensor can't leave a half-loaded model
            if v.shape != expected[k].shape:
                raise ValueError(f"Adapter weight {k!r} has shape {list(v.shape)}, expected {list(expected[k].shape)}.")
        self.base_model.load_state_dict(state, strict=False)

    def print_trainable_parameters(self) -> None:
        """Print how many parameters train (adapters plus modules_to_save) against the model's total."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.4f}")

    def _require_adapters(self) -> list[tuple[str, AdapterLayer]]:
        layers = self.adapter_layers()
        if not layers:
            raise ValueError(
                "This model has no adapter layers (it was merged with merge_and_unload). "
                "Export or summarise before merging."
            )
        return layers

    def circuit_summary(self) -> list[dict]:
        """Print and return one row per adapted layer, with a comparison to LoRA at the same rank.

        Each row has `layer`, `method`, `qubits` ("-" for tensor; "n_in+n_out" for unitary), `dials` (trainable
        circuit numbers), `gates` and `params` (all adapter parameters of that layer).

        Raises:
            ValueError: after merge_and_unload, which removes the adapters.
        """
        self._require_adapters()
        rows = [{"layer": n, **m.adapter.summary()} for n, m in self.adapter_layers()]
        r = self.quantum_config.rank
        lora = sum(r * (m.adapter.in_features + m.adapter.out_features) for _, m in self.adapter_layers())
        total = sum(row["params"] for row in rows)
        print(f"{'layer':<40} {'method':<10} {'qubits':>7} {'dials':>7} {'gates':>7} {'params':>8}")
        for row in rows:
            print(f"{row['layer']:<40} {row['method']:<10} {row['qubits']!s:>7} {row['dials']:>7,} "
                  f"{row['gates']:>7,} {row['params']:>8,}")
        print(f"Total adapter params: {total:,} (LoRA rank {r} would train {lora:,}: {lora / total:.1f}x more)")
        return rows

    def export_circuits(self) -> dict[str, list]:
        """Each adapted layer's trained circuits as Qiskit QuantumCircuits, keyed by layer name.

        `amplitude` gives one circuit per layer; `unitary` gives two (U and V). Our qubit q is Qiskit wire
        n - 1 - q, so Qiskit's Statevector of the circuit equals the simulator's state. Needs the
        `[qiskit]` extra.

        Raises:
            ValueError: for `tensor` (quantum-inspired, no circuit) or after merge_and_unload.
            ImportError: Qiskit is not installed.
        """
        from quantum_adapter.export import to_qiskit

        self._require_adapters()
        method_cls = METHOD_CLASSES[self.quantum_config.method]
        if not method_cls.quantum_gates:
            raise ValueError(
                f"method={self.quantum_config.method!r} is quantum-inspired (its gates aren't unitary), so it has "
                "no quantum circuit to export. Use method='amplitude' or 'unitary'."
            )
        return {n: [to_qiskit(spec) for spec in m.adapter.circuits()] for n, m in self.adapter_layers()}

    def _check_untied(self) -> None:
        names: dict[int, list[str]] = {}
        for n, p in self.base_model.named_parameters(remove_duplicate=False):
            names.setdefault(p.data_ptr(), []).append(n)
        for name, layer in self.adapter_layers():
            shared = [n for n in names.get(layer.base.weight.data_ptr(), []) if n != f"{name}.base.weight"]
            if shared:
                raise ValueError(
                    f"Can't merge {name!r}: its weight is tied to {', '.join(shared)}, so merging would change "
                    "that layer too. Nothing was merged. Keep the adapter unmerged, or untie the weights first."
                )

    def merge_and_unload(self) -> nn.Module:
        """Add each adapter's ΔW into its layer's weight and remove the adapters; returns `base_model`.

        The result is the plain transformers model (same class, size and speed as the original) and gives the
        same outputs as the adapted model, up to rounding in the weights' dtype. The merge is in place and
        one-way: this QuantumModel has no adapters afterwards, so save_pretrained, export_circuits and
        circuit_summary should be called first. modules_to_save weights are already part of the model.

        Raises:
            ValueError: a targeted layer's weight is tied to another parameter (for example an output layer
                tied to the input embeddings): merging would change both. Nothing is merged in that case.
        """
        self._check_untied()  # before merging anything, so a refusal leaves the model as it was
        for name, layer in self.adapter_layers():
            parent, child = _parent(self.base_model, name)
            setattr(parent, child, layer.merge())
        return self.base_model

    def _layer_shapes(self) -> dict[str, list[int]]:
        return {n: [m.adapter.in_features, m.adapter.out_features] for n, m in self.adapter_layers()}

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Write `adapter_config.json` and `adapter_model.safetensors` to `save_directory` (created if needed).

        Only the adapters and modules_to_save are written, never the base model. The config records the
        settings, `format_version`, the library version and each adapted layer's shape, which
        from_pretrained checks against the model it loads onto.
        """
        from quantum_adapter import __version__

        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        meta = {**self.quantum_config.to_dict(), "format_version": FORMAT_VERSION, "library_version": __version__,
                "layer_shapes": self._layer_shapes()}
        (path / CONFIG_NAME).write_text(json.dumps(meta, indent=2))
        # clone: tied weights in modules_to_save would otherwise share storage, which safetensors rejects
        tensors = {k: v.detach().cpu().clone().contiguous() for k, v in self.trainable_state_dict().items()}
        save_file(tensors, path / WEIGHTS_NAME, metadata={"format": "pt"})

    @classmethod
    def from_pretrained(
        cls,
        model: nn.Module,
        path_or_repo_id: str | Path,
        *,
        revision: str | None = None,
        is_trainable: bool = False,
        token: str | None = None,
        experimental: bool | None = None,
    ) -> QuantumModel:
        """Load a saved adapter onto `model`, the same base model it was trained on.

        Args:
            model: a fresh (not yet adapted) copy of the base model. It is adapted in place.
            path_or_repo_id: a directory written by save_pretrained, or a Hugging Face Hub repo id.
            revision: Hub branch, tag or commit; ignored for a local directory.
            is_trainable: False (default) freezes everything and puts the model in eval mode, for inference.
                True leaves the adapters and modules_to_save trainable, to continue training.
            token: Hub token, for private repos.
            experimental: overrides the saved config's `experimental` flag, e.g. True to load an adapter
                whose method has since been marked experimental. None keeps the saved value.

        Raises:
            ValueError: the model already has adapters, the adapter was saved for different layer shapes,
                its format_version is newer than this release reads, or the weights don't match the config.

        A failure leaves `model` exactly as it was.
        """
        _reject_adapted(model)  # before anything else, so the user's existing adapters are never touched
        path = Path(path_or_repo_id)
        if not path.is_dir():
            if _looks_like_local_path(str(path_or_repo_id)):
                raise FileNotFoundError(f"No adapter directory at {path_or_repo_id}")
            path = Path(snapshot_download(
                str(path_or_repo_id), revision=revision, token=token, allow_patterns=[CONFIG_NAME, WEIGHTS_NAME]))
        meta = json.loads((path / CONFIG_NAME).read_text())
        version = meta.pop("format_version", 1)  # adapters saved before the field existed are version 1
        if not isinstance(version, int) or isinstance(version, bool) or not 1 <= version <= FORMAT_VERSION:
            raise ValueError(
                f"This adapter has format_version {version!r}; this quantum-adapter reads versions up to "
                f"{FORMAT_VERSION}. It was probably saved by a newer release: upgrade with "
                "`pip install -U quantum-adapter`."
            )
        saved_shapes = meta.pop("layer_shapes", None)
        config = QuantumAdapterConfig.from_dict(meta)
        if experimental is not None:
            config.experimental = experimental
        planned = _planned_layer_shapes(model, config.target_modules)
        if saved_shapes is not None and planned and saved_shapes != planned:
            name = next(n for n in sorted(set(saved_shapes) | set(planned)) if saved_shapes.get(n) != planned.get(n))
            raise ValueError(
                f"This adapter was saved for a model with different layers (first difference: {name}: "
                f"saved {saved_shapes.get(name)}, this model {planned.get(name)}). "
                f"Load it onto the base model it was trained on: {config.base_model_name_or_path}."
            )
        state = load_file(path / WEIGHTS_NAME)
        flags = {id(p): p.requires_grad for p in model.parameters()}
        qm = None
        try:
            qm = get_quantum_model(model, config)
            qm._load_trainable_state(state)
        except BaseException:
            _restore(model, flags, qm.adapted_modules if qm is not None else [])
            raise
        if not is_trainable:
            qm.requires_grad_(False)
            qm.eval()
        return qm

    def push_to_hub(
        self, repo_id: str, *, private: bool = True, token: str | None = None,
        commit_message: str = "Upload quantum adapter",
    ) -> str:
        """Upload the adapter (the two save_pretrained files) to the Hugging Face Hub; returns the commit URL.

        The repo is created if it doesn't exist, as a **private** repo unless `private=False`; an existing
        repo keeps its visibility. `token` defaults to the logged-in Hugging Face token.
        """
        api = HfApi(token=token)
        api.create_repo(repo_id, private=private, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            self.save_pretrained(tmp)
            info = api.upload_folder(repo_id=repo_id, folder_path=tmp, commit_message=commit_message)
        return info.commit_url
