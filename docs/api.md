# API reference

Everything below is importable from `quantum_adapter`. Anything else (modules, functions or attributes whose names start with `_`, `quantum_adapter.layers`, `quantum_adapter.sim`, `quantum_adapter.methods`) is internal and may change between releases.

```python
from quantum_adapter import QuantumAdapterConfig, QuantumModel, QuantumTrainer, get_quantum_model
```

## `QuantumAdapterConfig`

A dataclass of adapter settings. It is validated when created (bad values raise `ValueError` with the fix) and saved as `adapter_config.json`.

| Field | Default | Meaning |
|---|---|---|
| `target_modules` | required | Layers to adapt. A list of names, each matching a module's full name or its last parts (`["c_attn"]` matches `transformer.h.0.attn.c_attn`), or one regex string matched against full names. Matched layers must be `nn.Linear` or transformers `Conv1D`. |
| `method` | `"amplitude"` | `"amplitude"`, `"unitary"` or `"tensor"`. |
| `rank` | `4` | Rank of ΔW for `amplitude` and `unitary`. Ignored by `tensor`. |
| `qubits` | `"auto"` | `amplitude` only: circuit width, or `"auto"` for the fewest qubits giving `rank * (in + out)` amplitudes. `unitary` always uses ⌈log2 in⌉ and ⌈log2 out⌉ qubits. |
| `depth` | `10` | Number of RY + entangler blocks per circuit (`amplitude`, `unitary`). |
| `entanglement` | `"ring"` | Qubit pairs each entangling layer links: `"ring"`, `"line"` or `"full"`. |
| `tensor_axes` | `4` | `tensor` only: how many axes each layer's input and output sizes are split into (2 to 8). |
| `modules_to_save` | `[]` | Extra modules trained in full and saved with the adapter, e.g. `["score"]` (GPT-2) or `["classifier"]` (BERT/RoBERTa). Same name matching as `target_modules`. An entry may not be, or contain, a target layer. |
| `max_qubits` | `20` | Refuse circuits wider than this. The simulator holds 2^qubits numbers per circuit. |
| `simulator_device` | `"cpu"` | Device the circuit simulation runs on. CPU is usually fastest; the result is moved to the model's device. |
| `experimental` | `False` | Must be `True` to use a method that has not passed the release benchmark (a warning is shown). |
| `base_model_name_or_path` | `None` | Recorded in the saved config. Filled from `model.config.name_or_path` when `None`. |

Methods: `to_dict()` returns the settings as a dict; `QuantumAdapterConfig.from_dict(d)` rebuilds one and raises on unknown keys (they mean a newer release saved it).

## `get_quantum_model(model, config) -> QuantumModel`

Adds adapters to `model` **in place**, freezes everything else and returns the wrapper.

- Each layer matching `target_modules` is wrapped so that it computes `x @ W + x @ ΔW`. ΔW is exactly zero at the start, so outputs are unchanged until training moves it.
- Only adapter parameters and `modules_to_save` train.
- `config` is copied, never modified.
- Raises `ValueError` if no layer matches, a `modules_to_save` entry matches nothing or overlaps a target, the model already has adapters, the method is experimental without `experimental=True`, or a layer needs more than `max_qubits` qubits; `TypeError` if a matched layer isn't `nn.Linear`/`Conv1D`. A failure leaves `model` unchanged.

## `QuantumModel`

The wrapper returned by `get_quantum_model` and `QuantumModel.from_pretrained`. It is an `nn.Module`:

- `forward(...)` calls the wrapped model and has its signature (so the Trainer keeps the right dataset columns).
- Unknown attributes are forwarded to the wrapped model: `model.config`, `model.generate(...)` work.
- Attributes: `base_model` (the wrapped model, adapted in place), `quantum_config`, `adapted_modules` (names of wrapped layers).
- It can be deep-copied, pickled and saved with `torch.save` (load with `torch.load(..., weights_only=False)`). For sharing, prefer `save_pretrained`.

### `print_trainable_parameters()`

Prints trainable parameters (adapters plus `modules_to_save`) against the total.

### `circuit_summary() -> list[dict]`

Prints and returns one row per adapted layer: `layer`, `method`, `qubits` (`"-"` for `tensor`, `"n_in+n_out"` for `unitary`), `dials` (trainable circuit numbers), `gates`, `params`; and the parameter count LoRA would need at the same rank. Raises `ValueError` after `merge_and_unload`.

### `export_circuits() -> dict[str, list[QuantumCircuit]]`

Each adapted layer's trained circuits as Qiskit circuits, keyed by layer name: one per layer for `amplitude`, two (U and V) for `unitary`. Our qubit `q` maps to Qiskit wire `n - 1 - q`, so Qiskit's statevector equals the simulator's. Needs `pip install "quantum-adapter[qiskit]"`. Raises `ValueError` for `tensor` (quantum-inspired, no circuit) and after `merge_and_unload`.

### `trainable_state_dict() -> dict[str, Tensor]`

The adapter parameters and buffers plus `modules_to_save`, keyed as in `base_model.state_dict()`. This is exactly what `save_pretrained` writes.

### `save_pretrained(save_directory)`

Writes two files: `adapter_config.json` (the settings, `format_version`, library version and each adapted layer's shape) and `adapter_model.safetensors` (the tensors of `trainable_state_dict()`). The base model is never written.

### `QuantumModel.from_pretrained(model, path_or_repo_id, *, revision=None, is_trainable=False, token=None, experimental=None)`

Loads a saved adapter onto `model`, which must be a fresh copy of the base model it was trained on.

- `path_or_repo_id`: a directory written by `save_pretrained`, or a Hub repo id. `revision` and `token` are passed to the Hub download.
- `is_trainable=False` (default) freezes everything and calls `eval()`, for inference. `True` keeps adapters and `modules_to_save` trainable, to continue training.
- `experimental`: overrides the saved `experimental` flag, e.g. `True` to load an adapter whose method was later marked experimental. `None` keeps the saved value.
- Raises `ValueError` if `model` already has adapters, the layer shapes differ from the saved ones, the adapter's `format_version` is newer than this release reads, or the weights don't match the config. A failure leaves `model` unchanged.

### `push_to_hub(repo_id, *, private=True, token=None, commit_message="Upload quantum adapter") -> str`

Uploads the two `save_pretrained` files and returns the commit URL. A new repo is created **private** unless `private=False`; an existing repo keeps its visibility. `token` defaults to your logged-in Hugging Face token.

### `merge_and_unload() -> nn.Module`

Adds each ΔW into its layer's weight, removes the adapters and returns the plain wrapped model (same class, size and speed as the original; same outputs up to rounding in the weights' dtype). In place and one-way: save, export and summarise first, because the `QuantumModel` has no adapters afterwards. Raises `ValueError`, merging nothing, if a targeted layer's weight is tied to another parameter (e.g. an output layer tied to the embeddings).

## Training guidance

Quantum adapters learn more slowly than LoRA. Train about 2x longer (e.g. 6 epochs where LoRA uses 3) with a learning rate near 1e-2. Depth 20 was worse in one test (3 epochs), and adapting more layers (`c_proj`, `c_fc` as well as `c_attn`) did not help. See [experiments.md](experiments.md) for the numbers and caveats.

## `QuantumTrainer`

A `transformers.Trainer` subclass for a `QuantumModel`, used exactly like `Trainer`. Needs `pip install "quantum-adapter[trainer]"` (adds `accelerate`); without it, importing `QuantumTrainer` raises an `ImportError` saying so. Checkpoints hold only the adapter (`adapter_config.json` + `adapter_model.safetensors`, plus the trainer's optimizer and state files), so they are small and work on models with tied embeddings. `train(resume_from_checkpoint=...)` and `load_best_model_at_end=True` load the adapter weights back into the model being trained, and every checkpoint directory loads with `QuantumModel.from_pretrained`. `QuantumTrainer` is imported on first use, so `import quantum_adapter` alone doesn't import the Trainer.

The plain `transformers.Trainer` also trains a `QuantumModel`, but only with `save_strategy="no"`.

## Saved format

`adapter_config.json` has `"format_version": 1`. A release refuses to load a newer version with a message to upgrade; files without the field are version 1. Version 1 fixes, among other things, the `unitary` parameterisation: `theta_u` and `theta_v` have shape `(depth + 1, qubits)`.
