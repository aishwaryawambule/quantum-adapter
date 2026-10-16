# quantum-adapter library: implementation plan

> Steps use checkbox (`- [ ]`) syntax for tracking. Work task by task, in order. Each task ends green and committed.

**Goal:** Build and prepare for release the `quantum-adapter` PyPI library: three circuit-generated adapter methods for Hugging Face models, with PEFT-like save/load/merge/hub, Qiskit export, tests, CI, a release benchmark that decides what ships, and docs.

**Architecture:** A small batched real state-vector simulator (`sim/`) feeds three adapter methods (`methods/`), each turning trainable parameters into a weight correction ΔW for one layer. `layers.py` wraps `nn.Linear` / `Conv1D` layers with a method; `model.py` finds target layers, freezes the base model and provides save/load/merge/hub. The benchmark (`benchmarks/`) runs the methods against LoRA and classical controls and applies the production-usefulness rule.

**Tech Stack:** Python ≥ 3.10, PyTorch ≥ 2.1, transformers, safetensors, huggingface_hub; Qiskit (optional extra); pytest, ruff, uv, hatchling; GitHub Actions; PyPI Trusted Publishing. Benchmark only: datasets, peft.

**Spec:** `docs/specs/2026-10-16-quantum-adapter-library-design.md`

## Global Constraints

- Package name `quantum-adapter`, import name `quantum_adapter`, first version `0.1.0`, status beta.
- `requires-python = ">=3.10"`; runtime deps only: torch ≥ 2.3 (CPU half-precision LayerNorm needs it), transformers ≥ 4.40, safetensors ≥ 0.4, huggingface_hub ≥ 0.25. Qiskit only via the `[qiskit]` extra.
- Licence Apache-2.0. Type hints on all public functions. `from __future__ import annotations` in every module.
- Adapter convention everywhere: `ΔW` has shape `(in_features, out_features)` and the correction is `x @ ΔW`.
- Simulator qubit order: qubit `q` is the most-significant-first bit `n - 1 - q` of the basis index. Qiskit export maps our qubit `q` to Qiskit wire `n - 1 - q`.
- Simulator runs on `config.simulator_device` (default `"cpu"`), in float32 (float64 if the parameters are float64).
- Every new adapter must leave the model's outputs unchanged at step 0 (neutral start).
- No AI-tool attribution trailers or references in any file, commit message or PR. Plain conventional commit messages.
- Datasets: SST-2 = `nyu-mll/glue` config `sst2`; AG News = `fancyzhx/ag_news`.
- Production-usefulness rule (spec §1): accuracy within 1.0 point of LoRA r=4 (mean of 3 seeds, both tasks); ≥ 5× fewer trainable params; training time ≤ 2× LoRA; merged inference time within 10% of the plain base model; beats the same-size classical control on both tasks.

## Review Focus

1. **Half-precision models** (`torch_dtype=torch.float16/bfloat16`): adapters must work, output dtype must equal the model's dtype. Pinned in Task 5 (`test_half_precision_model`).
2. **Model moved after adapting** (`qm.to("mps")` / `"cuda"`): adapter params follow, simulator still runs on CPU, output lands on the model's device. Pinned in Task 5 (`test_runs_on_accelerator`, skipped when no accelerator).
3. **Gradient accumulation** (two forwards, one backward; or eval between steps): no "backward through the graph a second time" errors and no stale eval cache. Pinned in Task 4 (`test_two_forwards_one_backward`, `test_eval_cache_invalidates_after_step`).
4. **`target_modules` matching a non-linear layer** (e.g. `"wte"` embedding): clear `TypeError`, not a crash deep in forward. Pinned in Task 5 (`test_rejects_unsupported_layer`).
5. **Adapting twice** (`get_quantum_model` on an already adapted model): clear `ValueError`. Pinned in Task 5 (`test_rejects_double_adapt`).

---

## File structure

```
pyproject.toml                       build, deps, tool config (Task 1)
LICENSE, CHANGELOG.md, README.md     (Task 1, Task 11)
src/quantum_adapter/
  __init__.py                        public exports + __version__
  config.py                          QuantumAdapterConfig, validation, dict round-trip
  sim/__init__.py                    re-exports
  sim/statevector.py                 RY / CNOT / CZ, entangling patterns, run_ansatz
  methods/__init__.py                METHOD_CLASSES, EXPERIMENTAL
  methods/base.py                    AdapterMethod base, CircuitSpec, eval cache, simulate()
  methods/amplitude.py               AmplitudeMethod
  methods/unitary.py                 UnitaryMethod (Quantum-PEFT)
  methods/tensor.py                  TensorMethod (QuanTA), balanced_factors
  layers.py                          AdapterLayer, layer_shape, SUPPORTED_LAYERS
  model.py                           inject_adapters, get_quantum_model, QuantumModel
  export.py                          to_qiskit
tests/
  conftest.py                        tiny GPT-2 / Llama factories
  test_config.py  test_statevector.py  test_methods.py  test_unitary.py  test_tensor.py
  test_model.py  test_save_load.py  test_trainer.py  test_export.py
benchmarks/
  controls.py                        RandomProjectionMethod (classical control)
  run.py                             one task × contenders × seeds → JSONL
  gate.py                            applies the production rule, writes docs/benchmarks.md
examples/quickstart.py               used by docs and the release smoke test
docs/how-it-works.md  docs/benchmarks.md
.github/workflows/tests.yml  .github/workflows/release.yml
```

`experiments/sst2_bench.py` and `results/` stay as the historical first experiment.

---

### Task 0: Commit the bigger-data experiment

**Files:** `experiments/sst2_bench.py` (already modified), `results/run_n8000.log`, `results/sst2_n8000.json`, `results/err_n8000.log`

- [ ] **Step 1: Confirm the run finished**

Run: `wc -l results/run_n8000.log && ls results/sst2_n8000.json`
Expected: 12 lines; the JSON file exists. If not, wait for the run.

- [ ] **Step 2: Add the n=8000 row to the README results**

Run:
```bash
.venv/bin/python - <<'EOF'
import json, statistics as st
r = json.load(open('results/sst2_n8000.json'))
for k in ['head', 'lora', 'quantum', 'random']:
    v = [x['val_acc'] for x in r if x['kind'] == k]
    print(k, f"{100*st.mean(v):.1f}% ({100*min(v):.1f}-{100*max(v):.1f})", r[[x['kind'] for x in r].index(k)]['trainable'],
          round(st.mean(x['train_s'] for x in r if x['kind'] == k)))
EOF
```
Add a second table under "First result" in `README.md` titled `### With 8,000 training reviews` with these four rows, same columns as the first table.

- [ ] **Step 3: Commit**

```bash
rm -f results/err_n8000.log
git add experiments/sst2_bench.py results/ README.md
git commit -m "feat: rerun benchmark with 8,000 training reviews"
```

---

### Task 1: Package skeleton and config

**Files:**
- Modify: `pyproject.toml`, `.gitignore`
- Create: `LICENSE`, `src/quantum_adapter/__init__.py`, `src/quantum_adapter/config.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `QuantumAdapterConfig` (dataclass, fields below), `.to_dict() -> dict`, `QuantumAdapterConfig.from_dict(dict) -> QuantumAdapterConfig`; constants `METHODS`, `ENTANGLEMENTS`; `too_many_qubits_message(n: int, limit: int) -> str`.

- [ ] **Step 1: Replace `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "quantum-adapter"
dynamic = ["version"]
description = "Fine-tune Hugging Face language models with adapters generated by quantum circuits"
readme = "README.md"
license = "Apache-2.0"
license-files = ["LICENSE"]
requires-python = ">=3.10"
authors = [{ name = "aishwaryawambule", email = "aishwaryawambule@gmail.com" }]
keywords = ["quantum", "fine-tuning", "lora", "peft", "llm", "transformers", "qiskit"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Physics",
]
dependencies = [
    "torch>=2.1",
    "transformers>=4.40",
    "safetensors>=0.4",
    "huggingface_hub>=0.25",
]

[project.optional-dependencies]
qiskit = ["qiskit>=1.0"]

[project.urls]
Homepage = "https://github.com/aishwaryawambule/quantum-adapter"
Issues = "https://github.com/aishwaryawambule/quantum-adapter/issues"

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6", "qiskit>=1.0", "accelerate>=0.26"]
bench = ["datasets>=3.0", "peft>=0.13"]

[tool.hatch.version]
path = "src/quantum_adapter/__init__.py"

[tool.hatch.build.targets.wheel]
packages = ["src/quantum_adapter"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 110

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }]
```

(The CPU index only affects this repo's dev environment and CI; PyPI users get whatever torch they install.)

- [ ] **Step 2: Add the licence and ignore build output**

```bash
curl -sL https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE
head -3 LICENSE        # expect "Apache License" / "Version 2.0, January 2004"
printf 'dist/\n.pytest_cache/\n.ruff_cache/\n' >> .gitignore
```

- [ ] **Step 3: Write the failing tests** — `tests/test_config.py`

```python
import pytest

from quantum_adapter import QuantumAdapterConfig


def test_defaults():
    c = QuantumAdapterConfig(target_modules=["c_attn"])
    assert (c.method, c.rank, c.qubits, c.depth, c.entanglement) == ("amplitude", 4, "auto", 10, "ring")
    assert c.modules_to_save == [] and c.experimental is False


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"method": "magic"}, "method must be one of"),
        ({"rank": 0}, "rank must be a positive integer"),
        ({"depth": -1}, "depth must be a positive integer"),
        ({"entanglement": "star"}, "entanglement must be one of"),
        ({"qubits": "many"}, "qubits must be 'auto' or a positive integer"),
        ({"qubits": 25}, "max_qubits=20"),
        ({"tensor_axes": 1}, "tensor_axes must be between 2 and 8"),
        ({"target_modules": []}, "target_modules must name at least one layer"),
    ],
)
def test_rejects_bad_values(kwargs, message):
    args = {"target_modules": ["c_attn"], **kwargs}
    with pytest.raises(ValueError, match=message):
        QuantumAdapterConfig(**args)


def test_dict_round_trip():
    c = QuantumAdapterConfig(target_modules=["q_proj", "v_proj"], method="unitary", rank=8, modules_to_save=["score"])
    assert QuantumAdapterConfig.from_dict(c.to_dict()) == c


def test_from_dict_ignores_library_version_but_rejects_unknown_keys():
    d = QuantumAdapterConfig(target_modules=["c_attn"]).to_dict()
    assert QuantumAdapterConfig.from_dict({**d, "library_version": "9.9.9"}).target_modules == ["c_attn"]
    with pytest.raises(ValueError, match="newer quantum-adapter"):
        QuantumAdapterConfig.from_dict({**d, "future_option": 1})


def test_regex_target_modules_kept_as_string():
    assert QuantumAdapterConfig(target_modules=r".*\.attn\.c_attn").target_modules == r".*\.attn\.c_attn"
```

- [ ] **Step 4: Run to see it fail**

Run: `uv sync --group dev && uv run pytest tests/test_config.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'quantum_adapter'`

- [ ] **Step 5: Implement** — `src/quantum_adapter/__init__.py`

```python
"""Fine-tune language models with adapters generated by quantum circuits."""

from __future__ import annotations

from quantum_adapter.config import QuantumAdapterConfig

__version__ = "0.1.0"
__all__ = ["QuantumAdapterConfig", "__version__"]
```

`src/quantum_adapter/config.py`

```python
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
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> QuantumAdapterConfig:
        data = dict(data)
        data.pop("library_version", None)
        unknown = sorted(set(data) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(
                f"Unknown config keys {unknown}. This adapter may have been saved by a newer quantum-adapter; "
                "upgrade with `pip install -U quantum-adapter`."
            )
        return cls(**data)
```

- [ ] **Step 6: Run to see it pass**

Run: `uv run pytest tests/test_config.py -q && uv run ruff check .`
Expected: 13 passed; ruff clean.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .gitignore LICENSE src tests
git commit -m "feat: package skeleton and adapter config"
```

---

### Task 2: State-vector simulator

**Files:**
- Create: `src/quantum_adapter/sim/__init__.py`, `src/quantum_adapter/sim/statevector.py`, `tests/test_statevector.py`

**Interfaces:**
- Produces:
  - `zero_state(batch: int, n: int, *, dtype=torch.float32, device=None) -> Tensor` shape `(batch, 2**n)`
  - `basis_states(k: int, n: int, *, dtype=torch.float32, device=None) -> Tensor` shape `(k, 2**n)`, rows `e_0..e_{k-1}`
  - `apply_ry(state, q, angle, n)`, `apply_cnot(state, control, target, n)`, `apply_cz(state, a, b, n)` → same shape
  - `entangling_pairs(n: int, pattern: str) -> list[tuple[int, int]]`
  - `run_ansatz(state, theta, n, *, entanglement="ring", gate="cx", final_rotation=False) -> Tensor`: for each row `k` of `theta` (shape `(layers, n)`), RY(`theta[k, q]`) on every qubit, then the entangling layer, except after the last row when `final_rotation=True`.

- [ ] **Step 1: Write the failing tests** — `tests/test_statevector.py`

```python
import math

import pytest
import torch

from quantum_adapter.sim import (
    apply_cnot, apply_cz, apply_ry, basis_states, entangling_pairs, run_ansatz, zero_state,
)

S = 1 / math.sqrt(2)


def test_ry_on_zero():
    out = apply_ry(zero_state(1, 1), 0, torch.tensor(1.0), 1)
    assert torch.allclose(out, torch.tensor([[math.cos(0.5), math.sin(0.5)]]))


def test_qubit_zero_is_most_significant_bit():
    out = apply_ry(zero_state(1, 2), 0, torch.tensor(math.pi), 2)   # |00> -> |10> = index 2
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, 1.0, 0.0]]), atol=1e-6)


def test_bell_state():
    s = apply_cnot(apply_ry(zero_state(1, 2), 0, torch.tensor(math.pi / 2), 2), 0, 1, 2)
    assert torch.allclose(s, torch.tensor([[S, 0.0, 0.0, S]]), atol=1e-6)


def test_cnot_respects_control_and_target():
    s = basis_states(4, 2)                      # rows |00>, |01>, |10>, |11>
    out = apply_cnot(s, 1, 0, 2)                # control q1, target q0: |01> -> |11>, |11> -> |01>
    assert out.argmax(1).tolist() == [0, 3, 2, 1]


def test_cz_flips_sign_of_11_only():
    s = apply_ry(apply_ry(zero_state(1, 2), 0, torch.tensor(math.pi / 2), 2), 1, torch.tensor(math.pi / 2), 2)
    assert torch.allclose(apply_cz(s, 0, 1, 2), torch.tensor([[0.5, 0.5, 0.5, -0.5]]), atol=1e-6)


def test_entangling_pairs():
    assert entangling_pairs(1, "ring") == []
    assert entangling_pairs(2, "ring") == [(0, 1)]
    assert entangling_pairs(3, "ring") == [(0, 1), (1, 2), (2, 0)]
    assert entangling_pairs(3, "line") == [(0, 1), (1, 2)]
    assert entangling_pairs(3, "full") == [(0, 1), (0, 2), (1, 2)]
    with pytest.raises(ValueError):
        entangling_pairs(3, "star")


@pytest.mark.parametrize("pattern", ["ring", "line", "full"])
@pytest.mark.parametrize("gate", ["cx", "cz"])
@pytest.mark.parametrize("final", [False, True])
def test_norm_is_preserved(pattern, gate, final):
    theta = torch.rand(3, 6) * 2 * math.pi
    out = run_ansatz(basis_states(5, 6), theta, 6, entanglement=pattern, gate=gate, final_rotation=final)
    assert torch.allclose(out.norm(dim=1), torch.ones(5), atol=1e-5)


def test_final_rotation_skips_last_entangler():
    theta = torch.zeros(1, 2)                     # identity rotations
    s = basis_states(4, 2)
    assert torch.allclose(run_ansatz(s, theta, 2, gate="cx", final_rotation=True), s)
    assert not torch.allclose(run_ansatz(s, theta, 2, gate="cx", final_rotation=False), s)


def test_gradients_match_finite_differences():
    theta = (torch.rand(2, 3, dtype=torch.float64) * 2 * math.pi).requires_grad_()
    start = zero_state(1, 3, dtype=torch.float64)
    weights = torch.randn(8, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda t: (run_ansatz(start, t, 3) * weights).sum(), (theta,))


def test_rejects_wrong_theta_width_and_gate():
    with pytest.raises(ValueError, match="theta has 4 columns but n=3"):
        run_ansatz(zero_state(1, 3), torch.zeros(2, 4), 3)
    with pytest.raises(ValueError, match="gate must be"):
        run_ansatz(zero_state(1, 3), torch.zeros(2, 3), 3, gate="swap")
```

- [ ] **Step 2: Run to see it fail**

Run: `uv run pytest tests/test_statevector.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'quantum_adapter.sim'`

- [ ] **Step 3: Implement** — `src/quantum_adapter/sim/statevector.py`

```python
"""Batched real state-vector simulator for RY, CNOT and CZ circuits.

Qubit q is tensor axis q, most significant first: basis index = sum(bit_q << (n - 1 - q)).
Every gate here is real, so states stay real and autograd gives exact gradients.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor


def zero_state(batch: int, n: int, *, dtype: torch.dtype = torch.float32, device=None) -> Tensor:
    state = torch.zeros(batch, 2**n, dtype=dtype, device=device)
    state[:, 0] = 1.0
    return state


def basis_states(k: int, n: int, *, dtype: torch.dtype = torch.float32, device=None) -> Tensor:
    return torch.eye(2**n, dtype=dtype, device=device)[:k]


def apply_ry(state: Tensor, q: int, angle: Tensor, n: int) -> Tensor:
    b = state.shape[0]
    s = state.reshape(b, 2**q, 2, 2 ** (n - q - 1))
    c, si = torch.cos(angle / 2), torch.sin(angle / 2)
    a0, a1 = s[:, :, 0], s[:, :, 1]
    return torch.stack((c * a0 - si * a1, si * a0 + c * a1), dim=2).reshape(b, -1)


@lru_cache(maxsize=None)
def _cnot_index(n: int, control: int, target: int, device: str) -> Tensor:
    idx = torch.arange(2**n)
    ctrl = (idx >> (n - 1 - control)) & 1
    return (idx ^ (ctrl << (n - 1 - target))).to(device)


def apply_cnot(state: Tensor, control: int, target: int, n: int) -> Tensor:
    return state[:, _cnot_index(n, control, target, str(state.device))]


@lru_cache(maxsize=None)
def _cz_sign(n: int, a: int, b: int, device: str, dtype: torch.dtype) -> Tensor:
    idx = torch.arange(2**n)
    both = ((idx >> (n - 1 - a)) & 1) & ((idx >> (n - 1 - b)) & 1)
    return (1 - 2 * both).to(device=device, dtype=dtype)


def apply_cz(state: Tensor, a: int, b: int, n: int) -> Tensor:
    return state * _cz_sign(n, a, b, str(state.device), state.dtype)


def entangling_pairs(n: int, pattern: str) -> list[tuple[int, int]]:
    if pattern not in ("ring", "line", "full"):
        raise ValueError(f"entanglement must be 'ring', 'line' or 'full', got {pattern!r}.")
    if n < 2:
        return []
    if pattern == "line" or n == 2:
        return [(q, q + 1) for q in range(n - 1)]
    if pattern == "ring":
        return [(q, (q + 1) % n) for q in range(n)]
    return [(a, b) for a in range(n) for b in range(a + 1, n)]


def run_ansatz(
    state: Tensor,
    theta: Tensor,
    n: int,
    *,
    entanglement: str = "ring",
    gate: str = "cx",
    final_rotation: bool = False,
) -> Tensor:
    """Apply RY layers theta[k] (one angle per qubit), each followed by an entangling layer.

    With final_rotation=True the last row of theta is a closing RY layer with no entangler after it.
    """
    if theta.shape[-1] != n:
        raise ValueError(f"theta has {theta.shape[-1]} columns but n={n} qubits.")
    if gate not in ("cx", "cz"):
        raise ValueError(f"gate must be 'cx' or 'cz', got {gate!r}.")
    entangle = apply_cnot if gate == "cx" else apply_cz
    pairs = entangling_pairs(n, entanglement)
    layers = theta.shape[0]
    for k in range(layers):
        for q in range(n):
            state = apply_ry(state, q, theta[k, q], n)
        if not (final_rotation and k == layers - 1):
            for a, b in pairs:
                state = entangle(state, a, b, n)
    return state
```

`src/quantum_adapter/sim/__init__.py`

```python
from __future__ import annotations

from quantum_adapter.sim.statevector import (
    apply_cnot, apply_cz, apply_ry, basis_states, entangling_pairs, run_ansatz, zero_state,
)

__all__ = ["apply_cnot", "apply_cz", "apply_ry", "basis_states", "entangling_pairs", "run_ansatz", "zero_state"]
```

- [ ] **Step 4: Run to see it pass**

Run: `uv run pytest tests/test_statevector.py -q`
Expected: all passed.

- [ ] **Step 5: Check speed did not regress from the experiment**

Run:
```bash
uv run python -c "
import time, torch
from quantum_adapter.sim import run_ansatz, zero_state
th = (torch.rand(10, 14) * 6.28).requires_grad_()
for _ in range(2):
    t = time.time(); run_ansatz(zero_state(1, 14), th, 14).sum().backward(); print(round(time.time() - t, 3))
"
```
Expected: second number ≤ 0.05 s (the experiment needed ~0.17 s for 12 such circuits). If it's much slower, stop and investigate before continuing.

- [ ] **Step 6: Commit**

```bash
git add src/quantum_adapter/sim tests/test_statevector.py
git commit -m "feat: batched real state-vector simulator"
```

---

### Task 3: Method base class and the `amplitude` method

**Files:**
- Create: `src/quantum_adapter/methods/__init__.py`, `src/quantum_adapter/methods/base.py`, `src/quantum_adapter/methods/amplitude.py`, `tests/test_methods.py`

**Interfaces:**
- Consumes: `run_ansatz`, `zero_state`, `basis_states`, `entangling_pairs` (Task 2); `QuantumAdapterConfig`, `too_many_qubits_message` (Task 1).
- Produces:
  - `CircuitSpec(name: str, n_qubits: int, theta: Tensor, entanglement: str, gate: str, final_rotation: bool)` (frozen dataclass)
  - `AdapterMethod(nn.Module)` with attributes `name: str`, `quantum_gates: bool`, `in_features: int`, `out_features: int`; methods `forward(x: Tensor) -> Tensor` (`(..., in) → (..., out)`, equals `x @ delta_weight()`), `delta_weight() -> Tensor` `(in, out)`, `summary() -> dict` with keys `method, qubits, dials, gates, params`, `circuits() -> list[CircuitSpec]`; helpers `simulate(initial, theta, n, gate, final_rotation) -> Tensor` and `_cached(compute: Callable)`.
  - `AmplitudeMethod(in_features, out_features, config)`; attributes `n_qubits`, `theta (depth, n)`, `scale ()`; `factors() -> tuple[Tensor, Tensor]`.
  - `METHOD_CLASSES: dict[str, type[AdapterMethod]]`, `EXPERIMENTAL: frozenset[str]` in `methods/__init__.py`.

- [ ] **Step 1: Write the failing tests** — `tests/test_methods.py` (shared behaviour; Tasks 4 extend `METHODS`)

```python
import math

import pytest
import torch
from torch.func import functional_call

from quantum_adapter import QuantumAdapterConfig
from quantum_adapter.methods import METHOD_CLASSES

IN, OUT = 12, 20
METHODS = ["amplitude"]


def make(method: str, **kw):
    cfg = QuantumAdapterConfig(target_modules=["x"], method=method, rank=2, depth=3, tensor_axes=3, **kw)
    return METHOD_CLASSES[method](IN, OUT, cfg)


def perturb(m):
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.3 * torch.randn_like(p))


@pytest.mark.parametrize("method", METHODS)
def test_neutral_start(method):
    m = make(method)
    x = torch.randn(5, IN)
    assert torch.count_nonzero(m(x)) == 0
    assert torch.count_nonzero(m.delta_weight()) == 0


@pytest.mark.parametrize("method", METHODS)
def test_forward_equals_x_times_delta_weight(method):
    m = make(method)
    perturb(m)
    x = torch.randn(3, 4, IN)
    assert m(x).shape == (3, 4, OUT)
    assert torch.allclose(m(x), x @ m.delta_weight(), atol=1e-5)


@pytest.mark.parametrize("method", METHODS)
def test_every_parameter_learns_after_one_step(method):
    m = make(method)
    x, target = torch.randn(8, IN), torch.randn(8, OUT)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    for _ in range(2):
        opt.zero_grad()
        ((m(x) - target) ** 2).mean().backward()
        opt.step()
    dead = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert dead == [], f"no gradient reached {dead}"


@pytest.mark.parametrize("method", METHODS)
def test_learns_a_linear_map(method):
    torch.manual_seed(0)
    m = make(method)
    x = torch.randn(64, IN)
    target = x @ (0.3 * torch.randn(IN, OUT))
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    first = None
    for _ in range(40):
        opt.zero_grad()
        loss = ((m(x) - target) ** 2).mean()
        first = first if first is not None else loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < 0.9 * first


@pytest.mark.parametrize("method", METHODS)
def test_gradients_match_finite_differences(method):
    m = make(method).double()
    perturb(m)
    names = [n for n, _ in m.named_parameters()]
    x = torch.randn(3, IN, dtype=torch.float64)

    def f(*params):
        return functional_call(m, dict(zip(names, params)), (x,))

    assert torch.autograd.gradcheck(f, tuple(p.detach().requires_grad_() for p in m.parameters()))


@pytest.mark.parametrize("method", METHODS)
def test_two_forwards_one_backward(method):
    m = make(method)
    perturb(m)
    x = torch.randn(4, IN)
    (m(x).sum() + m(x).sum()).backward()        # gradient accumulation pattern must not raise


@pytest.mark.parametrize("method", METHODS)
def test_eval_cache_invalidates_after_step(method):
    m = make(method)
    perturb(m)
    x = torch.randn(4, IN)
    with torch.no_grad():
        before = m(x).clone()
        assert torch.equal(m(x), before)        # cached, same result
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    opt.zero_grad()
    m(x).pow(2).sum().backward()
    opt.step()
    with torch.no_grad():
        assert not torch.allclose(m(x), before)


@pytest.mark.parametrize("method", METHODS)
def test_summary_keys(method):
    s = make(method).summary()
    assert set(s) == {"method", "qubits", "dials", "gates", "params"}
    assert s["params"] == sum(p.numel() for p in make(method).parameters())


def test_amplitude_sizes_for_gpt2_attention():
    cfg = QuantumAdapterConfig(target_modules=["c_attn"])
    m = METHOD_CLASSES["amplitude"](768, 2304, cfg)
    assert m.n_qubits == 14                       # 4 * (768 + 2304) = 12,288 <= 2**14
    assert sum(p.numel() for p in m.parameters()) == 141
    assert m.summary()["gates"] == 10 * (14 + 14)


def test_amplitude_rejects_too_few_qubits():
    cfg = QuantumAdapterConfig(target_modules=["c_attn"], qubits=13)
    with pytest.raises(ValueError, match="needs 12,288 amplitudes, but 13 qubits only give 8,192. Use qubits >= 14"):
        METHOD_CLASSES["amplitude"](768, 2304, cfg)


def test_amplitude_rejects_too_many_auto_qubits():
    cfg = QuantumAdapterConfig(target_modules=["x"], rank=64, max_qubits=16)
    with pytest.raises(ValueError, match="max_qubits=16"):
        METHOD_CLASSES["amplitude"](4096, 4096, cfg)


def test_scaling_both_factors_would_never_learn():
    """Pins the bug from the first experiment: scale must multiply B only."""
    m = make("amplitude")
    a, b = m.factors()
    assert torch.count_nonzero(a) > 0 and torch.count_nonzero(b) == 0
    assert math.isclose(m.scale.item(), 0.0)
```

- [ ] **Step 2: Run to see it fail**

Run: `uv run pytest tests/test_methods.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'quantum_adapter.methods'`

- [ ] **Step 3: Implement** — `src/quantum_adapter/methods/base.py`

```python
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
        """Recompute under autograd (fresh graph per forward); reuse under no_grad until a parameter changes."""
        if torch.is_grad_enabled():
            return compute()
        key = tuple((p.data_ptr(), p._version, p.dtype) for p in self.parameters())
        if key != self._cache_key:
            self._cache, self._cache_key = compute(), key
        return self._cache
```

`src/quantum_adapter/methods/amplitude.py`

```python
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
```

`src/quantum_adapter/methods/__init__.py`

```python
from __future__ import annotations

from quantum_adapter.methods.amplitude import AmplitudeMethod
from quantum_adapter.methods.base import AdapterMethod, CircuitSpec

METHOD_CLASSES: dict[str, type[AdapterMethod]] = {"amplitude": AmplitudeMethod}

# Methods that did not pass the release benchmark (docs/benchmarks.md). Set in Task 10.
EXPERIMENTAL: frozenset[str] = frozenset()

__all__ = ["METHOD_CLASSES", "EXPERIMENTAL", "AdapterMethod", "AmplitudeMethod", "CircuitSpec"]
```

- [ ] **Step 4: Run to see it pass**

Run: `uv run pytest tests/test_methods.py -q`
Expected: all passed. If `test_amplitude_sizes_for_gpt2_attention` gates count differs, the ring for n=14 has 14 pairs: `10 * (14 + 14) = 280`.

- [ ] **Step 5: Commit**

```bash
git add src/quantum_adapter/methods tests/test_methods.py
git commit -m "feat: amplitude adapter method"
```

---

### Task 4: `unitary` and `tensor` methods

**Files:**
- Create: `src/quantum_adapter/methods/unitary.py`, `src/quantum_adapter/methods/tensor.py`, `tests/test_unitary.py`, `tests/test_tensor.py`
- Modify: `src/quantum_adapter/methods/__init__.py`, `tests/test_methods.py:9` (`METHODS = ["amplitude", "unitary", "tensor"]`)

**Interfaces:**
- Consumes: `AdapterMethod`, `CircuitSpec` (Task 3); `basis_states`, `entangling_pairs` (Task 2).
- Produces:
  - `UnitaryMethod(in, out, config)`; attributes `n_in`, `n_out`, `theta_u (depth+1, n_in)`, `theta_v (depth+1, n_out)`, `lam (rank,)`; `frames() -> tuple[Tensor, Tensor]` = `U (in, rank)`, `V (out, rank)` with orthonormal columns. `delta_weight() = U diag(lam) Vᵀ`.
  - `balanced_factors(d: int, k: int) -> list[int]`; `TensorMethod(in, out, config)`; attributes `in_dims`, `out_dims`, `pairs`, `gates: nn.ParameterList`, buffers `frozen_0..`.

- [ ] **Step 1: Write the failing tests**

`tests/test_unitary.py`

```python
import pytest
import torch

from quantum_adapter import QuantumAdapterConfig
from quantum_adapter.methods import METHOD_CLASSES


def make(i, o, **kw):
    return METHOD_CLASSES["unitary"](i, o, QuantumAdapterConfig(target_modules=["x"], method="unitary", **kw))


@pytest.mark.parametrize("shape", [(16, 32), (12, 20), (768, 2304)])
def test_frames_are_orthonormal(shape):
    m = make(*shape)
    u, v = m.frames()
    assert u.shape == (shape[0], 4) and v.shape == (shape[1], 4)
    assert torch.allclose(u.T @ u, torch.eye(4), atol=1e-4)
    assert torch.allclose(v.T @ v, torch.eye(4), atol=1e-4)


def test_parameter_count_for_gpt2_attention():
    m = make(768, 2304)
    assert (m.n_in, m.n_out) == (10, 12)
    assert sum(p.numel() for p in m.parameters()) == 11 * (10 + 12) + 4   # 246


def test_rank_larger_than_layer_is_rejected():
    with pytest.raises(ValueError, match="rank=8 is larger than the layer's smaller side \\(6\\)"):
        make(6, 20, rank=8)


def test_qubit_limit():
    with pytest.raises(ValueError, match="max_qubits=8"):
        make(1024, 1024, max_qubits=8)
```

`tests/test_tensor.py`

```python
import pytest
import torch

from quantum_adapter import QuantumAdapterConfig
from quantum_adapter.methods import METHOD_CLASSES
from quantum_adapter.methods.tensor import balanced_factors


@pytest.mark.parametrize(
    "d, k, expected",
    [(12, 2, [3, 4]), (20, 2, [4, 5]), (12, 3, [2, 2, 3]), (20, 3, [2, 2, 5]), (768, 4, [6, 4, 4, 8])],
)
def test_balanced_factors(d, k, expected):
    assert balanced_factors(d, k) == expected


def test_prime_size_is_rejected():
    cfg = QuantumAdapterConfig(target_modules=["x"], method="tensor", tensor_axes=2)
    with pytest.raises(ValueError, match="can't split 13 into 2 factors larger than 1"):
        METHOD_CLASSES["tensor"](13, 20, cfg)


def test_gate_shapes_chain_from_input_to_output():
    cfg = QuantumAdapterConfig(target_modules=["x"], method="tensor", tensor_axes=3)
    m = METHOD_CLASSES["tensor"](12, 20, cfg)
    assert m.in_dims == [2, 2, 3] and m.out_dims == [2, 2, 5]
    assert [tuple(g.shape) for g in m.gates] == [(2, 2, 2, 2), (2, 3, 2, 5), (2, 5, 2, 5)]


def test_frozen_copy_is_saved_but_not_trained():
    cfg = QuantumAdapterConfig(target_modules=["x"], method="tensor", tensor_axes=3)
    m = METHOD_CLASSES["tensor"](12, 20, cfg)
    assert {n for n, _ in m.named_buffers()} == {"frozen_0", "frozen_1", "frozen_2"}
    assert all(not b.requires_grad for b in m.buffers())
    assert "frozen_0" in m.state_dict()
```

Then set `METHODS = ["amplitude", "unitary", "tensor"]` in `tests/test_methods.py`.

- [ ] **Step 2: Run to see it fail**

Run: `uv run pytest tests -q`
Expected: FAIL, `KeyError: 'unitary'` / `ModuleNotFoundError: quantum_adapter.methods.tensor`

- [ ] **Step 3: Implement** — `src/quantum_adapter/methods/unitary.py`

```python
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
```

`src/quantum_adapter/methods/tensor.py`

```python
"""Tensor method (QuanTA, arXiv 2406.00132): the hidden vector's axes act like qudits and trainable
tensors act on pairs of axes, like two-qubit gates. Quantum-inspired: the gates are not unitary."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from quantum_adapter.config import QuantumAdapterConfig
from quantum_adapter.methods.base import AdapterMethod

_AXES = "abcdefgh"


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

    def _apply(self, gates: list[Tensor], x: Tensor) -> Tensor:
        lead = x.shape[:-1]
        t = x.reshape(-1, *self.in_dims)
        axes = _AXES[: len(self.in_dims)]
        for (a, b), g in zip(self.pairs, gates):
            src = "z" + axes
            dst = src.replace(axes[a], "x").replace(axes[b], "y")
            t = torch.einsum(f"{src},{axes[a]}{axes[b]}xy->{dst}", t, g.to(t.dtype))
        return t.reshape(*lead, self.out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self._apply(list(self.gates), x) - self._apply(self._frozen(), x)

    def delta_weight(self) -> Tensor:
        g = self.gates[0]
        return self.forward(torch.eye(self.in_features, dtype=g.dtype, device=g.device))

    def summary(self) -> dict:
        params = sum(p.numel() for p in self.parameters())
        return {"method": self.name, "qubits": "-", "dials": params, "gates": len(self.pairs), "params": params}
```

Update `src/quantum_adapter/methods/__init__.py`:

```python
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

# Methods that did not pass the release benchmark (docs/benchmarks.md). Set in Task 10.
EXPERIMENTAL: frozenset[str] = frozenset()

__all__ = [
    "METHOD_CLASSES", "EXPERIMENTAL", "AdapterMethod", "AmplitudeMethod", "CircuitSpec",
    "TensorMethod", "UnitaryMethod",
]
```

- [ ] **Step 4: Run to see it pass**

Run: `uv run pytest tests -q`
Expected: all passed. `test_eval_cache_invalidates_after_step` for `tensor` passes without caching because its forward has no simulation.

- [ ] **Step 5: Commit**

```bash
git add src/quantum_adapter/methods tests
git commit -m "feat: unitary (Quantum-PEFT) and tensor (QuanTA) methods"
```

---

### Task 5: Adapter layer, model wrapper, freezing, merge

**Files:**
- Create: `src/quantum_adapter/layers.py`, `src/quantum_adapter/model.py`, `tests/conftest.py`, `tests/test_model.py`
- Modify: `src/quantum_adapter/__init__.py`

**Interfaces:**
- Consumes: `METHOD_CLASSES`, `EXPERIMENTAL`, `AdapterMethod` (Tasks 3–4); `QuantumAdapterConfig` (Task 1).
- Produces:
  - `SUPPORTED_LAYERS`, `layer_shape(module) -> tuple[int, int]`, `AdapterLayer(base, adapter)` with `.base`, `.adapter`, `merge() -> nn.Module`.
  - `inject_adapters(model, target_modules, make_method: Callable[[int, int], AdapterMethod]) -> list[str]`
  - `get_quantum_model(model, config) -> QuantumModel`
  - `QuantumModel(nn.Module)`: `.base_model`, `.quantum_config`, `.adapted_modules: list[str]`, `forward(*a, **kw)`, attribute fallback to the base model, `adapter_layers() -> list[tuple[str, AdapterLayer]]`, `trainable_state_dict() -> dict[str, Tensor]`, `print_trainable_parameters()`, `circuit_summary() -> list[dict]`, `merge_and_unload() -> nn.Module`.

- [ ] **Step 1: Shared tiny models** — `tests/conftest.py`

```python
import pytest
import torch
from transformers import GPT2Config, GPT2ForSequenceClassification, LlamaConfig, LlamaForCausalLM


@pytest.fixture
def tiny_gpt2():
    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=2, n_embd=32, n_head=2, n_positions=32, vocab_size=64, num_labels=2, pad_token_id=0)
    return GPT2ForSequenceClassification(cfg).eval()


@pytest.fixture
def tiny_llama():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, vocab_size=64, max_position_embeddings=32,
    )
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture
def ids():
    torch.manual_seed(1)
    return torch.randint(1, 64, (3, 10))
```

- [ ] **Step 2: Write the failing tests** — `tests/test_model.py`

```python
import pytest
import torch

from quantum_adapter import QuantumAdapterConfig, get_quantum_model
from quantum_adapter.layers import AdapterLayer

METHODS = ["amplitude", "unitary", "tensor"]


def cfg(method="amplitude", **kw):
    kw.setdefault("target_modules", ["c_attn"])
    return QuantumAdapterConfig(method=method, rank=2, depth=2, tensor_axes=2, **kw)


def perturb(qm):
    with torch.no_grad():
        for p in qm.parameters():
            if p.requires_grad:
                p.add_(0.3 * torch.randn_like(p))


@pytest.mark.parametrize("method", METHODS)
def test_wraps_targets_and_starts_neutral(tiny_gpt2, ids, method):
    before = tiny_gpt2(ids).logits.clone()
    qm = get_quantum_model(tiny_gpt2, cfg(method))
    assert qm.adapted_modules == ["transformer.h.0.attn.c_attn", "transformer.h.1.attn.c_attn"]
    assert isinstance(qm.base_model.transformer.h[0].attn.c_attn, AdapterLayer)
    assert torch.equal(qm(ids).logits, before)


def test_only_adapters_and_modules_to_save_train(tiny_gpt2):
    qm = get_quantum_model(tiny_gpt2, cfg(modules_to_save=["score"]))
    trainable = {n for n, p in qm.named_parameters() if p.requires_grad}
    assert trainable and all(".adapter." in n or n.startswith("base_model.score.") for n in trainable)
    assert "base_model.score.weight" in trainable


def test_llama_linear_layers(tiny_llama, ids):
    qm = get_quantum_model(tiny_llama, cfg(target_modules=["q_proj", "v_proj"]))
    assert len(qm.adapted_modules) == 4
    assert qm(ids).logits.shape == (3, 10, 64)


def test_regex_targets(tiny_gpt2):
    qm = get_quantum_model(tiny_gpt2, cfg(target_modules=r"transformer\.h\.0\.attn\.c_attn"))
    assert qm.adapted_modules == ["transformer.h.0.attn.c_attn"]


def test_no_match_lists_available_layers(tiny_gpt2):
    with pytest.raises(ValueError, match="No layers matched target_modules=\\['q_proj'\\].*c_attn.*c_fc.*c_proj"):
        get_quantum_model(tiny_gpt2, cfg(target_modules=["q_proj"]))


def test_rejects_unsupported_layer(tiny_gpt2):
    with pytest.raises(TypeError, match="'transformer.wte' is a Embedding"):
        get_quantum_model(tiny_gpt2, cfg(target_modules=["wte"]))


def test_rejects_double_adapt(tiny_gpt2):
    qm = get_quantum_model(tiny_gpt2, cfg())
    with pytest.raises(ValueError, match="already has quantum adapters"):
        get_quantum_model(qm.base_model, cfg())


def test_modules_to_save_must_match(tiny_gpt2):
    with pytest.raises(ValueError, match="modules_to_save=\\['classifier'\\] matched no module"):
        get_quantum_model(tiny_gpt2, cfg(modules_to_save=["classifier"]))


def test_experimental_method_needs_opt_in(tiny_gpt2, monkeypatch):
    monkeypatch.setattr("quantum_adapter.model.EXPERIMENTAL", frozenset({"tensor"}))
    with pytest.raises(ValueError, match="experimental=True"):
        get_quantum_model(tiny_gpt2, cfg("tensor"))
    with pytest.warns(UserWarning, match="experimental"):
        get_quantum_model(tiny_gpt2, cfg("tensor", experimental=True))


def test_user_config_is_not_mutated(tiny_gpt2):
    c = cfg()
    get_quantum_model(tiny_gpt2, c)
    assert c.base_model_name_or_path is None


@pytest.mark.parametrize("method", METHODS)
def test_merge_matches_unmerged(tiny_gpt2, ids, method):
    qm = get_quantum_model(tiny_gpt2, cfg(method))
    perturb(qm)
    with torch.no_grad():
        expected = qm(ids).logits
        merged = qm.merge_and_unload()
        assert not any(isinstance(m, AdapterLayer) for m in merged.modules())
        assert torch.allclose(merged(ids).logits, expected, atol=1e-5)


@pytest.mark.parametrize("method", METHODS)
def test_merge_linear_layers(tiny_llama, ids, method):
    qm = get_quantum_model(tiny_llama, cfg(method, target_modules=["q_proj", "v_proj"]))
    perturb(qm)
    with torch.no_grad():
        expected = qm(ids).logits
        assert torch.allclose(qm.merge_and_unload()(ids).logits, expected, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_model(tiny_gpt2, ids, dtype):
    qm = get_quantum_model(tiny_gpt2.to(dtype), cfg())
    perturb(qm)
    out = qm(ids).logits
    assert out.dtype == dtype and torch.isfinite(out).all()
    out.float().sum().backward()


def _accelerator():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


@pytest.mark.skipif(_accelerator() is None, reason="no GPU")
def test_runs_on_accelerator(tiny_gpt2, ids):
    dev = _accelerator()
    qm = get_quantum_model(tiny_gpt2, cfg()).to(dev)
    perturb(qm)
    out = qm(ids.to(dev)).logits
    assert out.device.type == dev
    out.sum().backward()


def test_trainable_parameter_report(tiny_gpt2, capsys):
    qm = get_quantum_model(tiny_gpt2, cfg())
    qm.print_trainable_parameters()
    assert "trainable params:" in capsys.readouterr().out


def test_circuit_summary(tiny_gpt2, capsys):
    rows = get_quantum_model(tiny_gpt2, cfg()).circuit_summary()
    assert [r["layer"] for r in rows] == ["transformer.h.0.attn.c_attn", "transformer.h.1.attn.c_attn"]
    assert "LoRA rank 2 would train" in capsys.readouterr().out


def test_attribute_fallback(tiny_gpt2):
    qm = get_quantum_model(tiny_gpt2, cfg())
    assert qm.config is tiny_gpt2.config
```

- [ ] **Step 3: Run to see it fail**

Run: `uv run pytest tests/test_model.py -q`
Expected: FAIL, `ImportError: cannot import name 'get_quantum_model'`

- [ ] **Step 4: Implement** — `src/quantum_adapter/layers.py`

```python
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
```

`src/quantum_adapter/model.py`

```python
"""Finds target layers, freezes the base model, and adds save / load / merge / hub."""

from __future__ import annotations

import dataclasses
import re
import warnings
from collections.abc import Callable

from torch import Tensor, nn

from quantum_adapter.config import QuantumAdapterConfig
from quantum_adapter.layers import SUPPORTED_LAYERS, AdapterLayer, layer_shape
from quantum_adapter.methods import EXPERIMENTAL, METHOD_CLASSES, AdapterMethod


def _matches(name: str, targets: list[str] | str) -> bool:
    if isinstance(targets, str):
        return re.fullmatch(targets, name) is not None
    return any(name == t or name.endswith("." + t) for t in targets)


def _parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, child = name.rpartition(".")
    return (model.get_submodule(parent_name) if parent_name else model), child


def inject_adapters(
    model: nn.Module, target_modules: list[str] | str, make_method: Callable[[int, int], AdapterMethod]
) -> list[str]:
    if any(isinstance(m, AdapterLayer) for m in model.modules()):
        raise ValueError(
            "This model already has quantum adapters. Call get_quantum_model on the original model, "
            "or load saved adapters with QuantumModel.from_pretrained."
        )
    targets = [(n, m) for n, m in model.named_modules() if n and _matches(n, target_modules)]
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
    for name, module in targets:
        method = make_method(*layer_shape(module)).to(module.weight.device)
        parent, child = _parent(model, name)
        setattr(parent, child, AdapterLayer(module, method))
    return [n for n, _ in targets]


def _unfreeze_modules_to_save(model: nn.Module, names: list[str]) -> None:
    if not names:
        return
    found = [m for n, m in model.named_modules() if n and _matches(n, names)]
    if not found:
        raise ValueError(f"modules_to_save={names!r} matched no module in this model.")
    for m in found:
        m.requires_grad_(True)


def get_quantum_model(model: nn.Module, config: QuantumAdapterConfig) -> QuantumModel:
    """Add quantum adapters to `model` in place, freeze everything else, and wrap it."""
    config = dataclasses.replace(config)
    if config.method in EXPERIMENTAL:
        if not config.experimental:
            raise ValueError(
                f"method={config.method!r} is experimental: it did not pass the release benchmark "
                "(see docs/benchmarks.md). Set experimental=True in the config to use it anyway."
            )
        warnings.warn(f"method={config.method!r} is experimental; see docs/benchmarks.md.", UserWarning)
    if config.base_model_name_or_path is None:
        config.base_model_name_or_path = getattr(getattr(model, "config", None), "name_or_path", None) or None
    model.requires_grad_(False)
    method_cls = METHOD_CLASSES[config.method]
    adapted = inject_adapters(model, config.target_modules, lambda i, o: method_cls(i, o, config))
    _unfreeze_modules_to_save(model, config.modules_to_save)
    return QuantumModel(model, config, adapted)


class QuantumModel(nn.Module):
    def __init__(self, base_model: nn.Module, config: QuantumAdapterConfig, adapted_modules: list[str]) -> None:
        super().__init__()
        self.base_model = base_model
        self.quantum_config = config
        self.adapted_modules = adapted_modules

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "base_model":
                raise
            return getattr(self.base_model, name)

    def adapter_layers(self) -> list[tuple[str, AdapterLayer]]:
        return [(n, m) for n, m in self.base_model.named_modules() if isinstance(m, AdapterLayer)]

    def trainable_state_dict(self) -> dict[str, Tensor]:
        """Adapter parameters and buffers, plus modules_to_save, keyed as in base_model.state_dict()."""
        save = self.quantum_config.modules_to_save
        return {
            k: v for k, v in self.base_model.state_dict().items()
            if ".adapter." in k or (save and _matches(k.rsplit(".", 1)[0], save))
        }

    def print_trainable_parameters(self) -> None:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.4f}")

    def circuit_summary(self) -> list[dict]:
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

    def merge_and_unload(self) -> nn.Module:
        for name, layer in self.adapter_layers():
            parent, child = _parent(self.base_model, name)
            setattr(parent, child, layer.merge())
        return self.base_model
```

Update `src/quantum_adapter/__init__.py`:

```python
"""Fine-tune language models with adapters generated by quantum circuits."""

from __future__ import annotations

from quantum_adapter.config import QuantumAdapterConfig
from quantum_adapter.model import QuantumModel, get_quantum_model

__version__ = "0.1.0"
__all__ = ["QuantumAdapterConfig", "QuantumModel", "get_quantum_model", "__version__"]
```

- [ ] **Step 5: Run to see it pass**

Run: `uv run pytest tests -q`
Expected: all passed (`test_runs_on_accelerator` runs on this Mac via MPS). If the MPS run fails with "scaled_dot_product_attention for MPS does not support dropout", the tiny GPT-2 is in train mode: the fixture calls `.eval()`, keep it.

- [ ] **Step 6: Commit**

```bash
git add src tests
git commit -m "feat: adapter layers, model wrapper and merge"
```

---

### Task 6: Save, load and Hugging Face Hub

**Files:**
- Modify: `src/quantum_adapter/model.py`
- Create: `tests/test_save_load.py`

**Interfaces:**
- Produces: `CONFIG_NAME = "adapter_config.json"`, `WEIGHTS_NAME = "adapter_model.safetensors"`; `QuantumModel.save_pretrained(save_directory) -> None`; `QuantumModel.from_pretrained(model, path_or_repo_id, *, revision=None, is_trainable=False, token=None) -> QuantumModel`; `QuantumModel.push_to_hub(repo_id, *, private=True, token=None, commit_message="Upload quantum adapter") -> str` (commit URL).

- [ ] **Step 1: Write the failing tests** — `tests/test_save_load.py`

```python
import json

import pytest
import torch
from transformers import GPT2Config, GPT2ForSequenceClassification

from quantum_adapter import QuantumAdapterConfig, QuantumModel, __version__, get_quantum_model
from quantum_adapter import model as model_module

METHODS = ["amplitude", "unitary", "tensor"]


def fresh(seed=0, n_embd=32):
    torch.manual_seed(seed)
    cfg = GPT2Config(n_layer=2, n_embd=n_embd, n_head=2, n_positions=32, vocab_size=64, num_labels=2, pad_token_id=0)
    return GPT2ForSequenceClassification(cfg).eval()


def trained(method):
    qm = get_quantum_model(fresh(), QuantumAdapterConfig(
        target_modules=["c_attn"], method=method, rank=2, depth=2, tensor_axes=2, modules_to_save=["score"]))
    with torch.no_grad():
        for p in qm.parameters():
            if p.requires_grad:
                p.add_(0.3 * torch.randn_like(p))
    return qm


@pytest.mark.parametrize("method", METHODS)
def test_round_trip_gives_identical_logits(tmp_path, ids, method):
    qm = trained(method)
    expected = qm(ids).logits
    qm.save_pretrained(tmp_path)
    loaded = QuantumModel.from_pretrained(fresh(), tmp_path)
    assert torch.equal(loaded(ids).logits, expected)


def test_files_are_small_and_versioned(tmp_path):
    qm = trained("amplitude")
    qm.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "adapter_config.json").read_text())
    assert saved["library_version"] == __version__
    assert saved["layer_shapes"] == {"transformer.h.0.attn.c_attn": [32, 96], "transformer.h.1.attn.c_attn": [32, 96]}
    from safetensors.torch import load_file
    keys = set(load_file(tmp_path / "adapter_model.safetensors"))
    assert keys == set(qm.trainable_state_dict())
    assert all(".adapter." in k or k.startswith("score.") for k in keys)


def test_loaded_model_is_frozen_unless_trainable(tmp_path):
    trained("amplitude").save_pretrained(tmp_path)
    assert not any(p.requires_grad for p in QuantumModel.from_pretrained(fresh(), tmp_path).parameters())
    assert any(p.requires_grad for p in QuantumModel.from_pretrained(fresh(), tmp_path, is_trainable=True).parameters())


def test_wrong_base_model_is_rejected(tmp_path):
    trained("amplitude").save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="saved for a model with different layers.*saved \\[32, 96\\], this model \\[64, 192\\]"):
        QuantumModel.from_pretrained(fresh(n_embd=64), tmp_path)


def test_loads_from_hub_repo_id(tmp_path, monkeypatch, ids):
    qm = trained("unitary")
    qm.save_pretrained(tmp_path)
    calls = {}

    def fake_download(repo_id, **kw):
        calls["repo_id"], calls["kw"] = repo_id, kw
        return str(tmp_path)

    monkeypatch.setattr(model_module, "snapshot_download", fake_download)
    loaded = QuantumModel.from_pretrained(fresh(), "someone/adapter", revision="v1")
    assert calls["repo_id"] == "someone/adapter" and calls["kw"]["revision"] == "v1"
    assert torch.equal(loaded(ids).logits, qm(ids).logits)


def test_push_to_hub_uploads_both_files(monkeypatch):
    uploaded = {}

    class FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, repo_id, private, exist_ok):
            uploaded["private"] = private

        def upload_folder(self, repo_id, folder_path, commit_message):
            import os
            uploaded["files"] = sorted(os.listdir(folder_path))

            class Info:
                commit_url = f"https://huggingface.co/{repo_id}/commit/abc"
            return Info()

    monkeypatch.setattr(model_module, "HfApi", FakeApi)
    url = trained("amplitude").push_to_hub("someone/adapter")
    assert uploaded == {"private": True, "files": ["adapter_config.json", "adapter_model.safetensors"]}
    assert url.endswith("/commit/abc")
```

- [ ] **Step 2: Run to see it fail**

Run: `uv run pytest tests/test_save_load.py -q`
Expected: FAIL. Until this task, `qm.save_pretrained` falls through to GPT-2's own method and saves the whole model, so the round-trip tests fail on missing `adapter_config.json`.

- [ ] **Step 3: Implement** — add to `src/quantum_adapter/model.py`

Add imports at the top:

```python
import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file, save_file
```

Add constants after the imports:

```python
CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "adapter_model.safetensors"
```

Add these methods to `QuantumModel`:

```python
    def _layer_shapes(self) -> dict[str, list[int]]:
        return {n: [m.adapter.in_features, m.adapter.out_features] for n, m in self.adapter_layers()}

    def save_pretrained(self, save_directory: str | Path) -> None:
        from quantum_adapter import __version__

        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        meta = {**self.quantum_config.to_dict(), "library_version": __version__, "layer_shapes": self._layer_shapes()}
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
    ) -> QuantumModel:
        path = Path(path_or_repo_id)
        if not path.is_dir():
            path = Path(snapshot_download(
                str(path_or_repo_id), revision=revision, token=token, allow_patterns=[CONFIG_NAME, WEIGHTS_NAME]))
        meta = json.loads((path / CONFIG_NAME).read_text())
        saved_shapes = meta.pop("layer_shapes", None)
        qm = get_quantum_model(model, QuantumAdapterConfig.from_dict(meta))
        actual = qm._layer_shapes()
        if saved_shapes is not None and saved_shapes != actual:
            name = next(n for n in sorted(set(saved_shapes) | set(actual)) if saved_shapes.get(n) != actual.get(n))
            raise ValueError(
                f"This adapter was saved for a model with different layers (first difference: {name}: "
                f"saved {saved_shapes.get(name)}, this model {actual.get(name)}). "
                f"Load it onto the base model it was trained on: {qm.quantum_config.base_model_name_or_path}."
            )
        state = load_file(path / WEIGHTS_NAME)
        expected = set(qm.trainable_state_dict())
        if set(state) != expected:
            raise ValueError(
                f"Adapter weights don't match this config: missing {sorted(expected - set(state))[:5]}, "
                f"unexpected {sorted(set(state) - expected)[:5]}."
            )
        qm.base_model.load_state_dict(state, strict=False)
        if not is_trainable:
            qm.requires_grad_(False)
            qm.eval()
        return qm

    def push_to_hub(
        self, repo_id: str, *, private: bool = True, token: str | None = None,
        commit_message: str = "Upload quantum adapter",
    ) -> str:
        api = HfApi(token=token)
        api.create_repo(repo_id, private=private, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            self.save_pretrained(tmp)
            info = api.upload_folder(repo_id=repo_id, folder_path=tmp, commit_message=commit_message)
        return info.commit_url
```

- [ ] **Step 4: Run to see it pass**

Run: `uv run pytest tests -q && uv run ruff check .`
Expected: all passed; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "feat: save, load and push adapters to the Hugging Face Hub"
```

---

### Task 7: Works with the Hugging Face Trainer

**Files:**
- Create: `tests/test_trainer.py`

**Interfaces:**
- Consumes: `get_quantum_model`, `QuantumModel.save_pretrained`, `QuantumModel.from_pretrained` (Tasks 5–6).

- [ ] **Step 1: Write the test** — `tests/test_trainer.py`

```python
import torch
from transformers import Trainer, TrainingArguments

from quantum_adapter import QuantumAdapterConfig, QuantumModel, get_quantum_model


class Toy(torch.utils.data.Dataset):
    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.ids = torch.randint(1, 64, (32, 10), generator=g)
        self.labels = (self.ids[:, 0] > 32).long()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "labels": self.labels[i]}


def test_trainer_trains_and_adapter_reloads(tiny_gpt2, tmp_path):
    qm = get_quantum_model(tiny_gpt2, QuantumAdapterConfig(target_modules=["c_attn"], rank=2, depth=2,
                                                           modules_to_save=["score"]))
    before = {k: v.clone() for k, v in qm.trainable_state_dict().items()}
    args = TrainingArguments(output_dir=str(tmp_path / "run"), max_steps=4, per_device_train_batch_size=8,
                             learning_rate=1e-2, report_to=[], use_cpu=True, save_strategy="no")
    Trainer(model=qm, args=args, train_dataset=Toy()).train()
    after = qm.trainable_state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)
    qm.save_pretrained(tmp_path / "adapter")
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_trainer.py -q`
Expected: PASS. If Trainer raises because the model lacks `config` or `_keys_to_ignore_on_save`, the `__getattr__` fallback in `QuantumModel` should provide them; if a specific attribute is still missing, add a property forwarding it and a one-line comment naming the Trainer version that needs it.

- [ ] **Step 3: Commit**

```bash
git add tests/test_trainer.py
git commit -m "test: quantum adapters train with the Hugging Face Trainer"
```

---

### Task 8: Qiskit export

**Files:**
- Create: `src/quantum_adapter/export.py`, `tests/test_export.py`
- Modify: `src/quantum_adapter/model.py` (add `export_circuits`)

**Interfaces:**
- Consumes: `CircuitSpec` (Task 3), `entangling_pairs`, `run_ansatz` (Task 2).
- Produces: `to_qiskit(spec: CircuitSpec) -> qiskit.QuantumCircuit`; `QuantumModel.export_circuits() -> dict[str, list[QuantumCircuit]]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_export.py`

```python
import math

import numpy as np
import pytest
import torch

qiskit = pytest.importorskip("qiskit")
from qiskit.quantum_info import Statevector  # noqa: E402

from quantum_adapter import QuantumAdapterConfig, get_quantum_model  # noqa: E402
from quantum_adapter.export import to_qiskit  # noqa: E402
from quantum_adapter.methods import CircuitSpec  # noqa: E402
from quantum_adapter.sim import basis_states, run_ansatz  # noqa: E402


@pytest.mark.parametrize("pattern", ["ring", "line", "full"])
@pytest.mark.parametrize("gate", ["cx", "cz"])
@pytest.mark.parametrize("final", [False, True])
def test_simulator_matches_qiskit(pattern, gate, final):
    n = 5
    theta = torch.rand(3, n, dtype=torch.float64) * 2 * math.pi
    qc = to_qiskit(CircuitSpec("t", n, theta, pattern, gate, final))
    ours = run_ansatz(basis_states(4, n, dtype=torch.float64), theta, n,
                      entanglement=pattern, gate=gate, final_rotation=final).numpy()
    for j in range(4):
        sv = Statevector.from_int(j, 2**n).evolve(qc).data
        assert np.allclose(sv.imag, 0, atol=1e-10)
        assert np.allclose(sv.real, ours[j], atol=1e-10)


def test_model_export(tiny_gpt2):
    qm = get_quantum_model(tiny_gpt2, QuantumAdapterConfig(target_modules=["c_attn"], method="unitary", rank=2, depth=2))
    circuits = qm.export_circuits()
    assert list(circuits) == ["transformer.h.0.attn.c_attn", "transformer.h.1.attn.c_attn"]
    u, v = circuits["transformer.h.0.attn.c_attn"]
    assert (u.num_qubits, v.num_qubits) == (5, 7)       # 32 -> 5 qubits, 96 -> 7 qubits


def test_tensor_method_has_nothing_to_export(tiny_gpt2):
    qm = get_quantum_model(tiny_gpt2, QuantumAdapterConfig(target_modules=["c_attn"], method="tensor", tensor_axes=2))
    with pytest.raises(ValueError, match="quantum-inspired"):
        qm.export_circuits()
```

- [ ] **Step 2: Run to see it fail**

Run: `uv run pytest tests/test_export.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'quantum_adapter.export'`

- [ ] **Step 3: Implement** — `src/quantum_adapter/export.py`

```python
"""Turn trained circuits into Qiskit circuits (needs the [qiskit] extra)."""

from __future__ import annotations

from quantum_adapter.methods import CircuitSpec
from quantum_adapter.sim import entangling_pairs


def to_qiskit(spec: CircuitSpec):
    try:
        from qiskit import QuantumCircuit
    except ImportError as e:
        raise ImportError("Qiskit export needs the optional extra: pip install 'quantum-adapter[qiskit]'") from e
    n = spec.n_qubits
    qc = QuantumCircuit(n, name=spec.name)

    def wire(q: int) -> int:          # our qubit 0 is the most significant bit; Qiskit's is the least
        return n - 1 - q

    pairs = entangling_pairs(n, spec.entanglement)
    layers = spec.theta.shape[0]
    for k in range(layers):
        for q in range(n):
            qc.ry(float(spec.theta[k, q]), wire(q))
        if not (spec.final_rotation and k == layers - 1):
            for a, b in pairs:
                (qc.cx if spec.gate == "cx" else qc.cz)(wire(a), wire(b))
    return qc
```

Add to `QuantumModel` in `model.py`:

```python
    def export_circuits(self) -> dict[str, list]:
        """Each adapted layer's trained circuits as Qiskit QuantumCircuits."""
        from quantum_adapter.export import to_qiskit

        method_cls = METHOD_CLASSES[self.quantum_config.method]
        if not method_cls.quantum_gates:
            raise ValueError(
                f"method={self.quantum_config.method!r} is quantum-inspired (its gates aren't unitary), so it has "
                "no quantum circuit to export. Use method='amplitude' or 'unitary'."
            )
        return {n: [to_qiskit(spec) for spec in m.adapter.circuits()] for n, m in self.adapter_layers()}
```

- [ ] **Step 4: Run to see it pass**

Run: `uv run pytest tests -q`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "feat: export trained circuits to Qiskit"
```

---

### Task 9: Release benchmark

**Files:**
- Create: `benchmarks/controls.py`, `benchmarks/run.py`, `benchmarks/gate.py`, `tests/test_controls.py`

**Interfaces:**
- Consumes: `AdapterMethod`, `METHOD_CLASSES` (Tasks 3–4), `inject_adapters`, `get_quantum_model` (Task 5).
- Produces: `RandomProjectionMethod(in, out, config, n_params)`; CLI `benchmarks/run.py --task {sst2,ag_news} --contenders ... --n-train N --out FILE` writing one JSON object per line with keys `task, contender, lr, seed, trainable, adapter_params, dev_acc, eval_acc, train_s, infer_s`; CLI `benchmarks/gate.py FILE... --markdown docs/benchmarks.md` printing a verdict per method and exiting 0.

- [ ] **Step 1: Write the failing control test** — `tests/test_controls.py`

```python
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from controls import RandomProjectionMethod  # noqa: E402

from quantum_adapter import QuantumAdapterConfig  # noqa: E402


def test_control_matches_param_count_and_starts_neutral():
    m = RandomProjectionMethod(12, 20, QuantumAdapterConfig(target_modules=["x"], rank=2), n_params=61)
    assert sum(p.numel() for p in m.parameters()) == 61
    x = torch.randn(3, 12)
    assert torch.count_nonzero(m(x)) == 0
    assert torch.allclose(m(x), x @ m.delta_weight())
```

- [ ] **Step 2: Run to see it fail**

Run: `uv run pytest tests/test_controls.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'controls'`

- [ ] **Step 3: Implement** — `benchmarks/controls.py`

```python
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
```

`benchmarks/run.py`

```python
"""Release benchmark: one task, many contenders, LR picked on a dev split, then 3 seeds.

Usage:
  uv run --group bench python benchmarks/run.py --task sst2 --out results/bench-sst2.jsonl
(The script's own folder is on sys.path, so `controls` imports from any working directory.)
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from controls import RandomProjectionMethod
from datasets import load_dataset
from transformers import GPT2ForSequenceClassification, GPT2TokenizerFast

from quantum_adapter import QuantumAdapterConfig, get_quantum_model
from quantum_adapter.methods import METHOD_CLASSES
from quantum_adapter.model import inject_adapters

DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
TASKS = {
    "sst2": dict(path="nyu-mll/glue", name="sst2", text="sentence", labels=2, eval_split="validation", maxlen=64,
                 eval_limit=None),
    "ag_news": dict(path="fancyzhx/ag_news", name=None, text="text", labels=4, eval_split="test", maxlen=128,
                    eval_limit=2000),
}
LRS = {
    "head": [None], "lora": [3e-4, 1e-3, 3e-3], "tensor": [3e-4, 1e-3, 3e-3], "control:tensor": [3e-4, 1e-3, 3e-3],
    "amplitude": [3e-3, 1e-2, 3e-2], "unitary": [3e-3, 1e-2, 3e-2],
    "control:amplitude": [3e-3, 1e-2, 3e-2], "control:unitary": [3e-3, 1e-2, 3e-2],
}
HEAD_LR, EPOCHS, BS, N_DEV, RANK = 1e-3, 3, 32, 500, 4


def load(task: dict, n_train: int):
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    ds = load_dataset(task["path"], task["name"])
    pool = ds["train"].shuffle(seed=0)
    ev = ds[task["eval_split"]]
    if task["eval_limit"]:
        ev = ev.shuffle(seed=0).select(range(task["eval_limit"]))

    def enc(split):
        e = tok(list(split[task["text"]]), truncation=True, max_length=task["maxlen"], padding="max_length",
                return_tensors="pt")
        return e["input_ids"], e["attention_mask"], torch.tensor(list(split["label"]))

    return enc(pool.select(range(n_train))), enc(pool.select(range(n_train, n_train + N_DEV))), enc(ev), tok


def base_model(task: dict, tok):
    m = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=task["labels"], attn_pdrop=0.0)
    m.config.pad_token_id = tok.eos_token_id
    return m


def qcfg(method: str) -> QuantumAdapterConfig:
    return QuantumAdapterConfig(target_modules=["c_attn"], method=method, rank=RANK, modules_to_save=["score"],
                                experimental=True)


def build(contender: str, task: dict, tok):
    m = base_model(task, tok)
    if contender == "head":
        m.requires_grad_(False)
        m.score.requires_grad_(True)
        return m
    if contender == "lora" or contender == "control:tensor":
        from peft import LoraConfig, get_peft_model
        rank = RANK
        if contender == "control:tensor":   # LoRA at the rank whose parameter count is nearest the tensor method's
            per_layer = METHOD_CLASSES["tensor"](768, 2304, qcfg("tensor")).summary()["params"]
            rank = max(1, round(per_layer / (768 + 2304)))
        pm = get_peft_model(m, LoraConfig(r=rank, target_modules=["c_attn"], fan_in_fan_out=True, lora_dropout=0.0,
                                          modules_to_save=["score"]))
        return pm
    if contender.startswith("control:"):
        method = contender.split(":", 1)[1]
        cfg = qcfg(method)
        n_params = METHOD_CLASSES[method](768, 2304, cfg).summary()["params"]
        m.requires_grad_(False)
        inject_adapters(m, ["c_attn"], lambda i, o: RandomProjectionMethod(i, o, cfg, n_params))
        m.score.requires_grad_(True)
        return m
    return get_quantum_model(m, qcfg(contender))


@torch.no_grad()
def accuracy(model, data) -> tuple[float, float]:
    model.eval()
    ids, mask, y = data
    correct, t0 = 0, time.time()
    for i in range(0, len(y), 128):
        out = model(input_ids=ids[i:i + 128].to(DEV), attention_mask=mask[i:i + 128].to(DEV)).logits
        correct += (out.argmax(-1).cpu() == y[i:i + 128]).sum().item()
    if DEV == "mps":
        torch.mps.synchronize()
    return correct / len(y), time.time() - t0


def adapter_param_count(model) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "score" not in n)


def run(contender: str, lr: float | None, seed: int, task_name: str, data) -> dict:
    task = TASKS[task_name]
    train, dev, ev, tok = data
    torch.manual_seed(seed)
    model = build(contender, task, tok).to(DEV)
    head = [p for n, p in model.named_parameters() if p.requires_grad and "score" in n]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and "score" not in n]
    groups = [{"params": head, "lr": HEAD_LR}] + ([{"params": rest, "lr": lr}] if rest else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    ids, mask, y = train
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(y), generator=g)
        for i in range(0, len(y), BS):
            b = perm[i:i + BS]
            out = model(input_ids=ids[b].to(DEV), attention_mask=mask[b].to(DEV), labels=y[b].to(DEV))
            opt.zero_grad()
            out.loss.backward()
            opt.step()
    if DEV == "mps":
        torch.mps.synchronize()
    train_s = time.time() - t0
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_adapter = adapter_param_count(model)
    dev_acc, _ = accuracy(model, dev)
    merged = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
    eval_acc, infer_s = accuracy(merged, ev)
    return {"task": task_name, "contender": contender, "lr": lr, "seed": seed, "trainable": trainable,
            "adapter_params": n_adapter, "dev_acc": dev_acc, "eval_acc": eval_acc,
            "train_s": round(train_s, 1), "infer_s": round(infer_s, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--contenders", nargs="+", default=list(LRS))
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = load(TASKS[a.task], a.n_train)
    with open(a.out, "a") as f:
        for c in a.contenders:
            best = None
            for lr in LRS[c]:
                r = run(c, lr, 0, a.task, data)
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(json.dumps(r), flush=True)
                if best is None or r["dev_acc"] > best["dev_acc"]:
                    best = r
            for seed in (1, 2):
                r = run(c, best["lr"], seed, a.task, data)
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
```

`benchmarks/gate.py`

```python
"""Apply the production-usefulness rule (spec section 1) to benchmark results.

Usage: uv run python benchmarks/gate.py results/bench-sst2.jsonl results/bench-ag_news.jsonl --markdown docs/benchmarks.md
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict

METHODS = ["amplitude", "unitary", "tensor"]


def best_runs(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Per (task, contender): the 3 runs at the LR that won on the dev split for seed 0."""
    by = defaultdict(list)
    for r in rows:
        by[(r["task"], r["contender"])].append(r)
    out = {}
    for key, rs in by.items():
        seed0 = [r for r in rs if r["seed"] == 0]
        lr = max(seed0, key=lambda r: r["dev_acc"])["lr"]
        out[key] = [r for r in rs if r["lr"] == lr]
    return out


def mean(rs: list[dict], k: str) -> float:
    return st.mean(r[k] for r in rs)


def verdicts(runs: dict) -> dict[str, dict]:
    tasks = sorted({t for t, _ in runs})
    result = {}
    for m in METHODS:
        checks = defaultdict(list)
        for t in tasks:
            q, lora, head, ctrl = runs[(t, m)], runs[(t, "lora")], runs[(t, "head")], runs[(t, f"control:{m}")]
            checks["accuracy within 1.0 of LoRA"].append(mean(q, "eval_acc") >= mean(lora, "eval_acc") - 0.01)
            checks["≥5x fewer params than LoRA"].append(lora[0]["adapter_params"] >= 5 * q[0]["adapter_params"])
            checks["train time ≤2x LoRA"].append(mean(q, "train_s") <= 2 * mean(lora, "train_s"))
            checks["merged inference within 10% of base"].append(mean(q, "infer_s") <= 1.1 * mean(head, "infer_s"))
            checks["beats same-size classical control"].append(mean(q, "eval_acc") > mean(ctrl, "eval_acc"))
        passed = {name: all(v) for name, v in checks.items()}
        acc = st.mean(mean(runs[(t, m)], "eval_acc") for t in tasks)
        result[m] = {"checks": passed, "stable": all(passed.values()), "mean_acc": acc}
    return result


def table(runs: dict) -> str:
    lines = []
    for t in sorted({t for t, _ in runs}):
        lines += [f"\n### {t}\n", "| Contender | Adapter params | Accuracy (mean, min–max) | Train s | Inference s |",
                  "|---|---|---|---|---|"]
        for (task, c), rs in sorted(runs.items()):
            if task != t:
                continue
            v = [100 * r["eval_acc"] for r in rs]
            lines.append(f"| {c} | {rs[0]['adapter_params']:,} | {st.mean(v):.1f}% ({min(v):.1f}–{max(v):.1f}) | "
                         f"{mean(rs, 'train_s'):.0f} | {mean(rs, 'infer_s'):.1f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--markdown")
    a = ap.parse_args()
    rows = [json.loads(line) for f in a.files for line in open(f) if line.strip()]
    runs = best_runs(rows)
    v = verdicts(runs)
    stable = [m for m in METHODS if v[m]["stable"]]
    default = max(stable, key=lambda m: v[m]["mean_acc"]) if stable else None
    report = ["# Benchmark results", "", "GPT-2 small, adapters on `c_attn`, 3 seeds at the dev-selected LR.",
              table(runs), "", "## Verdicts", ""]
    for m in METHODS:
        report.append(f"- **{m}**: {'stable' if v[m]['stable'] else 'experimental'} — " + ", ".join(
            f"{name} {'✅' if ok else '❌'}" for name, ok in v[m]["checks"].items()))
    report += ["", f"Default method: **{default or 'none passed: release blocked'}**", ""]
    text = "\n".join(report)
    print(text)
    if a.markdown:
        open(a.markdown, "w").write(text)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the control test and a smoke benchmark**

Run:
```bash
uv run pytest tests/test_controls.py -q
rm -f /tmp/smoke.jsonl
uv run --group bench python benchmarks/run.py --task sst2 --n-train 64 --contenders head lora amplitude control:amplitude --out /tmp/smoke.jsonl
```
Expected: the test passes; the smoke run prints 18 JSON lines (head 3, then 5 for each other contender: 3 LRs on seed 0 plus seeds 1 and 2), each with a finite `eval_acc`. Accuracy is meaningless at 64 examples; this only checks the plumbing. (`gate.py` needs every contender on every task, so it is first run in Task 10.)

- [ ] **Step 5: Commit**

```bash
git add benchmarks tests/test_controls.py
git commit -m "feat: release benchmark with classical controls and production gate"
```

---

### Task 10: Run the release benchmark and apply the rule

**Files:**
- Create: `results/bench-sst2.jsonl`, `results/bench-ag_news.jsonl`, `docs/benchmarks.md`
- Modify: `src/quantum_adapter/methods/__init__.py` (`EXPERIMENTAL`), `src/quantum_adapter/config.py` (default `method`), `tests/test_config.py` (default assertion)

- [ ] **Step 1: Run both tasks** (several hours; run in the background)

```bash
uv run --group bench python benchmarks/run.py --task sst2 --out results/bench-sst2.jsonl
uv run --group bench python benchmarks/run.py --task ag_news --out results/bench-ag_news.jsonl
```
Expected: SST-2 38 lines, AG News 38 lines (head 3 + 7 contenders × 5).

- [ ] **Step 2: Apply the rule**

Run: `uv run python benchmarks/gate.py results/bench-sst2.jsonl results/bench-ag_news.jsonl --markdown docs/benchmarks.md`

- [ ] **Step 3: Decide**

- If the report says `none passed: release blocked`: **stop**. Commit the results and report them to the project owner; do not continue to Task 12.
- Otherwise set, in `src/quantum_adapter/methods/__init__.py`, `EXPERIMENTAL = frozenset({...})` to exactly the methods listed as experimental (e.g. `frozenset({"tensor"})`), and in `config.py` set `method: str = "<default from the report>"`. Update `test_defaults` in `tests/test_config.py` to assert that default.

- [ ] **Step 4: Run all tests**

Run: `uv run pytest tests -q`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add results docs/benchmarks.md src tests
git commit -m "feat: apply release benchmark verdicts"
```

---

### Task 11: Docs, example, CI

**Files:**
- Create: `examples/quickstart.py`, `docs/how-it-works.md`, `CHANGELOG.md`, `.github/workflows/tests.yml`
- Modify: `README.md`

- [ ] **Step 1: Quick start example** — `examples/quickstart.py`

```python
"""Quick start. `--tiny` uses a small random GPT-2 so it runs in seconds without downloads."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2ForSequenceClassification

from quantum_adapter import QuantumAdapterConfig, QuantumModel, get_quantum_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    if a.tiny:
        model = GPT2ForSequenceClassification(GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=64,
                                                         num_labels=2, pad_token_id=0))
        batch = {"input_ids": torch.randint(1, 64, (4, 12)), "labels": torch.tensor([0, 1, 0, 1])}
    else:
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
        model.config.pad_token_id = tok.eos_token_id
        batch = dict(tok(["a wonderful film", "tiresomely dull"], return_tensors="pt", padding=True))
        batch["labels"] = torch.tensor([1, 0])

    qm = get_quantum_model(model, QuantumAdapterConfig(target_modules=["c_attn"], modules_to_save=["score"]))
    qm.circuit_summary()
    opt = torch.optim.AdamW([p for p in qm.parameters() if p.requires_grad], lr=1e-2)
    for step in range(5):
        loss = qm(**batch).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step {step}: loss {loss.item():.4f}")
    qm.save_pretrained("quickstart-adapter")
    QuantumModel.from_pretrained(model.__class__(model.config), "quickstart-adapter")
    print("saved and reloaded: quickstart-adapter/")


if __name__ == "__main__":
    main()
```

Run: `uv run python examples/quickstart.py --tiny && rm -rf quickstart-adapter`
Expected: a summary table, 5 loss lines, "saved and reloaded".

Note: reloading onto `model.__class__(model.config)` uses a fresh random base, which is fine for the tiny demo; the README shows the real pattern (load the same pretrained base).

- [ ] **Step 2: README** — replace `README.md` with: title and one-line gist; "How it works" (the 4-step diagram from the spec conversation: frozen model → correction → circuit makes the correction → merge); install (`pip install quantum-adapter`, `[qiskit]`); quick start code from spec §3; the three methods table with their papers and stable/experimental status from `docs/benchmarks.md`; the results table copied from `docs/benchmarks.md`; honest limits (simulator training, real hardware via export only, ≤ ~1B-parameter text models, slower training); citations (BibTeX for arXiv 2503.05431, 2406.00132, and a mention of 2503.12790); licence. Keep the first-experiment tables under "History".

- [ ] **Step 3: `docs/how-it-works.md`** — for beginners: towers picture (2ⁿ amplitudes), what RY and CNOT/CZ do, how dials become ΔW per method, why neutral start matters, why merge makes inference free, what the classical control proves. One short paragraph and one diagram per idea.

- [ ] **Step 4: `CHANGELOG.md`**

```markdown
# Changelog

## 0.1.0 (unreleased)

- First release: `amplitude`, `unitary` (Quantum-PEFT) and `tensor` (QuanTA) adapter methods.
- Save / load / push to the Hugging Face Hub, merge for zero-overhead inference, Qiskit export.
- Release benchmark on SST-2 and AG News; see docs/benchmarks.md.
```

- [ ] **Step 5: CI** — `.github/workflows/tests.yml`

```yaml
name: tests
on:
  push:
  pull_request:
jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python: ["3.10", "3.11", "3.12", "3.13"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          python-version: ${{ matrix.python }}
      - run: uv sync --group dev
      - run: uv run ruff check .
      - run: uv run pytest -q
      - run: uv run python examples/quickstart.py --tiny
```

- [ ] **Step 6: Verify locally**

Run: `uv run ruff check . && uv run pytest -q && uv build && ls dist`
Expected: clean, all passed, `quantum_adapter-0.1.0-py3-none-any.whl` and `quantum_adapter-0.1.0.tar.gz`.

Run: `uv venv /tmp/qa-clean && VIRTUAL_ENV=/tmp/qa-clean uv pip install dist/*.whl && /tmp/qa-clean/bin/python examples/quickstart.py --tiny`
Expected: works from the built wheel alone.

- [ ] **Step 7: Check for forbidden references, then commit**

```bash
grep -rniE "co-authored-by|generated with" --exclude-dir=.venv --exclude-dir=.git . || echo clean
git log --all --format=%B | grep -iE "co-authored-by|generated with" || echo clean
git add README.md docs examples CHANGELOG.md .github
git commit -m "docs: readme, how-it-works, quick start and CI"
```

---

### Task 12: Release pipeline (publishing waits for the owner)

**Files:**
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Release workflow** — `.github/workflows/release.yml`

```yaml
name: release
on:
  release:
    types: [published]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv build
      - uses: actions/upload-artifact@v4
        with: { name: dist, path: dist/ }
  testpypi:
    needs: build
    runs-on: ubuntu-latest
    environment: testpypi
    permissions: { id-token: write }
    steps:
      - uses: actions/download-artifact@v4
        with: { name: dist, path: dist/ }
      - uses: pypa/gh-action-pypi-publish@release/v1
        with: { repository-url: "https://test.pypi.org/legacy/" }
  smoke:
    needs: testpypi
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - name: Install from TestPyPI and run the quick start
        run: |
          VERSION="${GITHUB_REF_NAME#v}"
          for i in 1 2 3 4 5 6; do
            pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ \
              "quantum-adapter==$VERSION" && break
            sleep 30
          done
          cd /tmp && python "$GITHUB_WORKSPACE/examples/quickstart.py" --tiny
  pypi:
    needs: smoke
    runs-on: ubuntu-latest
    environment: pypi
    permissions: { id-token: write }
    steps:
      - uses: actions/download-artifact@v4
        with: { name: dist, path: dist/ }
      - uses: pypa/gh-action-pypi-publish@release/v1
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci: release to TestPyPI, smoke test, then PyPI"
```

- [ ] **Step 3: Hand over to the owner (do not do these without their go-ahead)**

These steps publish to outside services. Ask the owner, then guide them:
1. Create the **public** GitHub repo `aishwaryawambule/quantum-adapter` and push (`git -c credential.helper='!gh auth git-credential' push -u origin main`).
2. On pypi.org and test.pypi.org: add a **pending trusted publisher** for project `quantum-adapter`, owner `aishwaryawambule`, repo `quantum-adapter`, workflow `release.yml`, environments `pypi` / `testpypi`.
3. In the GitHub repo settings, create environments `testpypi` and `pypi` (optionally require approval for `pypi`).
4. Confirm CI is green on GitHub, then create release `v0.1.0`.
```
