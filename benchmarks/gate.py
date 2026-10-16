"""Apply the production-usefulness rule (spec section 1) to benchmark results.

Usage: uv run python benchmarks/gate.py results/bench-sst2.jsonl results/bench-ag_news.jsonl results/bench-e2e.jsonl --markdown docs/benchmarks.md

The gate refuses mixed, partial or duplicated results: every (task, contender, lr, seed) must appear once, the
selected LR must have exactly seeds 0, 1, 2, all required tasks must be present, and every row (across all tasks) must
agree on n_train, git_sha, library_version and device, with no "-dirty" git_sha (rows written before run.py
recorded them have none).

Metrics: sst2 and ag_news score accuracy; e2e scores corpus BLEU. Every row stores its score under `dev_acc` /
`eval_acc` as a fraction (accuracy, or BLEU / 100), so the 1.0-point rule is 0.01 for both and is applied per task
in that task's own metric (`metric` on each row; rows without it are accuracy). The e2e `head` contender is the
frozen pretrained model with no training, serving as the baseline the lookups below expect.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

METHODS = ["amplitude", "unitary", "tensor"]
REQUIRED_TASKS = {"sst2", "ag_news", "e2e"}
SEEDS = {0, 1, 2}
RANK = 4  # LoRA rank used by run.py
# Controls built from RandomProjectionMethod are never merged; control:tensor is PEFT LoRA, which is.
UNMERGED = {"control:amplitude", "control:unitary"}
EPS = 1e-9
# Recorded by run.py on every row since provenance was added; older rows have none of them.
PROVENANCE = ("n_train", "git_sha", "library_version", "device")
METRIC_LABELS = {"accuracy": "Accuracy", "bleu": "BLEU"}


def metric_of(rows: list[dict]) -> str:
    return rows[0].get("metric", "accuracy")


class GateError(ValueError):
    """The results cannot be judged (mixed, partial or duplicated runs)."""


def tensor_control_rank() -> int:
    """The LoRA rank run.py matches to the tensor method (same computation as run.py's tensor_control_rank)."""
    from quantum_adapter import QuantumAdapterConfig
    from quantum_adapter.methods import METHOD_CLASSES

    cfg = QuantumAdapterConfig(target_modules=["c_attn"], method="tensor", rank=RANK, modules_to_save=["score"],
                               experimental=True)
    per_layer = METHOD_CLASSES["tensor"](768, 2304, cfg).summary()["params"]
    return max(1, round(per_layer / (768 + 2304)))


def check_rows(rows: list[dict]) -> None:
    seen = set()
    for r in rows:
        if "base_infer_s" not in r:
            raise GateError("results predate the inference-timing fix (no base_infer_s); rerun the benchmark")
        key = (r["task"], r["contender"], r["lr"], r["seed"])
        if key in seen:
            raise GateError(f"duplicate run {key}: results from several runs were mixed in one file")
        seen.add(key)
    if any(k in r for r in rows for k in PROVENANCE):  # legacy rows with none at all still pass
        for key in PROVENANCE:  # provenance must agree across ALL tasks
            values = {r.get(key) for r in rows}
            if len(values) > 1:
                raise GateError(f"rows disagree on {key} ({sorted(map(str, values))}): results from "
                                "different runs, data sizes, code versions or devices were mixed")
    dirty = sorted({str(r["git_sha"]) for r in rows if str(r.get("git_sha") or "").endswith("-dirty")})
    if dirty:
        raise GateError(f"results come from uncommitted code (git_sha {dirty}); commit and rerun the benchmark")
    for task in sorted({r["task"] for r in rows}):
        metrics = {r.get("metric", "accuracy") for r in rows if r["task"] == task}
        if len(metrics) > 1:
            raise GateError(f"{task}: rows disagree on metric ({sorted(metrics)})")
    missing = REQUIRED_TASKS - {r["task"] for r in rows}
    if missing:
        raise GateError(f"missing required task(s): {sorted(missing)}")


def best_runs(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Per (task, contender): the 3 runs at the LR that won on the dev split for seed 0."""
    check_rows(rows)
    by = defaultdict(list)
    for r in rows:
        by[(r["task"], r["contender"])].append(r)
    out = {}
    for key, rs in by.items():
        seed0 = [r for r in rs if r["seed"] == 0]
        if not seed0:
            raise GateError(f"{key}: no seed 0 run to select the learning rate from")
        lr = max(seed0, key=lambda r: r["dev_acc"])["lr"]
        chosen = [r for r in rs if r["lr"] == lr]
        if {r["seed"] for r in chosen} != SEEDS:
            raise GateError(f"{key}: LR {lr} has seeds {sorted(r['seed'] for r in chosen)}, expected {sorted(SEEDS)}")
        out[key] = chosen
    return out


def mean(rs: list[dict], k: str) -> float:
    return st.mean(r[k] for r in rs)


def control_runs(runs: dict, task: str, method: str) -> list[dict]:
    """The same-size control; control:tensor is absent when its matched LoRA rank equals LoRA's, so use LoRA."""
    key = (task, f"control:{method}")
    if key in runs:
        return runs[key]
    if method == "tensor" and tensor_control_rank() == RANK:
        return runs[(task, "lora")]
    raise GateError(f"missing contender control:{method} for task {task}")


def verdicts(runs: dict) -> dict[str, dict]:
    tasks = sorted({t for t, _ in runs})
    for t in tasks:
        for c in ["head", "lora", *METHODS]:
            if (t, c) not in runs:
                raise GateError(f"missing contender {c} for task {t}")
    result = {}
    for m in METHODS:
        checks = defaultdict(list)
        for t in tasks:
            q, lora = runs[(t, m)], runs[(t, "lora")]
            ctrl = control_runs(runs, t, m)
            checks["accuracy within 1.0 of LoRA"].append(mean(q, "eval_acc") >= mean(lora, "eval_acc") - 0.01 - EPS)
            checks["≥5x fewer params than LoRA"].append(lora[0]["adapter_params"] >= 5 * q[0]["adapter_params"])
            checks["train time ≤2x LoRA"].append(mean(q, "train_s") <= 2 * mean(lora, "train_s") + EPS)
            checks["merged inference within 10% of base"].append(
                mean(q, "infer_s") <= 1.1 * mean(q, "base_infer_s") + EPS)
            checks["merges cleanly"].append(all(r.get("merged_ok") is True for r in q))
            checks["beats same-size classical control"].append(mean(q, "eval_acc") > mean(ctrl, "eval_acc"))
        passed = {name: all(v) for name, v in checks.items()}
        acc = st.mean(mean(runs[(t, m)], "eval_acc") for t in tasks)
        result[m] = {"checks": passed, "stable": all(passed.values()), "mean_acc": acc}
    return result


def default_method(v: dict) -> str | None:
    stable = [m for m in METHODS if v[m]["stable"]]
    return max(stable, key=lambda m: v[m]["mean_acc"]) if stable else None


def table(runs: dict) -> str:
    lines = []
    for t in sorted({t for t, _ in runs}):
        label = METRIC_LABELS.get(metric_of(next(rs for (task, _), rs in runs.items() if task == t)), "Score")
        lines += [f"\n### {t}\n", f"| Contender | Adapter params | {label} (mean, min–max) | Train s | Inference s |",
                  "|---|---|---|---|---|"]
        for (task, c), rs in sorted(runs.items()):
            if task != t:
                continue
            v = [100 * r["eval_acc"] for r in rs]
            infer = f"{mean(rs, 'infer_s'):.1f}" + (" (unmerged)" if c in UNMERGED else "")
            unit = "" if label == "BLEU" else "%"  # BLEU is reported in points, accuracy as a percentage
            lines.append(f"| {c} | {rs[0]['adapter_params']:,} | {st.mean(v):.1f}{unit} ({min(v):.1f}–{max(v):.1f}) | "
                         f"{mean(rs, 'train_s'):.0f} | {infer} |")
    return "\n".join(lines)


def report(runs: dict) -> str:
    v = verdicts(runs)
    default = default_method(v)
    lines = ["# Benchmark results", "", "GPT-2 small, adapters on `c_attn`, 3 seeds at the dev-selected LR.",
             table(runs), ""]
    if any((t, "control:tensor") not in runs for t in sorted({t for t, _ in runs})):
        lines += [("Note: `control:tensor` was not run where its matched LoRA rank equals LoRA's rank; "
                   "the LoRA rows serve as the tensor control."), ""]
    lines += ["## Verdicts", ""]
    for m in METHODS:
        lines.append(f"- **{m}**: {'stable' if v[m]['stable'] else 'experimental'} — " + ", ".join(
            f"{name} {'✅' if ok else '❌'}" for name, ok in v[m]["checks"].items()))
    lines += ["", f"Default method: **{default or 'none passed: release blocked'}**", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--markdown")
    a = ap.parse_args()
    rows = [json.loads(line) for f in a.files
            for line in Path(f).read_text().splitlines() if line.strip()]
    try:
        text = report(best_runs(rows))
    except GateError as e:
        raise SystemExit(f"gate refused: {e}") from e
    print(text)
    if a.markdown:
        Path(a.markdown).write_text(text)


if __name__ == "__main__":
    main()
