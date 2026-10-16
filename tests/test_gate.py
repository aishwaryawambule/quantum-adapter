import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import gate
from gate import GateError, best_runs, default_method, report, verdicts

TASKS = ["sst2", "ag_news", "e2e"]
ALL = ["head", "lora", "amplitude", "unitary", "tensor", "control:amplitude", "control:unitary", "control:tensor"]


def rows_for(task, contender, acc=0.80, params=100, train_s=10.0, infer_s=1.0, base=1.0, lr=1, dev=0.5, ok=True):
    return [{"task": task, "metric": "bleu" if task == "e2e" else "accuracy", "contender": contender, "lr": lr, "seed": s, "trainable": params,
             "adapter_params": params, "dev_acc": dev, "eval_acc": acc, "train_s": train_s, "infer_s": infer_s,
             "base_infer_s": base, "merged_ok": ok} for s in (0, 1, 2)]


def build(over=None, drop=()):
    """Passing baseline: LoRA 0.80 with 1000 params; others 0.80, 200 params; controls 0.70."""
    over = over or {}
    rows = []
    for t in TASKS:
        for c in ALL:
            if c in drop:
                continue
            kw = {"acc": 0.70 if c.startswith("control:") else 0.80, "params": 1000 if c == "lora" else 200}
            if c == "head":
                kw["params"] = 0
            kw.update(over.get((t, c), {}) or over.get(c, {}))
            rows += rows_for(t, c, **kw)
    return rows


def stable(rows, m="amplitude"):
    return verdicts(best_runs(rows))[m]["checks"]


def test_all_pass_baseline():
    v = verdicts(best_runs(build()))
    assert all(v[m]["stable"] for m in gate.METHODS)


def test_best_runs_picks_lr_on_dev_seed0():
    rows = []
    for lr, dev in [(1, 0.5), (2, 0.9), (3, 0.6)]:
        rows.append({**rows_for("sst2", "lora", lr=lr, dev=dev)[0]})
    for lr in (2,):
        rows += rows_for("sst2", "lora", lr=lr, dev=0.9)[1:]
    rows += [r for r in build() if r["task"] != "sst2"]
    out = best_runs(rows)
    assert {r["lr"] for r in out[("sst2", "lora")]} == {2}
    assert len(out[("sst2", "lora")]) == 3


def test_duplicates_raise():
    rows = build()
    with pytest.raises(GateError, match="duplicate"):
        best_runs(rows + rows[:1])


def test_missing_seed_raises():
    rows = [r for r in build() if not (r["task"] == "sst2" and r["contender"] == "lora" and r["seed"] == 2)]
    with pytest.raises(GateError, match="seeds"):
        best_runs(rows)


def test_missing_task_raises():
    with pytest.raises(GateError, match="ag_news"):
        best_runs([r for r in build() if r["task"] == "sst2"])


def test_results_without_e2e_are_refused_naming_the_task():
    with pytest.raises(GateError, match=r"missing required task\(s\): \['e2e'\]"):
        best_runs([r for r in build() if r["task"] != "e2e"])


def test_required_tasks_are_the_three_release_tasks():
    assert gate.REQUIRED_TASKS == {"sst2", "ag_news", "e2e"}


def test_table_labels_each_task_with_its_metric():
    text = report(best_runs(build()))
    sections = {s.split("\n", 1)[0].strip(): s for s in text.split("### ")[1:]}
    assert "Accuracy (mean" in sections["sst2"] and "BLEU" not in sections["sst2"]
    assert "Accuracy (mean" in sections["ag_news"]
    assert "BLEU (mean" in sections["e2e"] and "Accuracy" not in sections["e2e"]
    assert "| 80.0 (80.0–80.0) |" in sections["e2e"].replace("%", "")  # BLEU points, not a percentage
    assert "80.0% (80.0–80.0)" in sections["sst2"]


def test_one_point_rule_applies_in_bleu_units_on_e2e():
    assert stable(build({("e2e", "amplitude"): {"acc": 0.79}}))["accuracy within 1.0 of LoRA"] is True
    assert stable(build({("e2e", "amplitude"): {"acc": 0.7899}}))["accuracy within 1.0 of LoRA"] is False


def test_mixed_metrics_within_a_task_are_refused():
    rows = build()
    rows[0] = {**rows[0], "metric": "bleu"}
    with pytest.raises(GateError, match="metric"):
        best_runs(rows)


@pytest.mark.parametrize(("acc", "ok"), [(0.79, True), (0.7899, False)])
def test_accuracy_boundary(acc, ok):
    assert stable(build({"amplitude": {"acc": acc}}))["accuracy within 1.0 of LoRA"] is ok


@pytest.mark.parametrize(("params", "ok"), [(200, True), (201, False)])
def test_param_boundary(params, ok):
    assert stable(build({"amplitude": {"params": params}}))["≥5x fewer params than LoRA"] is ok


@pytest.mark.parametrize(("train_s", "ok"), [(20.0, True), (20.1, False)])
def test_train_time_boundary(train_s, ok):
    assert stable(build({"amplitude": {"train_s": train_s}}))["train time ≤2x LoRA"] is ok


@pytest.mark.parametrize(("infer", "ok"), [(1.1, True), (1.11, False)])
def test_inference_boundary(infer, ok):
    assert stable(build({"amplitude": {"infer_s": infer}}))["merged inference within 10% of base"] is ok


def test_equal_to_control_fails_and_better_passes():
    key = "beats same-size classical control"
    assert stable(build({"control:amplitude": {"acc": 0.80}}))[key] is False
    assert stable(build({"control:amplitude": {"acc": 0.79}}))[key] is True


def test_unmerged_fails():
    assert stable(build({"amplitude": {"ok": False}}))["merges cleanly"] is False


def test_control_tensor_falls_back_to_lora():
    rows = build({"tensor": {"acc": 0.81}}, drop=("control:tensor",))
    assert stable(rows, "tensor")["beats same-size classical control"] is True
    assert "serve as the tensor control" in report(best_runs(rows))
    lora_like = build({"tensor": {"acc": 0.80}}, drop=("control:tensor",))
    assert stable(lora_like, "tensor")["beats same-size classical control"] is False


def test_default_is_best_stable_and_blocked_when_none():
    rows = build({"tensor": {"acc": 0.805}})
    assert default_method(verdicts(best_runs(rows))) == "tensor"
    rows = build({"amplitude": {"acc": 0.5}, "unitary": {"acc": 0.5}, "tensor": {"acc": 0.5}})
    v = verdicts(best_runs(rows))
    assert default_method(v) is None
    assert "release blocked" in report(best_runs(rows))


def test_inference_uses_base_infer_s_not_head_rows():
    # Head rows are slow (2.0) while the method's own base timing is 1.0; the verdict must use base_infer_s.
    rows = build({"head": {"infer_s": 2.0}, "amplitude": {"infer_s": 1.5, "base": 1.5}})
    assert stable(rows)["merged inference within 10% of base"] is True
    rows = build({"head": {"infer_s": 2.0}, "amplitude": {"infer_s": 1.5, "base": 1.0}})
    assert stable(rows)["merged inference within 10% of base"] is False


def test_method_failing_one_task_is_experimental():
    rows = build({("ag_news", "amplitude"): {"acc": 0.5}})
    v = verdicts(best_runs(rows))["amplitude"]
    assert v["stable"] is False
    assert v["checks"]["accuracy within 1.0 of LoRA"] is False


def test_lr_selection_uses_seed0_only():
    rows = [r for r in build() if not (r["task"] == "sst2" and r["contender"] == "lora")]
    rows += rows_for("sst2", "lora", lr=1, dev=0.5) + rows_for("sst2", "lora", lr=2, dev=0.4)[:1]
    rows += [{**rows_for("sst2", "lora", lr=2, dev=0.99)[1]}]  # seed 1 at lr 2 has higher dev_acc; must be ignored
    rows += rows_for("sst2", "lora", lr=2, dev=0.4)[2:]
    assert {r["lr"] for r in best_runs(rows)[("sst2", "lora")]} == {1}


def test_legacy_rows_without_base_infer_s_raise_cleanly():
    rows = build()
    del rows[0]["base_infer_s"]
    with pytest.raises(GateError, match="rerun"):
        best_runs(rows)


def test_control_tensor_missing_for_other_reason_refuses(monkeypatch):
    monkeypatch.setattr(gate, "tensor_control_rank", lambda: gate.RANK + 1)
    with pytest.raises(GateError, match="control:tensor"):
        verdicts(best_runs(build(drop=("control:tensor",))))


def test_unmerged_label_only_on_random_projection_controls():
    t = gate.table(best_runs(build()))
    for line in t.splitlines():
        assert ("(unmerged)" in line) == line.startswith(("| control:amplitude", "| control:unitary"))


PROVENANCE = {"n_train": 2000, "git_sha": "abc123", "library_version": "0.1.0", "device": "mps"}


def with_provenance(rows, **over):
    return [{**r, **PROVENANCE, **over} for r in rows]


def test_rows_without_provenance_are_accepted():
    assert verdicts(best_runs(build()))  # rows written before run.py recorded provenance


def test_consistent_provenance_is_accepted():
    rows = with_provenance(build())
    assert verdicts(best_runs(rows))


@pytest.mark.parametrize("key, value", [("n_train", 4000), ("git_sha", "def456"), ("library_version", "0.2.0"),
                                        ("device", "cuda")])
def test_provenance_must_agree_across_tasks(key, value):
    rows = [r if r["task"] == "sst2" else {**r, key: value} for r in with_provenance(build())]
    with pytest.raises(GateError, match=f"rows disagree on {key}"):
        best_runs(rows)


@pytest.mark.parametrize("sha", ["abc123-dirty", "-dirty"])
def test_dirty_git_sha_is_refused(sha):
    with pytest.raises(GateError, match="uncommitted"):
        best_runs(with_provenance(build(), git_sha=sha))


def test_failed_merge_recorded_by_run_fails_the_method_not_the_run():
    rows = build({("e2e", "tensor"): {"ok": False}})
    checks = stable(rows, "tensor")
    assert checks["merges cleanly"] is False
    assert verdicts(best_runs(rows))["amplitude"]["stable"]


@pytest.mark.parametrize("key, value", [("n_train", 64), ("git_sha", "def456"), ("library_version", "0.2.0"),
                                        ("device", "cuda"), ("device", None)])
def test_inconsistent_provenance_within_a_task_is_refused(key, value):
    rows = with_provenance(build())
    i = next(i for i, r in enumerate(rows) if r["task"] == "sst2" and r["contender"] == "lora")
    rows[i] = {**rows[i], key: value}
    if value is None:
        del rows[i][key]
    with pytest.raises(GateError, match=f"rows disagree on {key}"):
        best_runs(rows)
