"""Release benchmark: one task, many contenders, LR picked on a dev split, then 3 seeds.

Usage:
  uv run --group bench python benchmarks/run.py --task sst2 --out results/bench-sst2.jsonl

Tasks: sst2 and ag_news (classification, metric "accuracy") and e2e (text generation, metric "bleu").
Every row stores its score under `dev_acc` / `eval_acc` as a fraction, so the gate's units are the same for all
tasks: accuracy for classification, corpus BLEU / 100 for e2e (0.25 means BLEU 25.0). Rows carry `metric` too.
For e2e the contender `head` is the frozen pretrained model with no training: the baseline the gate looks up
under that name for every task (classification `head` trains only the classification head).
(The script's own folder is on sys.path, so `controls` imports from any working directory.)
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import textgen
import torch
from controls import RandomProjectionMethod
from datasets import load_dataset
from transformers import GPT2ForSequenceClassification, GPT2LMHeadModel, GPT2TokenizerFast

from quantum_adapter import QuantumAdapterConfig, __version__, get_quantum_model
from quantum_adapter.layers import AdapterLayer
from quantum_adapter.methods import METHOD_CLASSES
from quantum_adapter.model import inject_adapters

DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
TASKS = {
    "sst2": {"path": "nyu-mll/glue", "name": "sst2", "text": "sentence", "labels": 2, "eval_split": "validation",
             "maxlen": 64, "eval_limit": None},
    "ag_news": {"path": "fancyzhx/ag_news", "name": None, "text": "text", "labels": 4, "eval_split": "test",
                "maxlen": 128, "eval_limit": 2000},
    # The repo's loading script is unsupported by the installed `datasets`, so use its parquet conversion.
    "e2e": {"path": "GEM/e2e_nlg", "revision": "refs/convert/parquet", "kind": "generation", "metric": "bleu",
            "maxlen": 128, "eval_limit": 500},
}
LRS = {
    "head": [None], "lora": [3e-4, 1e-3, 3e-3], "tensor": [3e-4, 1e-3, 3e-3], "control:tensor": [3e-4, 1e-3, 3e-3],
    "amplitude": [3e-3, 1e-2, 3e-2], "unitary": [3e-3, 1e-2, 3e-2],
    "control:amplitude": [3e-3, 1e-2, 3e-2], "control:unitary": [3e-3, 1e-2, 3e-2],
}
# 6 epochs for every contender and task, so all train equally long. Gap experiment (SST-2, n=2000, 3 seeds, GPT-2 small,
# c_attn): at 3 epochs LoRA r4 reached 87.1% and amplitude 84.8%; at 6 epochs amplitude reached 87.3% and LoRA 87.7%.
# Quantum adapters learn more slowly, so the short schedule understated them. See docs/experiments.md.
HEAD_LR, EPOCHS, BS, N_DEV, RANK = 1e-3, 6, 32, 500, 4
MERGE_TOL = 0.002  # classification: max dev accuracy change caused by merging
# generation: max abs change of the first-token logits (fp32) caused by merging. Measured noise floor: real GPT-2 LM head,
# amplitude adapter on c_attn with parameters scaled 0.05-0.5, 16 real token batches, fp32 on MPS: max abs diff 6.7e-4 over
# all positions (logits reach ~230; the classification head differs by only 6e-6). Tolerance = max(1e-3, 10x measured).
MERGE_LOGIT_TOL = 7e-3
MERGE_PROMPTS = 16
DIRTY_PATHS = ("src", "benchmarks", "pyproject.toml", "uv.lock")
LIBRARY = ("amplitude", "unitary", "tensor")


def git_sha() -> str | None:
    """The checked-out commit, with "-dirty" if the library, benchmark code or dependency pins are modified."""
    root = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *DIRTY_PATHS], cwd=root,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha + ("-dirty" if dirty else "")


def provenance(n_train: int) -> dict:
    """Recorded on every row, so gate.py can refuse results from mixed runs."""
    return {"n_train": n_train, "git_sha": git_sha(), "library_version": __version__, "device": DEV}


def is_generation(task: dict) -> bool:
    return task.get("kind") == "generation"


def metric_name(task: dict) -> str:
    return task.get("metric", "accuracy")


def load_generation(task: dict, n_train: int, tok):
    """Train/dev rows are (tokens, mask, labels) / (meaning representations, [target]); eval is (mrs, references)."""
    tok.padding_side = "left"  # generation pads on the left; training batches are padded on the right by hand
    ds = load_dataset(task["path"], revision=task["revision"])
    pool = ds["train"].shuffle(seed=0)
    train, dev = pool.select(range(n_train)), pool.select(range(n_train, n_train + N_DEV))
    mrs, refs = textgen.group_references(ds["test"], limit=task["eval_limit"])
    return (textgen.encode_dataset(tok, list(train["meaning_representation"]), list(train["target"]), task["maxlen"]),
            (list(dev["meaning_representation"]), [[t] for t in dev["target"]]), (mrs, refs), tok)


def load(task: dict, n_train: int):
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    if is_generation(task):
        return load_generation(task, n_train, tok)
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
    if is_generation(task):
        return GPT2LMHeadModel.from_pretrained("gpt2", attn_pdrop=0.0)
    m = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=task["labels"], attn_pdrop=0.0)
    m.config.pad_token_id = tok.eos_token_id
    return m


def qcfg(method: str, generation: bool = False) -> QuantumAdapterConfig:
    return QuantumAdapterConfig(target_modules=["c_attn"], method=method, rank=RANK,
                                modules_to_save=[] if generation else ["score"], experimental=True)


def tensor_control_rank() -> int:
    """LoRA at the rank whose parameter count is nearest the tensor method's."""
    per_layer = METHOD_CLASSES["tensor"](768, 2304, qcfg("tensor")).summary()["params"]
    return max(1, round(per_layer / (768 + 2304)))


def build(contender: str, task: dict, tok):
    gen = is_generation(task)
    m = base_model(task, tok)
    if contender == "head":
        m.requires_grad_(False)
        if not gen:
            m.score.requires_grad_(True)
        return m
    if contender == "lora" or contender == "control:tensor":
        from peft import LoraConfig, get_peft_model
        rank = RANK
        if contender == "control:tensor":
            rank = tensor_control_rank()
        kw = {"task_type": "CAUSAL_LM"} if gen else {"modules_to_save": ["score"]}
        pm = get_peft_model(m, LoraConfig(r=rank, target_modules=["c_attn"], fan_in_fan_out=True, lora_dropout=0.0,
                                          **kw))
        return pm
    if contender.startswith("control:"):
        method = contender.split(":", 1)[1]
        cfg = qcfg(method, gen)
        n_params = METHOD_CLASSES[method](768, 2304, cfg).summary()["params"]
        m.requires_grad_(False)
        inject_adapters(m, ["c_attn"], lambda i, o: RandomProjectionMethod(i, o, cfg, n_params))
        if not gen:
            m.score.requires_grad_(True)
        return m
    return get_quantum_model(m, qcfg(contender, gen))


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


@torch.no_grad()
def bleu_score(model, data, tok) -> tuple[float, float]:
    """Corpus BLEU / 100 of greedy generations for (meaning representations, references), and seconds taken."""
    mrs, refs = data
    t0 = time.time()
    hyps = textgen.generate(model, tok, mrs, DEV)
    if DEV == "mps":
        torch.mps.synchronize()
    return textgen.bleu(hyps, refs), time.time() - t0


def score(model, data, task: dict, tok) -> tuple[float, float]:
    """(score, seconds): accuracy for classification, BLEU / 100 for generation."""
    return bleu_score(model, data, tok) if is_generation(task) else accuracy(model, data)


@torch.no_grad()
def timing_pass(model, data, task: dict, tok) -> float:
    """Seconds for one pass over the eval data. Generation decodes a fixed number of tokens, so every model
    does identical work: a trained model that stops at EOS would otherwise look faster than the base."""
    if is_generation(task):
        t0 = time.time()
        textgen.generate(model, tok, data[0], DEV, fixed_length=True)
        if DEV == "mps":
            torch.mps.synchronize()
        return time.time() - t0
    return accuracy(model, data)[1]


@torch.no_grad()
def timed_inference(model, data, task: dict, tok, passes: int = 3) -> float:
    """Median seconds of `passes` evaluation passes after one warm-up pass."""
    timing_pass(model, data, task, tok)
    return statistics.median(timing_pass(model, data, task, tok) for _ in range(passes))


_BASE = {}


def cached_base(task: dict, tok):
    """The plain base model for timing, loaded once per task per process."""
    key = task["path"]
    if key not in _BASE:
        _BASE[key] = base_model(task, tok).to(DEV).eval()
    return _BASE[key]


def adapter_param_count(model) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "score" not in n)


def run(contender: str, lr: float | None, seed: int, task_name: str, data, run_id: str) -> dict:
    task = TASKS[task_name]
    train, dev, ev, tok = data
    torch.manual_seed(seed)
    model = build(contender, task, tok).to(DEV)
    head = [p for n, p in model.named_parameters() if p.requires_grad and "score" in n]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and "score" not in n]
    groups = ([{"params": head, "lr": HEAD_LR}] if head else []) + ([{"params": rest, "lr": lr}] if rest else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.0) if groups else None  # none: the untrained generation baseline
    ids, mask, y = train  # y: class labels, or token labels with -100 on prompt and padding (loss on target only)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for _ in range(EPOCHS if opt else 0):
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
    dev_acc, _ = score(model, dev, task, tok)
    merged_ok, dev_merged = None, None
    if hasattr(model, "merge_and_unload"):
        gen = is_generation(task)
        if gen:
            probe = dev[0][:MERGE_PROMPTS]
            before = textgen.next_token_logits(model, tok, probe, DEV)
        merged = model.merge_and_unload()
        dev_merged, _ = score(merged, dev, task, tok)
        if gen:  # greedy decoding can flip a token on a tiny numerical change, so compare logits, not BLEU
            changed = (textgen.next_token_logits(merged, tok, probe, DEV) - before).abs().max().item() > MERGE_LOGIT_TOL
        else:
            changed = abs(dev_merged - dev_acc) > MERGE_TOL
        leftover = any(isinstance(m, AdapterLayer) for m in merged.modules())
        ok = not changed and not leftover
        if not ok:
            print(f"WARNING {contender}: merge check failed (dev {metric_name(task)} {dev_acc:.4f} -> "
                  f"{dev_merged:.4f}, adapter layers left: {leftover})", flush=True)
        if contender in LIBRARY:
            merged_ok = ok
        elif not ok:
            merged_ok = False
    else:
        merged = model  # random-projection controls have no merge; their inference time is for the unmerged model
    eval_acc, _ = score(merged, ev, task, tok)
    # Time the merged model and the cached base model back to back so machine drift cancels.
    infer_s = timed_inference(merged, ev, task, tok)
    base_infer_s = timed_inference(cached_base(task, tok), ev, task, tok)
    return {"run_id": run_id, "task": task_name, "metric": metric_name(task), "contender": contender, "lr": lr,
            "seed": seed,
            "trainable": trainable, "adapter_params": n_adapter, "dev_acc": dev_acc, "eval_acc": eval_acc,
            "train_s": round(train_s, 1), "infer_s": round(infer_s, 3), "base_infer_s": round(base_infer_s, 3),
            "merged_ok": merged_ok, "dev_before_merge": dev_acc if dev_merged is not None else None,
            "dev_after_merge": dev_merged}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--contenders", nargs="+", default=list(LRS))
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true", help="add to an existing non-empty --out file")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and out.stat().st_size > 0 and not a.append:
        raise SystemExit(f"{a.out} already has results; pass --append to add to it, or choose a new --out")
    run_id = f"{a.task}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    data = load(TASKS[a.task], a.n_train)
    meta = provenance(a.n_train)
    with open(a.out, "a") as f:
        for c in a.contenders:
            if c == "control:tensor" and tensor_control_rank() == RANK:
                print("skipping control:tensor: its matched LoRA rank equals LoRA's, so it would duplicate lora",
                      flush=True)
                continue
            best = None
            for lr in LRS[c]:
                r = {**run(c, lr, 0, a.task, data, run_id), **meta}
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(json.dumps(r), flush=True)
                if best is None or r["dev_acc"] > best["dev_acc"]:
                    best = r
            for seed in (1, 2):
                r = {**run(c, best["lr"], seed, a.task, data, run_id), **meta}
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
