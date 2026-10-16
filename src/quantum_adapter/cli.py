"""`quantum-adapter train` and `quantum-adapter predict`: fine-tune a text classifier on your own CSV file."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn

from quantum_adapter.config import QuantumAdapterConfig
from quantum_adapter.layers import SUPPORTED_LAYERS
from quantum_adapter.model import QuantumModel, _find_modules_to_save, get_quantum_model

LABELS_NAME = "labels.json"
# Attention layers to adapt, per model family, tried in order: GPT-2, Llama/Mistral/Qwen, BERT/RoBERTa,
# DistilBERT, T5.
TARGET_CANDIDATES = (["c_attn"], ["q_proj", "v_proj"], ["query", "value"], ["q_lin", "v_lin"], ["q", "v"])
# Classification heads, trained in full and saved with the adapter: GPT-2/Llama, DistilBERT (two layers, both
# freshly initialised), BERT/RoBERTa, T5/BART.
HEAD_CANDIDATES = (["score"], ["pre_classifier", "classifier"], ["classifier"], ["classification_head"])


def read_csv(path: str | Path, text_column: str = "text", label_column: str = "label") -> tuple[list[str], list[str]]:
    """Return the texts and labels (as strings) of a CSV file with a header row; rows with an empty cell are skipped."""
    # utf-8-sig: Excel's "CSV UTF-8" export starts with a byte-order mark
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in (text_column, label_column) if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} has no column {missing[0]!r}; its columns are {reader.fieldnames}. "
                             "Pick others with --text-column / --label-column.")
        rows = [((r[text_column] or "").strip(), (r[label_column] or "").strip()) for r in reader]  # short rows: None
    rows = [(t, y) for t, y in rows if t and y]
    if not rows:
        raise ValueError(f"{path} has no rows with both a text and a label.")
    labels = sorted({y for _, y in rows})
    if len(labels) < 2:
        raise ValueError(f"{path} has only one label ({labels[0]!r}); a classifier needs at least two.")
    return [t for t, _ in rows], [y for _, y in rows]


def auto_targets(model: nn.Module) -> list[str]:
    """The attention layers to adapt for a known model family; raises if none of the known names is present."""
    present = {n.rsplit(".", 1)[-1] for n, m in model.named_modules() if isinstance(m, SUPPORTED_LAYERS)}
    for names in TARGET_CANDIDATES:
        if all(n in present for n in names):
            return names
    raise ValueError(f"Couldn't tell which layers to adapt in {type(model).__name__}; pass --target-modules "
                     f"(adaptable layer names in this model: {sorted(present)}).")


def auto_head(model: nn.Module) -> list[str]:
    """The classification head's module names, which are trained in full and saved with the adapter."""
    children = dict(model.named_children())
    for names in HEAD_CANDIDATES:
        if all(n in children for n in names):
            return names
    raise ValueError(f"Couldn't find the classification head of {type(model).__name__}; pass --head "
                     f"(its top-level modules: {sorted(children)}).")


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def _batches(texts: list[str], ys: list[int], size: int, shuffle: bool, rng: random.Random):
    order = list(range(len(texts)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), size):
        idx = order[i:i + size]
        yield [texts[j] for j in idx], torch.tensor([ys[j] for j in idx])


def _encode(tokenizer, texts: list[str], max_length: int, device: str) -> dict:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def accuracy(model: nn.Module, tokenizer, texts: list[str], ys: list[int], *, batch_size: int = 32,
             max_length: int = 128, device: str = "cpu") -> float:
    model.eval()
    correct = 0
    for bt, by in _batches(texts, ys, batch_size, False, random.Random(0)):
        logits = model(**_encode(tokenizer, bt, max_length, device)).logits
        correct += (logits.argmax(-1).cpu() == by).sum().item()
    return correct / len(texts)


def fit(model: QuantumModel, tokenizer, texts: list[str], ys: list[int], *, val_texts: list[str] | None = None,
        val_ys: list[int] | None = None, epochs: int = 6, lr: float = 1e-2, head_lr: float = 1e-3, batch_size: int = 32,
        max_length: int = 128, seed: int = 0, device: str = "cpu", log=print) -> float | None:
    """Train the adapter (at `lr`) and the modules_to_save head (at `head_lr`) with AdamW and no weight decay;
    returns the best validation accuracy (None without a val set).

    Circuit angles need a much higher learning rate than a freshly initialised head, and weight decay would pull
    the angles towards zero, so both follow the benchmark's settings.

    With a validation set, the model ends with the weights of its best epoch, since validation accuracy can
    swing by several points between epochs; without one it keeps the last epoch's weights.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model.to(device)
    head_ids = {id(p) for _, m in _find_modules_to_save(model.base_model, model.quantum_config.modules_to_save)
                for p in m.parameters()}
    trainable = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for p in trainable if id(p) not in head_ids], "lr": lr},
              {"params": [p for p in trainable if id(p) in head_ids], "lr": head_lr}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=0.0)
    best_acc, best_epoch, best_state = None, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total, steps = 0.0, 0
        for bt, by in _batches(texts, ys, batch_size, True, rng):
            loss = model(**_encode(tokenizer, bt, max_length, device), labels=by.to(device)).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, steps = total + loss.item(), steps + 1
        line = f"epoch {epoch}/{epochs}: train loss {total / steps:.4f}"
        if val_texts:
            val_acc = accuracy(model, tokenizer, val_texts, val_ys, max_length=max_length, device=device)
            line += f"  validation accuracy {val_acc:.1%}"
            if best_acc is None or val_acc > best_acc:
                best_acc, best_epoch = val_acc, epoch
                best_state = {k: v.detach().clone() for k, v in model.trainable_state_dict().items()}
        log(line)
    if best_state is not None and best_epoch != epochs:
        model._load_trainable_state(best_state)
        log(f"kept epoch {best_epoch} (best validation accuracy {best_acc:.1%})")
    return best_acc


def _load_base(model_name: str, labels: list[str]):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:  # GPT-2 and Llama-style tokenizers have none; batches need one
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), id2label=dict(enumerate(labels)),
        label2id={y: i for i, y in enumerate(labels)})
    model.config.pad_token_id = tokenizer.pad_token_id  # so the classifier finds each row's last real token
    return model, tokenizer


def split(ys: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Row indices (validation, training), holding out the same share of every label; each label keeps at
    least one training row."""
    rng = random.Random(seed)
    val, tr = [], []
    for label in sorted(set(ys)):
        rows = [i for i, y in enumerate(ys) if y == label]
        rng.shuffle(rows)
        n_val = min(int(len(rows) * val_fraction), len(rows) - 1)
        val += rows[:n_val]
        tr += rows[n_val:]
    rng.shuffle(tr)
    return sorted(val), tr


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)  # before the head and the circuit angles are initialised, so runs repeat
    texts, raw = read_csv(args.csv, args.text_column, args.label_column)
    labels = sorted(set(raw))
    ys = [labels.index(y) for y in raw]
    val, tr = split(ys, args.val_fraction, args.seed)
    print(f"{len(texts)} rows, {len(labels)} labels {labels}: training on {len(tr)}, validating on {len(val)}.")

    # A local model directory is stored as an absolute path, so predict works from any directory.
    base_model = str(Path(args.model).resolve()) if Path(args.model).is_dir() else args.model
    model, tokenizer = _load_base(base_model, labels)
    config = QuantumAdapterConfig(
        target_modules=args.target_modules or auto_targets(model), method=args.method, rank=args.rank,
        modules_to_save=args.head or auto_head(model), experimental=args.experimental,
        base_model_name_or_path=base_model)
    qm = get_quantum_model(model, config)
    qm.print_trainable_parameters()
    fit(qm, tokenizer, [texts[i] for i in tr], [ys[i] for i in tr],
        val_texts=[texts[i] for i in val] or None, val_ys=[ys[i] for i in val] or None, epochs=args.epochs,
        lr=args.lr, head_lr=args.head_lr, batch_size=args.batch_size, max_length=args.max_length, seed=args.seed,
        device=args.device or default_device())
    qm.save_pretrained(args.out)
    (Path(args.out) / LABELS_NAME).write_text(json.dumps(
        {"base_model": base_model, "labels": labels, "max_length": args.max_length}, indent=2))
    print(f"Saved the adapter to {args.out}/. Try it: quantum-adapter predict --adapter {args.out} \"some text\"")


def load_classifier(adapter: str | Path):
    """Load an adapter saved by `quantum-adapter train` onto its base model; returns (model, tokenizer, info)."""
    info = json.loads((Path(adapter) / LABELS_NAME).read_text())
    model, tokenizer = _load_base(info["base_model"], info["labels"])
    return QuantumModel.from_pretrained(model, adapter), tokenizer, info


@torch.no_grad()
def predict_texts(model: nn.Module, tokenizer, texts: list[str], labels: list[str], *, max_length: int = 128,
                  batch_size: int = 32, device: str = "cpu") -> list[tuple[str, float]]:
    """The predicted label and its probability for each text."""
    model.to(device).eval()
    out = []
    for i in range(0, len(texts), batch_size):
        probs = model(**_encode(tokenizer, texts[i:i + batch_size], max_length, device)).logits.softmax(-1).cpu()
        out += [(labels[int(p.argmax())], float(p.max())) for p in probs]
    return out


def predict(args: argparse.Namespace) -> None:
    if not args.texts and sys.stdin.isatty():
        sys.exit('Give texts to label, e.g. quantum-adapter predict --adapter DIR "some text", or pipe one per line.')
    texts = args.texts or [line.strip() for line in sys.stdin if line.strip()]
    if not texts:
        sys.exit("No texts to label.")
    model, tokenizer, info = load_classifier(args.adapter)
    for text, (label, p) in zip(texts, predict_texts(model, tokenizer, texts, info["labels"],
                                                     max_length=info["max_length"],
                                                     device=args.device or default_device())):
        print(f"{label}\t{p:.0%}\t{text}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="quantum-adapter", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="fine-tune a text classifier on a CSV file")
    t.add_argument("--csv", required=True, help="CSV file with a header row, e.g. columns text,label")
    t.add_argument("--out", required=True, help="directory to save the adapter in")
    t.add_argument("--model", default="gpt2", help="Hugging Face model id (default: gpt2)")
    t.add_argument("--text-column", default="text")
    t.add_argument("--label-column", default="label")
    t.add_argument("--method", default="amplitude", choices=["amplitude", "unitary", "tensor"])
    t.add_argument("--rank", type=int, default=4)
    t.add_argument("--experimental", action="store_true", help="allow the experimental unitary and tensor methods")
    t.add_argument("--target-modules", nargs="+", help="layers to adapt (default: picked for the model family)")
    t.add_argument("--head", nargs="+", help="classification head modules (default: picked for the model family)")
    # Quantum adapters learn more slowly than LoRA: about twice LoRA's epochs at a learning rate near 1e-2.
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--lr", type=float, default=1e-2, help="learning rate of the circuit angles")
    t.add_argument("--head-lr", type=float, default=1e-3, help="learning rate of the classification head")
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--max-length", type=int, default=128, help="tokens kept per text")
    t.add_argument("--val-fraction", type=float, default=0.1, help="share of rows held out for validation")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", help="cuda, mps or cpu (default: the fastest available)")
    t.set_defaults(func=train)

    p = sub.add_parser("predict", help="label texts with an adapter saved by train")
    p.add_argument("--adapter", required=True, help="directory written by train")
    p.add_argument("texts", nargs="*", help="texts to label (default: one per line from stdin)")
    p.add_argument("--device")
    p.set_defaults(func=predict)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train" and not 0 <= args.val_fraction < 1:
        build_parser().error("--val-fraction must be at least 0 and below 1")
    args.func(args)


if __name__ == "__main__":
    main()
