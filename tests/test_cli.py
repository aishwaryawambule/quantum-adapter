import json
import subprocess
import sys

import pytest
import torch
from conftest import make_tiny_gpt2
from transformers import RobertaConfig, RobertaForSequenceClassification

from quantum_adapter import cli
from quantum_adapter.config import QuantumAdapterConfig
from quantum_adapter.model import QuantumModel, get_quantum_model


class CharTokenizer:
    """Maps each character to an id in 1..63 and pads with 0, like a Hugging Face tokenizer call."""

    pad_token = "<pad>"
    pad_token_id = 0

    def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        rows = [[1 + ord(c) % 63 for c in t][:max_length] for t in texts]
        width = max(len(r) for r in rows)
        ids = torch.tensor([r + [0] * (width - len(r)) for r in rows])
        return {"input_ids": ids, "attention_mask": (ids != 0).long()}


def write_csv(path, rows, header="text,label"):
    path.write_text(header + "\n" + "\n".join(f"{t},{y}" for t, y in rows) + "\n")
    return path


# "positive" texts are all a's, "negative" all z's: separable from the first character.
ROWS = [("a" * (3 + i % 5), "positive") for i in range(24)] + [("z" * (3 + i % 5), "negative") for i in range(24)]


def test_read_csv_skips_empty_rows_and_keeps_string_labels(tmp_path):
    texts, labels = cli.read_csv(write_csv(tmp_path / "d.csv", [("good", "pos"), ("", "neg"), ("bad", "neg")]))
    assert texts == ["good", "bad"] and labels == ["pos", "neg"]


def test_read_csv_custom_columns(tmp_path):
    p = write_csv(tmp_path / "d.csv", [("x", "a"), ("y", "b")], header="review,stars")
    assert cli.read_csv(p, "review", "stars") == (["x", "y"], ["a", "b"])


@pytest.mark.parametrize("rows, header, match", [
    ([("x", "a")], "body,label", "no column 'text'"),
    ([("x", "a"), ("y", "a")], "text,label", "only one label"),
    ([("", "a")], "text,label", "no rows"),
])
def test_read_csv_errors(tmp_path, rows, header, match):
    with pytest.raises(ValueError, match=match):
        cli.read_csv(write_csv(tmp_path / "d.csv", rows, header))


def test_auto_targets_and_head_per_family(tiny_llama):
    gpt2 = make_tiny_gpt2()
    roberta = RobertaForSequenceClassification(RobertaConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=40, num_labels=2))
    assert cli.auto_targets(gpt2) == ["c_attn"] and cli.auto_head(gpt2) == "score"
    assert cli.auto_targets(tiny_llama) == ["q_proj", "v_proj"]
    assert cli.auto_targets(roberta) == ["query", "value"] and cli.auto_head(roberta) == "classifier"


def test_auto_targets_and_head_explain_failure():
    plain = torch.nn.Sequential(torch.nn.Linear(4, 4))
    with pytest.raises(ValueError, match="--target-modules"):
        cli.auto_targets(plain)
    with pytest.raises(ValueError, match="--head"):
        cli.auto_head(plain)


def test_fit_learns_a_separable_task():
    qm = get_quantum_model(make_tiny_gpt2(), QuantumAdapterConfig(
        target_modules=["c_attn"], rank=2, depth=2, modules_to_save=["score"]))
    texts, labels = [t for t, _ in ROWS], [int(y == "positive") for _, y in ROWS]
    tok = CharTokenizer()
    logs = []
    acc = cli.fit(qm, tok, texts, labels, val_texts=texts, val_ys=labels, epochs=8, lr=3e-2, batch_size=8,
                  log=logs.append)
    assert len(logs) == 8 and "validation accuracy" in logs[-1]
    assert acc == 1.0


def test_train_then_predict_round_trip(tmp_path, monkeypatch, capsys):
    labels_seen = []

    def fake_load_base(model_name, labels):
        labels_seen.append((model_name, labels))
        return make_tiny_gpt2(), CharTokenizer()

    monkeypatch.setattr(cli, "_load_base", fake_load_base)
    csv_path = write_csv(tmp_path / "d.csv", ROWS)
    out = tmp_path / "adapter"
    cli.main(["train", "--csv", str(csv_path), "--out", str(out), "--model", "tiny", "--epochs", "8",
              "--lr", "3e-2", "--batch-size", "8", "--rank", "2", "--device", "cpu"])
    assert labels_seen[0] == ("tiny", ["negative", "positive"])
    info = json.loads((out / cli.LABELS_NAME).read_text())
    assert info == {"base_model": "tiny", "labels": ["negative", "positive"], "max_length": 128}
    saved = json.loads((out / "adapter_config.json").read_text())
    assert saved["target_modules"] == ["c_attn"] and saved["modules_to_save"] == ["score"]
    assert "Saved the adapter" in capsys.readouterr().out

    cli.main(["predict", "--adapter", str(out), "--device", "cpu", "aaaa", "zzzzz"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert [ln.split("\t")[0] for ln in lines] == ["positive", "negative"]
    assert [ln.split("\t")[2] for ln in lines] == ["aaaa", "zzzzz"]


def test_load_classifier_returns_a_frozen_model(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_load_base", lambda name, labels: (make_tiny_gpt2(), CharTokenizer()))
    qm = get_quantum_model(make_tiny_gpt2(), QuantumAdapterConfig(
        target_modules=["c_attn"], rank=2, depth=2, modules_to_save=["score"]))
    qm.save_pretrained(tmp_path)
    (tmp_path / cli.LABELS_NAME).write_text(json.dumps({"base_model": "tiny", "labels": ["a", "b"], "max_length": 8}))
    model, _, info = cli.load_classifier(tmp_path)
    assert isinstance(model, QuantumModel) and not any(p.requires_grad for p in model.parameters())
    assert info["labels"] == ["a", "b"]


def test_experimental_method_needs_the_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_load_base", lambda name, labels: (make_tiny_gpt2(), CharTokenizer()))
    csv_path = write_csv(tmp_path / "d.csv", ROWS)
    args = ["train", "--csv", str(csv_path), "--out", str(tmp_path / "o"), "--method", "unitary", "--epochs", "1",
            "--device", "cpu"]
    with pytest.raises(ValueError, match="experimental"):
        cli.main(args)
    with pytest.warns(UserWarning, match="experimental"):
        cli.main(args + ["--experimental"])
    assert (tmp_path / "o" / "adapter_config.json").exists()


def test_bad_val_fraction_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["train", "--csv", "x.csv", "--out", "o", "--val-fraction", "1"])


def test_console_script_help():
    r = subprocess.run([sys.executable, "-m", "quantum_adapter.cli", "train", "--help"], capture_output=True,
                       text=True, check=True)
    assert "--csv" in r.stdout and "--target-modules" in r.stdout
