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
                                                         num_labels=2, pad_token_id=0, bos_token_id=0, eos_token_id=0))
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
    # Real training: quantum adapters learn more slowly than LoRA, so train about twice as long
    # (e.g. 6 epochs where LoRA uses 3) with a learning rate near 1e-2. See docs/experiments.md.
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
