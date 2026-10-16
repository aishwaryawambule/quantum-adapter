import importlib
import math
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

from quantum_adapter import QuantumAdapterConfig, QuantumModel, QuantumTrainer, get_quantum_model


class Toy(torch.utils.data.Dataset):
    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.ids = torch.randint(1, 64, (32, 10), generator=g)
        self.labels = (self.ids[:, 0] > 32).long()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "labels": self.labels[i]}


def test_trainer_trains_and_adapter_reloads(tiny_gpt2, fresh_tiny_gpt2, ids, tmp_path):
    qm = get_quantum_model(tiny_gpt2, QuantumAdapterConfig(target_modules=["c_attn"], rank=2, depth=2,
                                                           modules_to_save=["score"]))
    before = {k: v.clone() for k, v in qm.trainable_state_dict().items()}
    args = TrainingArguments(output_dir=str(tmp_path / "run"), max_steps=4, per_device_train_batch_size=8,
                             learning_rate=1e-2, report_to=[], use_cpu=True, save_strategy="no")
    trainer = Trainer(model=qm, args=args, train_dataset=Toy(), eval_dataset=Toy())
    assert trainer.label_names == ["labels"]
    trainer.train()
    assert math.isfinite(trainer.evaluate()["eval_loss"])
    after = qm.trainable_state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)

    qm.save_pretrained(tmp_path / "adapter")
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()

    reloaded = QuantumModel.from_pretrained(fresh_tiny_gpt2(), tmp_path / "adapter")
    qm.eval()
    with torch.no_grad():
        expected = qm(ids).logits
        actual = reloaded(ids).logits
    assert torch.allclose(actual, expected, atol=1e-6)
    assert not torch.allclose(actual, fresh_tiny_gpt2()(ids).logits, atol=1e-6)  # the training really changed outputs


class ToyLM(torch.utils.data.Dataset):
    def __init__(self):
        self.ids = torch.randint(1, 64, (32, 10), generator=torch.Generator().manual_seed(0))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "labels": self.ids[i]}


def tiny_lm():
    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=2, n_embd=32, n_head=2, n_positions=32, vocab_size=64, tie_word_embeddings=True)
    return GPT2LMHeadModel(cfg)


def lm_trainer(out, max_steps, model=None):
    qm = model or get_quantum_model(tiny_lm(), QuantumAdapterConfig(target_modules=["c_attn"], rank=2, depth=2))
    # Default TrainingArguments apart from a short run: checkpoints are saved every 2 steps.
    args = TrainingArguments(output_dir=str(out), max_steps=max_steps, save_steps=2, per_device_train_batch_size=8,
                             learning_rate=1e-2, report_to=[], use_cpu=True)
    return qm, QuantumTrainer(model=qm, args=args, train_dataset=ToyLM())


def test_quantum_trainer_saves_adapter_only_checkpoints_on_a_tied_lm(tmp_path):
    qm, trainer = lm_trainer(tmp_path, 4)
    assert qm.base_model.lm_head.weight is qm.base_model.transformer.wte.weight
    trainer.train()
    for step in (2, 4):
        ckpt = tmp_path / f"checkpoint-{step}"
        assert (ckpt / "adapter_config.json").exists()
        assert not (ckpt / "model.safetensors").exists() and not (ckpt / "pytorch_model.bin").exists()
        assert set(load_file(ckpt / "adapter_model.safetensors")) == set(qm.trainable_state_dict())
    reloaded = QuantumModel.from_pretrained(tiny_lm(), tmp_path / "checkpoint-4")
    ids = ToyLM().ids[:2]
    qm.eval()
    with torch.no_grad():
        assert torch.equal(reloaded(ids).logits, qm(ids).logits)


def test_quantum_trainer_resumes_from_checkpoint(tmp_path):
    lm_trainer(tmp_path, 4)[1].train()
    saved = load_file(tmp_path / "checkpoint-4" / "adapter_model.safetensors")
    # Resuming at the last step trains nothing, so the model must hold exactly the checkpoint's weights.
    qm, trainer = lm_trainer(tmp_path, 4)
    trainer.train(resume_from_checkpoint=str(tmp_path / "checkpoint-4"))
    assert all(torch.equal(v, saved[k]) for k, v in qm.trainable_state_dict().items())
    qm, trainer = lm_trainer(tmp_path, 6)
    trainer.train(resume_from_checkpoint=True)
    assert trainer.state.global_step == 6
    assert (tmp_path / "checkpoint-6" / "adapter_model.safetensors").exists()


def test_importing_the_library_does_not_import_trainer():
    code = "import sys, quantum_adapter; assert 'transformers.trainer' not in sys.modules; quantum_adapter.QuantumTrainer"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_quantum_trainer_without_accelerate_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "accelerate", None)  # makes `import accelerate` raise ImportError
    monkeypatch.delitem(sys.modules, "quantum_adapter.trainer", raising=False)
    with pytest.raises(ImportError, match=r"pip install 'quantum-adapter\[trainer\]'"):
        importlib.import_module("quantum_adapter.trainer")
