"""A transformers Trainer whose checkpoints hold only the adapter, so they are small and tied weights are fine."""

from __future__ import annotations

import os

try:
    import accelerate  # noqa: F401  (transformers.Trainer needs it)
except ImportError as e:
    raise ImportError(
        "QuantumTrainer needs the `accelerate` package: pip install 'quantum-adapter[trainer]'") from e
import torch
from safetensors.torch import load_file
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME

from quantum_adapter.model import WEIGHTS_NAME, QuantumModel


class QuantumTrainer(Trainer):
    """`transformers.Trainer` for a `QuantumModel`, with adapter-only checkpoints.

    The plain Trainer saves the whole wrapped model at every checkpoint: that fails on models with tied
    embeddings (safetensors refuses shared tensors) and writes the full base model each time. QuantumTrainer
    instead writes each checkpoint with `QuantumModel.save_pretrained` (`adapter_config.json` +
    `adapter_model.safetensors`, plus the trainer's usual optimizer and state files), and
    `train(resume_from_checkpoint=...)` / `load_best_model_at_end` copy the adapter weights back into the
    model being trained. Every checkpoint directory can also be loaded with `QuantumModel.from_pretrained`.

    Use it exactly like `Trainer`; `model` must be the `QuantumModel` returned by `get_quantum_model`.
    """

    def _quantum_model(self, model: torch.nn.Module | None = None) -> QuantumModel:
        qm = self.accelerator.unwrap_model(model if model is not None else self.model)
        if not isinstance(qm, QuantumModel):
            raise TypeError(f"QuantumTrainer needs a QuantumModel, got {type(qm).__name__}; use get_quantum_model.")
        return qm

    def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if not self.args.should_save:
            return
        os.makedirs(output_dir, exist_ok=True)
        self._quantum_model().save_pretrained(output_dir)
        # transformers renamed `tokenizer` to `processing_class`; support both.
        processor = self.processing_class if hasattr(self, "processing_class") else getattr(self, "tokenizer", None)
        if processor is not None:
            processor.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_adapter(self, checkpoint: str, model: torch.nn.Module | None = None) -> None:
        weights = os.path.join(checkpoint, WEIGHTS_NAME)
        if not os.path.isfile(weights):
            raise ValueError(f"{checkpoint} has no {WEIGHTS_NAME}; was it saved by QuantumTrainer?")
        qm = self._quantum_model(model)
        device = next(qm.parameters()).device
        qm._load_trainable_state({k: v.to(device) for k, v in load_file(weights).items()})

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model: torch.nn.Module | None = None) -> None:
        self._load_adapter(resume_from_checkpoint, model)

    def _load_best_model(self) -> None:
        self._load_adapter(self.state.best_model_checkpoint)
