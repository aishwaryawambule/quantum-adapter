# Changelog

## 0.1.0 - 2026-10-16 (beta)

- Method statuses: `amplitude` is the default (beta); `unitary` and `tensor` are experimental and need `experimental=True`.
- Headline: LoRA-level accuracy (0.4 points behind on SST-2 at 6 epochs) with about 87x fewer trainable parameters. At 6 epochs the circuit beat an equal-size classical generator by 0.8 points in all 3 seeds (86.9% vs 86.1%); small, and one task so far.
- Results are preliminary: one task (SST-2, GPT-2 small) so far, see docs/benchmarks.md. The full benchmark (SST-2, AG News, E2E) runs next and its results will be published in 0.1.1; the formal production gate has not been applied yet.
- First release: `amplitude`, `unitary` (Quantum-PEFT) and `tensor` (QuanTA) adapter methods.
- Save / load / push to the Hugging Face Hub, merge for zero-overhead inference, Qiskit export.
- `quantum-adapter train` / `predict` command: fine-tune a text classifier on your own CSV file without writing code.
- `QuantumTrainer`: a `transformers.Trainer` with adapter-only checkpoints and `resume_from_checkpoint`.
- Saved adapters carry `format_version: 1`.
- API reference in docs/api.md.
- Release benchmark harness for SST-2 and AG News (accuracy) and E2E text generation (BLEU); see docs/benchmarks.md.
- Benchmark trains every contender for 6 epochs (was 3) and uses a generation merge tolerance of 7e-3 measured on real GPT-2; docs now say to train quantum adapters about 2x longer than LoRA (docs/experiments.md).
