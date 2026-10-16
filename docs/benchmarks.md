# Preliminary results (0.1.0)

These are early results from one task. The full benchmark runs next; see [Status of the full benchmark](#status-of-the-full-benchmark).

## 3-epoch SST-2 baseline

Setup: SST-2, GPT-2 small, 2,000 training examples, `c_attn` adapted, 3 epochs, 3 seeds. The learning rate was picked on a held-out dev split (seed 0), then 3 seeds were run at that rate. Accuracy is on the 872-review GLUE validation set. Raw rows: [`results/baseline-3ep-sst2.jsonl`](../results/baseline-3ep-sst2.jsonl) (this run predates the provenance fields, so its rows carry none). A partial AG News run (13 rows, stopped early) is kept in [`results/baseline-3ep-ag_news-partial.jsonl`](../results/baseline-3ep-ag_news-partial.jsonl) and is not analysed here.

| Method | Learning rate | Accuracy, 3 seeds: mean (min-max) | Adapter params | Trainable params | Train time (mean) |
|---|---|---|---|---|---|
| Head only (no adapter) | - | 71.7% (65.4-75.1) | 0 | 1,536 | 28 s |
| LoRA, rank 4 | 0.001 | 87.2% (85.4-88.4) | 147,456 | 148,992 | 68 s |
| amplitude | 0.01 | 84.8% (83.6-85.4) | 1,692 | 3,228 | 108 s |
| unitary | 0.01 | 83.9% (80.5-86.8) | 2,952 | 4,488 | 133 s |
| tensor | 0.0003 | 86.2% (85.4-86.7) | 156,288 | 157,824 | 127 s |
| Classical control, same size as amplitude | 0.03 | 84.7% (83.5-85.7) | 1,692 | 3,228 | 63 s |
| Classical control, same size as unitary | 0.03 | 84.8% (84.1-86.2) | 2,952 | 4,488 | 65 s |

What the table shows:

- At 3 epochs `amplitude` is 2.4 points behind LoRA r4 (84.8% against 87.2%) with 87 times fewer adapter parameters.
- The same-size classical controls score the same as the quantum adapters here (84.7% and 84.8%). This run gives no evidence that the circuit helps beyond having few parameters. The earlier standalone experiment in the README showed a 7-point lead over its own, weaker control; that lead did not hold up against this control. At 6 epochs the picture changes; see the next section.
- `unitary` is 3.3 points behind LoRA and its seeds vary widely (80.5 to 86.8).
- `tensor` scores close to LoRA (86.2%) but trains 156,288 adapter parameters, more than LoRA r4's 147,456, so it saves nothing.

## 6-epoch comparison (amplitude and LoRA)

Quantum adapters learn more slowly than LoRA, so 3 epochs is not a fair comparison. The 6-epoch comparison below is the fair one. Same setup as above; learning rates 1e-2 for amplitude and 1e-3 for LoRA. Raw rows, including the other variants tried: [`results/gap-experiment-sst2.jsonl`](../results/gap-experiment-sst2.jsonl). Details in [experiments.md](experiments.md).

| Method | Epochs | Accuracy, 3 seeds: mean (min-max) | Adapter params |
|---|---|---|---|
| LoRA, rank 4 | 6 | 87.7% (87.3-88.2) | 147,456 |
| amplitude | 6 | 87.3% (86.5-88.6) | 1,692 |

At 6 epochs the gap is 0.4 points, within the seed-to-seed spread. `unitary` and `tensor` were not run at 6 epochs.

## 6-epoch comparison: quantum circuit against an equal-size classical generator

Does the circuit help, or would any generator of 1,692 numbers do as well? Same setup, both trained 6 epochs by the benchmark harness (`benchmarks/run.py --task sst2 --contenders amplitude control:amplitude`, commit 96a0442, MPS). Learning rate picked on the dev split (seed 0): 3e-3 for amplitude, 1e-2 for the control. Raw rows: [`results/control-6ep-sst2.jsonl`](../results/control-6ep-sst2.jsonl).

| Seed | amplitude | Classical control, same size | Difference |
|---|---|---|---|
| 0 | 86.6% | 85.9% | +0.7 |
| 1 | 86.8% | 86.1% | +0.7 |
| 2 | 87.2% | 86.2% | +0.9 |
| Mean | 86.9% | 86.1% | +0.8 |

- The circuit led in every seed. Paired by seed, t = 10.0 with 2 degrees of freedom (p ≈ 0.01); unpaired (Welch) t = 3.9.
- The edge is small (0.8 points) and comes from one task, 2,000 training examples and 3 seeds. The control is one kind of classical generator (a fixed random projection); other generators may do better.
- The circuit took 1.7 times as long to train (230 s against 134 s).
- This amplitude run (86.9% at learning rate 3e-3) and the earlier 6-epoch run (87.3% at 1e-2) agree within their seed spreads.

## Status per method

| Method | Status | Why |
|---|---|---|
| `amplitude` | default, beta | 0.4 points behind LoRA at 6 epochs on SST-2 with 87 times fewer parameters. Not yet shown on other tasks. |
| `unitary` | experimental | About 3 points behind LoRA at 3 epochs on SST-2, with high seed variance. Not run at 6 epochs. |
| `tensor` | experimental | Trains more parameters than LoRA r4 on GPT-2-sized layers (13,024 against 12,288 per `c_attn` layer), so it cannot meet the 5x fewer-parameters rule. |

Experimental methods need `experimental=True` in the config and emit a warning.

## Status of the full benchmark

These results cover one task (SST-2), one model and one data size (2,000 examples). Do not treat them as general.

The full benchmark (SST-2, AG News and E2E text generation, 6 epochs, all contenders, 3 seeds) runs next. Its results will be published in 0.1.1. The formal production gate (accuracy within 1.0 point of LoRA r4 on both tasks, at least 5x fewer parameters, training time at most 2x LoRA, merged inference within 10% of the base model, and beating the same-size classical control) has not been applied yet, so no method is claimed to meet it.
