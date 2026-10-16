# Closing the gap to LoRA

Setup: SST-2, n=2000 training examples, 3 seeds, GPT-2 small, `c_attn` adapted unless noted, the same training script for every row. Accuracy is the mean over seeds.

| Configuration | Accuracy | Per seed |
|---|---|---|
| LoRA r4, 3 epochs, lr 1e-3 | 87.1% | 87.6 / 88.4 / 85.4 |
| amplitude, 3 epochs, lr 1e-2 | 84.8% | 85.4 / 83.6 / 85.3 |
| amplitude, depth 20 (3,372 params), 3 epochs | 82.8% | worse and less stable |
| amplitude on `c_attn` + `c_proj` + `c_fc` (6,648 params), 3 epochs | 85.3% | |
| amplitude + per-output scale (29,340 params), 3 epochs | 85.4% | |
| amplitude, 6 epochs | 87.3% | 86.5 / 86.7 / 88.6 |
| LoRA r4, 6 epochs | 87.7% | 87.6 / 87.3 / 88.2 |

## What it shows

- The gap at 3 epochs (2.3 points) mostly closes with longer training: at 6 epochs amplitude is 0.4 points behind LoRA.
- Depth 20 was worse and less stable in this test (3 epochs); deeper circuits were not tried at 6 epochs.
- Adapting more layers, or adding a per-output scale, recovered little (about half a point) at 3 epochs.

The 6-epoch comparison is the fair one, because quantum adapters learn more slowly: with enough training the gap closes. Learning rates: amplitude 1e-2, LoRA 1e-3. The 3-epoch rows show where each variant stands at a budget that favours LoRA.

So the recommendation is to train quantum adapters about 2x longer than LoRA with a learning rate near 1e-2. The release benchmark trains every contender for 6 epochs.

## Caveat

This is one task, one model, n=2000 and 3 seeds. Differences of a point or less are within seed-to-seed spread (LoRA's own seeds range over 3 points at 3 epochs). Treat the numbers as guidance, not a general law.
