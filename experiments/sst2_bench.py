"""First experiment: can a quantum circuit fine-tune GPT-2?

Frozen GPT-2 small on SST-2 sentiment. Every variant trains the classifier head; adapters sit on each layer's attention input projection (c_attn):
  head     : no adapter
  lora     : delta W = A @ B, rank 4, A and B trained directly
  quantum  : A and B are read from the amplitudes of a simulated 14-qubit circuit (RY layers + CNOT ring); only angles + 1 scale trained
  random   : same-size classical adapter: same A @ B shape and same parameter count as quantum, but A and B come from a fixed random projection of the trained parameters
LR picked on a held-out dev split (from train); reported on the GLUE validation set.
"""
import json, math, os, sys, time
import torch, torch.nn as nn
from datasets import load_dataset
from transformers import GPT2ForSequenceClassification, GPT2TokenizerFast

DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
RANK, NQ, DEPTH = 4, 14, 10
N_TRAIN, N_DEV, EPOCHS, BS, MAXLEN = int(os.environ.get('N_TRAIN', 2000)), 500, 3, 32, 64
# REUSE_LR=<results file>: skip the LR sweep and use each contender's best LR from that run
REUSE_LR = os.environ.get('REUSE_LR')

tok = GPT2TokenizerFast.from_pretrained('gpt2')
tok.pad_token = tok.eos_token
ds = load_dataset('nyu-mll/glue', 'sst2')
pool = ds['train'].shuffle(seed=0)

def tensors(split):
    e = tok(list(split['sentence']), truncation=True, max_length=MAXLEN, padding='max_length', return_tensors='pt')
    return e['input_ids'], e['attention_mask'], torch.tensor(list(split['label']))

TRAIN = tensors(pool.select(range(N_TRAIN)))
DEVSET = tensors(pool.select(range(N_TRAIN, N_TRAIN + N_DEV)))
VAL = tensors(ds['validation'])


def circuit_amplitudes(theta):
    """theta: (B, DEPTH, NQ) angles -> (B, 2**NQ) real amplitudes. RY on every qubit, then a CNOT ring, per layer."""
    B = theta.shape[0]
    state = torch.zeros(B, 2 ** NQ, device=theta.device)
    state[:, 0] = 1.0
    state = state.view(B, *([2] * NQ))
    for d in range(DEPTH):
        for q in range(NQ):
            st = state.movedim(q + 1, 1)
            shp = st.shape
            st = st.reshape(B, 2, -1)
            c = torch.cos(theta[:, d, q] / 2)[:, None]
            s = torch.sin(theta[:, d, q] / 2)[:, None]
            a, b = st[:, 0], st[:, 1]
            st = torch.stack([c * a - s * b, s * a + c * b], 1).reshape(shp)
            state = st.movedim(1, q + 1)
        for q in range(NQ):  # CNOT q -> q+1 (ring)
            t = (q + 1) % NQ
            st = state.movedim((q + 1, t + 1), (1, 2))
            st = torch.stack([st[:, 0], st[:, 1].flip(1)], 1)
            state = st.movedim((1, 2), (q + 1, t + 1))
    return state.reshape(B, -1)


class Adapted(nn.Module):
    """Wraps a GPT-2 Conv1D (y = x @ W + b) and adds x @ (A @ B) * scale."""
    def __init__(self, base, owner, idx):
        super().__init__()
        self.base, self.owner, self.idx = base, [owner], idx  # list: don't register owner as submodule

    def forward(self, x):
        A, B = self.owner[0].factors(self.idx)
        return self.base(x) + (x @ A) @ B


class AdapterBank(nn.Module):
    def __init__(self, kind, n_layers, nx, nf):
        super().__init__()
        self.kind, self.nx, self.nf = kind, nx, nf
        self.need = RANK * (nx + nf)
        if kind == 'lora':
            self.A = nn.Parameter(torch.randn(n_layers, nx, RANK) / math.sqrt(nx))
            self.B = nn.Parameter(torch.zeros(n_layers, RANK, nf))
        else:
            assert 2 ** NQ >= self.need
            self.theta = nn.Parameter(torch.rand(n_layers, DEPTH, NQ) * 2 * math.pi)
            self.scale = nn.Parameter(torch.zeros(n_layers))
            if kind == 'random':
                g = torch.Generator().manual_seed(123)
                self.register_buffer('P', torch.randn(DEPTH * NQ, self.need, generator=g) / math.sqrt(self.need))
        self.cache = None

    def refresh(self):
        if self.kind == 'quantum':
            v = circuit_amplitudes(self.theta.cpu())[:, : self.need].to(self.theta.device) * math.sqrt(2 ** NQ)  # sim on CPU: ~35x faster than MPS here
            self.cache = v
        elif self.kind == 'random':
            v = self.theta.flatten(1) @ self.P * math.sqrt(self.need) / math.sqrt(DEPTH * NQ)
            self.cache = v

    def factors(self, i):
        if self.kind == 'lora':
            return self.A[i], self.B[i]
        v = self.cache[i]
        A = v[: self.nx * RANK].view(self.nx, RANK) / math.sqrt(self.nx)
        B = v[self.nx * RANK:].view(RANK, self.nf) / math.sqrt(RANK) * self.scale[i]  # scale only B (like LoRA's B=0 init); scaling both gives zero grad at 0
        return A, B


def build(kind, seed):
    torch.manual_seed(seed)
    m = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2, attn_pdrop=0.0)  # MPS SDPA can't do attention dropout
    m.config.pad_token_id = tok.eos_token_id
    for p in m.parameters():
        p.requires_grad = False
    m.score.weight.requires_grad = True
    bank = None
    if kind != 'head':
        blocks = m.transformer.h
        nx, nf = blocks[0].attn.c_attn.weight.shape
        bank = AdapterBank(kind, len(blocks), nx, nf)
        for i, blk in enumerate(blocks):
            blk.attn.c_attn = Adapted(blk.attn.c_attn, bank, i)
    return m.to(DEV), (bank.to(DEV) if bank else None)


@torch.no_grad()
def accuracy(m, bank, data):
    m.eval()
    if bank: bank.refresh()
    ids, mask, y = data
    correct = 0
    for i in range(0, len(y), 128):
        out = m(input_ids=ids[i:i+128].to(DEV), attention_mask=mask[i:i+128].to(DEV)).logits
        correct += (out.argmax(-1).cpu() == y[i:i+128]).sum().item()
    return correct / len(y)


def run(kind, lr, seed):
    m, bank = build(kind, seed)
    params = [m.score.weight] + (list(bank.parameters()) if bank else [])
    n_train = sum(p.numel() for p in params)
    n_adapter = n_train - m.score.weight.numel()
    opt = torch.optim.AdamW([{'params': [m.score.weight], 'lr': 1e-3}] + ([{'params': list(bank.parameters()), 'lr': lr}] if bank else []), weight_decay=0.0)
    ids, mask, y = TRAIN
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for ep in range(EPOCHS):
        m.train()
        perm = torch.randperm(len(y), generator=g)
        for i in range(0, len(y), BS):
            b = perm[i:i+BS]
            if bank: bank.refresh()
            out = m(input_ids=ids[b].to(DEV), attention_mask=mask[b].to(DEV), labels=y[b].to(DEV))
            opt.zero_grad(); out.loss.backward(); opt.step()
    if DEV == 'mps': torch.mps.synchronize()
    secs = time.time() - t0
    return {'kind': kind, 'lr': lr, 'seed': seed, 'adapter_params': n_adapter, 'trainable': n_train,
            'dev_acc': accuracy(m, bank, DEVSET), 'val_acc': accuracy(m, bank, VAL), 'train_s': round(secs, 1)}


if __name__ == '__main__':
    grid = {'head': [None], 'lora': [3e-4, 1e-3, 3e-3], 'quantum': [3e-3, 1e-2, 3e-2], 'random': [3e-3, 1e-2, 3e-2]}
    kinds = sys.argv[1:] or list(grid)
    if REUSE_LR:
        prev = json.load(open(REUSE_LR))
        for k in grid:
            rs = [r for r in prev if r['kind'] == k][: len(grid[k])]
            grid[k] = [max(rs, key=lambda r: r['dev_acc'])['lr'] or None]
    results = []
    for k in kinds:
        best = None
        for lr in grid[k]:
            r = run(k, lr or 0.0, seed=0)
            print(json.dumps(r), flush=True); results.append(r)
            if best is None or r['dev_acc'] > best['dev_acc']: best = r
        for seed in (1, 2):
            r = run(k, best['lr'], seed)
            print(json.dumps(r), flush=True); results.append(r)
    json.dump(results, open(f'results/sst2_n{N_TRAIN}.json', 'w'), indent=1)
