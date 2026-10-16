"""Text-generation helpers for the E2E task: prompts, loss masking, reference handling, decoding, BLEU.

Kept free of dataset and model downloads so the pieces can be unit tested. Scores are returned as BLEU / 100
so that the gate's "fraction" units hold for both metrics.
"""

from __future__ import annotations

import ast

import torch

SEPARATOR = " => "
IGNORE = -100
MAX_NEW_TOKENS = 40


def prompt(mr: str) -> str:
    return f"{mr}{SEPARATOR}"


def parse_references(value) -> list[str]:
    """The references column may arrive as a list or as a stringified Python list."""
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return [str(v) for v in value]


def group_references(rows, limit: int) -> tuple[list[str], list[list[str]]]:
    """First `limit` unique meaning representations (in order) with all their references.

    Rows sharing a meaning representation are merged; a row with no references contributes its target.
    """
    refs: dict[str, list[str]] = {}
    for r in rows:
        mr = r["meaning_representation"]
        if mr not in refs and len(refs) >= limit:
            continue
        bucket = refs.setdefault(mr, [])
        for ref in parse_references(r["references"]) or [r["target"]]:
            if ref not in bucket:
                bucket.append(ref)
    return list(refs), list(refs.values())


def encode_pair(prompt_ids: list[int], target_ids: list[int], eos: int, maxlen: int) -> tuple[list[int], list[int]]:
    """Token ids and labels for one training example: loss only on the target tokens and the closing EOS."""
    room = max(0, maxlen - len(prompt_ids))
    # A target cut by maxlen still ends in EOS, so the model is always taught to stop.
    target = (list(target_ids[: max(0, room - 1)]) + [eos])[:room]
    return list(prompt_ids) + target, [IGNORE] * len(prompt_ids) + target


def pad_examples(examples, pad: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad (ids, labels) pairs; padding is masked out of attention and of the loss."""
    n = max(len(ids) for ids, _ in examples)
    ids = torch.full((len(examples), n), pad, dtype=torch.long)
    mask = torch.zeros((len(examples), n), dtype=torch.long)
    labels = torch.full((len(examples), n), IGNORE, dtype=torch.long)
    for i, (x, y) in enumerate(examples):
        ids[i, : len(x)] = torch.tensor(x)
        mask[i, : len(x)] = 1
        labels[i, : len(y)] = torch.tensor(y)
    return ids, mask, labels


def encode_dataset(tok, mrs: list[str], targets: list[str], maxlen: int):
    """Prompt and target are tokenised separately, so training sees exactly the prompt tokens generation will."""
    examples = [encode_pair(tok(prompt(mr))["input_ids"], tok(t)["input_ids"], tok.eos_token_id, maxlen)
                for mr, t in zip(mrs, targets)]
    return pad_examples(examples, tok.eos_token_id)


def clean_generation(text: str) -> str:
    return text.split("\n", 1)[0].strip()


@torch.no_grad()
def generate(model, tok, mrs: list[str], device: str, batch_size: int = 64, fixed_length: bool = False) -> list[str]:
    """Greedy decoding, left padded, stopping at EOS or a newline; the prompt is stripped from the output.

    `fixed_length=True` is for timing only: it forbids stopping early (min_new_tokens == max_new_tokens), so
    every model decodes exactly MAX_NEW_TOKENS tokens whatever it has learned. Without it a trained model that
    stops at EOS would look faster than the untrained base, which never stops early.
    """
    model.eval()
    stop = [tok.eos_token_id, *tok("\n")["input_ids"], *tok("\n\n")["input_ids"]]
    out = []
    for i in range(0, len(mrs), batch_size):
        enc = tok([prompt(m) for m in mrs[i:i + batch_size]], return_tensors="pt", padding=True).to(device)
        gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, eos_token_id=stop,
                             pad_token_id=tok.eos_token_id,
                             **({"min_new_tokens": MAX_NEW_TOKENS} if fixed_length else {}))
        new = gen[:, enc["input_ids"].shape[1]:]
        out += [clean_generation(t) for t in tok.batch_decode(new, skip_special_tokens=True)]
    return out


@torch.no_grad()
def next_token_logits(model, tok, mrs: list[str], device: str) -> torch.Tensor:
    """Logits for the first generated token of each prompt (left padded), on cpu: a continuous merge check."""
    model.eval()
    enc = tok([prompt(m) for m in mrs], return_tensors="pt", padding=True).to(device)
    positions = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)  # what generate() passes for left padding
    return model(**enc, position_ids=positions).logits[:, -1].float().cpu()


def bleu(hypotheses: list[str], references: list[list[str]]) -> float:
    """Corpus BLEU (sacrebleu defaults) divided by 100. Hypotheses may have different numbers of references."""
    from sacrebleu import corpus_bleu

    width = max(len(r) for r in references)
    streams = [[r[k] if k < len(r) else None for r in references] for k in range(width)]
    return corpus_bleu(hypotheses, streams).score / 100
