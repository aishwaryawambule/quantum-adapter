import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import textgen


def test_parse_references_accepts_list_and_stringified_list():
    assert textgen.parse_references(["a", "b"]) == ["a", "b"]
    assert textgen.parse_references("['a', \"it's b\"]") == ["a", "it's b"]
    assert textgen.parse_references([]) == []


def test_group_references_merges_rows_with_the_same_mr_and_keeps_first_seen_order():
    rows = [
        {"meaning_representation": "m1", "target": "t1", "references": ["a"]},
        {"meaning_representation": "m2", "target": "t2", "references": "['x', 'y']"},
        {"meaning_representation": "m1", "target": "t1b", "references": ["a", "b"]},
        {"meaning_representation": "m3", "target": "t3", "references": []},
    ]
    mrs, refs = textgen.group_references(rows, limit=3)
    assert mrs == ["m1", "m2", "m3"]
    assert refs == [["a", "b"], ["x", "y"], ["t3"]]  # no references: fall back to the row's target
    assert textgen.group_references(rows, limit=2)[0] == ["m1", "m2"]


def test_encode_pair_masks_prompt_and_keeps_target_and_eos():
    ids, labels = textgen.encode_pair([1, 2, 3], [7, 8], eos=0, maxlen=10)
    assert ids == [1, 2, 3, 7, 8, 0]
    assert labels == [-100, -100, -100, 7, 8, 0]


def test_encode_pair_truncates_the_target_not_the_prompt():
    ids, labels = textgen.encode_pair([1, 2, 3], [7, 8, 9, 10], eos=0, maxlen=5)
    assert ids == [1, 2, 3, 7, 0]  # the cut target still ends in EOS
    assert labels == [-100, -100, -100, 7, 0]
    assert textgen.encode_pair([1, 2, 3], [7], eos=0, maxlen=3) == ([1, 2, 3], [-100, -100, -100])


def test_pad_examples_masks_padding_in_attention_and_labels():
    ids, mask, labels = textgen.pad_examples([([1, 2, 7, 0], [-100, -100, 7, 0]), ([1, 5, 0], [-100, 5, 0])], pad=0)
    assert ids.tolist() == [[1, 2, 7, 0], [1, 5, 0, 0]]
    assert mask.tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    assert labels.tolist() == [[-100, -100, 7, 0], [-100, 5, 0, -100]]


def test_prompt_format():
    assert textgen.prompt("name[X]") == "name[X] => "


def test_gpt2_tokenizer_masking_matches_the_generation_prompt():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    except OSError:
        pytest.skip("gpt2 tokenizer not cached")
    mr, target = "name[The Eagle], food[Thai]", "The Eagle serves Thai food."
    ids, mask, labels = textgen.encode_dataset(tok, [mr], [target], maxlen=128)
    n_prompt = len(tok(textgen.prompt(mr))["input_ids"])
    assert ids[0, :n_prompt].tolist() == tok(textgen.prompt(mr))["input_ids"]
    assert (labels[0, :n_prompt] == -100).all()
    assert labels[0, n_prompt:].tolist() == ids[0, n_prompt:].tolist()
    assert ids[0, -1].item() == tok.eos_token_id and mask[0].all()


def test_clean_generation_cuts_at_newline_and_strips():
    assert textgen.clean_generation("  The Eagle is nice.\nname[X] => foo") == "The Eagle is nice."
    assert textgen.clean_generation("plain") == "plain"


def test_bleu_toy_corpus():
    pytest.importorskip("sacrebleu")
    hyps = ["the cat sat on the mat today", "a big dog ran in the park"]
    assert textgen.bleu(hyps, [[h] for h in hyps]) == pytest.approx(1.0)  # BLEU/100
    worse = textgen.bleu(["the cat sat on a mat today", "a big dog ran in the park"], [[h] for h in hyps])
    assert 0.0 < worse < 1.0
    assert textgen.bleu(["zzz qqq"], [["the cat sat"]]) == 0.0


def test_bleu_accepts_different_numbers_of_references_per_hypothesis():
    pytest.importorskip("sacrebleu")
    hyps = ["the cat sat on the mat today", "a big dog ran in the park"]
    refs = [["the cat sat on the mat today", "the cat sat on a mat"], ["a big dog ran in the park"]]
    assert textgen.bleu(hyps, refs) == pytest.approx(1.0)


# --- generation on a tiny random GPT-2 with a character-level stub tokenizer (no downloads) ---

EOS, NEWLINE = 0, 10


class CharTok:
    """ids are character codes (< 128); id 0 is EOS and the padding token; pads on the left."""

    eos_token_id = EOS

    def __call__(self, text, return_tensors=None, padding=False):
        from transformers import BatchEncoding

        single = isinstance(text, str)
        seqs = [[ord(c) for c in t] for t in ([text] if single else text)]
        if single:
            return {"input_ids": seqs[0]}
        n = max(map(len, seqs))
        ids = torch.tensor([[EOS] * (n - len(s)) + s for s in seqs])
        mask = torch.tensor([[0] * (n - len(s)) + [1] * len(s) for s in seqs])
        return BatchEncoding({"input_ids": ids, "attention_mask": mask})

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["".join(chr(i) for i in row if i != EOS) for row in ids.tolist()]


def tiny_model(forced: int | None = None):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    m = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=128, n_positions=128, n_embd=16, n_layer=1, n_head=2, eos_token_id=EOS)).eval()
    if forced is not None:  # constant final hidden state, and only token `forced` has a non-zero output row
        with torch.no_grad():
            m.transformer.ln_f.weight.zero_()
            m.transformer.ln_f.bias.fill_(1.0)
            m.transformer.wte.weight.zero_()
            m.transformer.wte.weight[forced] = 1.0
    return m


def test_generate_left_pads_so_batching_does_not_change_the_output():
    m, tok = tiny_model(), CharTok()
    alone = textgen.generate(m, tok, ["ab"], "cpu", fixed_length=True)
    batched = textgen.generate(m, tok, ["ab", "a much longer meaning representation"], "cpu", fixed_length=True)
    assert alone[0] == batched[0]
    prompt_text = textgen.prompt("ab")
    assert not alone[0].startswith(prompt_text)  # only the continuation is returned


@pytest.mark.parametrize("stop", [EOS, NEWLINE])
def test_generate_stops_at_eos_or_newline_unless_timing(stop, monkeypatch):
    m, tok = tiny_model(forced=stop), CharTok()
    assert textgen.generate(m, tok, ["ab", "cd"], "cpu") == ["", ""]
    seen = {}
    real = m.generate

    def spy(**kw):
        out = real(**kw)
        seen["new"] = out.shape[1] - kw["input_ids"].shape[1]
        return out

    monkeypatch.setattr(m, "generate", spy)
    textgen.generate(m, tok, ["ab"], "cpu")
    assert seen["new"] == 1  # stopped straight after the forced token
    textgen.generate(m, tok, ["ab"], "cpu", fixed_length=True)
    assert seen["new"] == textgen.MAX_NEW_TOKENS  # timing never stops early


def test_next_token_logits_shape_and_padding_invariance():
    m, tok = tiny_model(), CharTok()
    both = textgen.next_token_logits(m, tok, ["ab", "a much longer meaning representation"], "cpu")
    alone = textgen.next_token_logits(m, tok, ["ab"], "cpu")
    assert both.shape == (2, 128)
    assert torch.allclose(both[0], alone[0], atol=1e-4)
