import pytest
import torch
from transformers import GPT2Config, GPT2ForSequenceClassification, LlamaConfig, LlamaForCausalLM


def make_tiny_gpt2():
    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=2, n_embd=32, n_head=2, n_positions=32, vocab_size=64, num_labels=2, pad_token_id=0,
                     bos_token_id=0, eos_token_id=0)
    return GPT2ForSequenceClassification(cfg).eval()


@pytest.fixture
def tiny_gpt2():
    return make_tiny_gpt2()


@pytest.fixture
def fresh_tiny_gpt2():
    """A factory for another tiny GPT-2 with the same seed and config as `tiny_gpt2`."""
    return make_tiny_gpt2


@pytest.fixture
def tiny_llama():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, vocab_size=64, max_position_embeddings=32,
    )
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture
def ids():
    torch.manual_seed(1)
    return torch.randint(1, 64, (3, 10))
