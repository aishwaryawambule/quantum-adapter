import copy
import inspect
import pickle
import sys

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from quantum_adapter import QuantumAdapterConfig, QuantumModel, get_quantum_model
from quantum_adapter.layers import AdapterLayer

METHODS = ["amplitude", "unitary", "tensor"]
pytestmark = pytest.mark.filterwarnings("ignore:method=.* is experimental:UserWarning")


def cfg(method="amplitude", **kw):
    kw.setdefault("target_modules", ["c_attn"])
    kw.setdefault("experimental", method != "amplitude")  # unitary and tensor are experimental
    return QuantumAdapterConfig(method=method, rank=2, depth=2, tensor_axes=2, **kw)


def perturb(qm):
    with torch.no_grad():
        for p in qm.parameters():
            if p.requires_grad:
                p.add_(0.3 * torch.randn_like(p))


@pytest.mark.parametrize("method", METHODS)
def test_wraps_targets_and_starts_neutral(tiny_gpt2, ids, method):
    before = tiny_gpt2(ids).logits.clone()
    qm = get_quantum_model(tiny_gpt2, cfg(method))
    assert qm.adapted_modules == ["transformer.h.0.attn.c_attn", "transformer.h.1.attn.c_attn"]
    assert isinstance(qm.base_model.transformer.h[0].attn.c_attn, AdapterLayer)
    assert torch.equal(qm(ids).logits, before)


def test_only_adapters_and_modules_to_save_train(tiny_gpt2):
    qm = get_quantum_model(tiny_gpt2, cfg(modules_to_save=["score"]))
    trainable = {n for n, p in qm.named_parameters() if p.requires_grad}
    assert trainable and all(".adapter." in n or n.startswith("base_model.score.") for n in trainable)
    assert "base_model.score.weight" in trainable


def test_llama_linear_layers(tiny_llama, ids):
    qm = get_quantum_model(tiny_llama, cfg(target_modules=["q_proj", "v_proj"]))
    assert len(qm.adapted_modules) == 4
    assert qm(ids).logits.shape == (3, 10, 64)


def test_regex_targets(tiny_gpt2):
    qm = get_quantum_model(tiny_gpt2, cfg(target_modules=r"transformer\.h\.0\.attn\.c_attn"))
    assert qm.adapted_modules == ["transformer.h.0.attn.c_attn"]


def test_no_match_lists_available_layers(tiny_gpt2):
    with pytest.raises(ValueError, match="No layers matched target_modules=\\['q_proj'\\].*c_attn.*c_fc.*c_proj"):
        get_quantum_model(tiny_gpt2, cfg(target_modules=["q_proj"]))


def test_rejects_unsupported_layer(tiny_gpt2):
    with pytest.raises(TypeError, match="'transformer.wte' is a Embedding"):
        get_quantum_model(tiny_gpt2, cfg(target_modules=["wte"]))


def test_rejects_double_adapt(tiny_gpt2):
    qm = get_quantum_model(tiny_gpt2, cfg())
    with pytest.raises(ValueError, match="already has quantum adapters"):
        get_quantum_model(qm.base_model, cfg())


def test_modules_to_save_must_match(tiny_gpt2):
    with pytest.raises(ValueError, match="modules_to_save=\\['classifier'\\] matched no module"):
        get_quantum_model(tiny_gpt2, cfg(modules_to_save=["classifier"]))


def test_experimental_method_needs_opt_in(tiny_gpt2, monkeypatch):
    monkeypatch.setattr("quantum_adapter.model.EXPERIMENTAL", frozenset({"tensor"}))
    with pytest.raises(ValueError, match="experimental=True"):
        get_quantum_model(tiny_gpt2, cfg("tensor", experimental=False))
    with pytest.warns(UserWarning, match="experimental"):
        get_quantum_model(tiny_gpt2, cfg("tensor", experimental=True))


def test_user_config_is_not_mutated(tiny_gpt2):
    c = cfg()
    get_quantum_model(tiny_gpt2, c)
    assert c.base_model_name_or_path is None


@pytest.mark.parametrize("method", METHODS)
def test_merge_matches_unmerged(tiny_gpt2, ids, method):
    qm = get_quantum_model(tiny_gpt2, cfg(method))
    perturb(qm)
    with torch.no_grad():
        expected = qm(ids).logits
        merged = qm.merge_and_unload()
        assert not any(isinstance(m, AdapterLayer) for m in merged.modules())
        assert torch.allclose(merged(ids).logits, expected, atol=1e-5)


@pytest.mark.parametrize("method", METHODS)
def test_merge_linear_layers(tiny_llama, ids, method):
    qm = get_quantum_model(tiny_llama, cfg(method, target_modules=["q_proj", "v_proj"]))
    perturb(qm)
    with torch.no_grad():
        expected = qm(ids).logits
        assert torch.allclose(qm.merge_and_unload()(ids).logits, expected, atol=1e-5)


# PyTorch's half-precision CPU kernels crash with an illegal instruction (0xc000001d) on some GitHub Windows
# runners, in PyTorch itself; the test passes on other Windows runners and on Linux and macOS.
@pytest.mark.skipif(sys.platform == "win32", reason="PyTorch half-precision CPU kernels crash on some Windows CPUs")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_model(tiny_gpt2, ids, dtype):
    qm = get_quantum_model(tiny_gpt2.to(dtype), cfg())
    perturb(qm)
    out = qm(ids).logits
    assert out.dtype == dtype and torch.isfinite(out).all()
    out.float().sum().backward()


def _accelerator():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


@pytest.mark.skipif(_accelerator() is None, reason="no GPU")
def test_runs_on_accelerator(tiny_gpt2, ids):
    dev = _accelerator()
    qm = get_quantum_model(tiny_gpt2, cfg()).to(dev)
    perturb(qm)
    out = qm(ids.to(dev)).logits
    assert out.device.type == dev
    out.sum().backward()


def test_trainable_parameter_report(tiny_gpt2, capsys):
    qm = get_quantum_model(tiny_gpt2, cfg())
    qm.print_trainable_parameters()
    assert "trainable params:" in capsys.readouterr().out


def test_circuit_summary(tiny_gpt2, capsys):
    rows = get_quantum_model(tiny_gpt2, cfg()).circuit_summary()
    assert [r["layer"] for r in rows] == ["transformer.h.0.attn.c_attn", "transformer.h.1.attn.c_attn"]
    assert "LoRA rank 2 would train" in capsys.readouterr().out


def test_attribute_fallback(tiny_gpt2):
    qm = get_quantum_model(tiny_gpt2, cfg())
    assert qm.config is tiny_gpt2.config


def _flags(model):
    return {n: p.requires_grad for n, p in model.named_parameters()}


@pytest.mark.parametrize(
    "bad, error",
    [
        ({"target_modules": ["q_proj"]}, ValueError),
        ({"target_modules": ["wte"]}, TypeError),
        ({"modules_to_save": ["classifier"]}, ValueError),
        ({"modules_to_save": ["c_attn"]}, ValueError),
        ({"method": "unitary", "max_qubits": 1}, ValueError),
    ],
)
def test_failed_adapt_leaves_model_untouched(tiny_gpt2, ids, bad, error):
    flags = _flags(tiny_gpt2)
    before = tiny_gpt2(ids).logits.clone()
    with pytest.raises(error):
        get_quantum_model(tiny_gpt2, cfg(**bad))
    assert not any(isinstance(m, AdapterLayer) for m in tiny_gpt2.modules())
    assert _flags(tiny_gpt2) == flags
    qm = get_quantum_model(tiny_gpt2, cfg(modules_to_save=["score"]))
    assert torch.equal(qm(ids).logits, before)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cast_after_adapting(tiny_gpt2, ids, dtype):
    qm = get_quantum_model(tiny_gpt2, cfg()).to(dtype)
    perturb(qm)
    out = qm(ids).logits
    assert out.dtype == dtype and torch.isfinite(out).all()
    out.float().sum().backward()


@pytest.mark.parametrize("dtype, atol", [(torch.float16, 5e-3), (torch.bfloat16, 5e-2)])
def test_merge_in_half_precision(tiny_gpt2, ids, dtype, atol):
    # Tolerances follow the dtype's epsilon (fp16 ~1e-3, bf16 ~8e-3) times logit scale.
    qm = get_quantum_model(tiny_gpt2.to(dtype), cfg())
    perturb(qm)
    with torch.no_grad():
        expected = qm(ids).logits
        merged = qm.merge_and_unload()
        assert not any(isinstance(m, AdapterLayer) for m in merged.modules())
        got = merged(ids).logits
    assert got.dtype == dtype
    assert torch.allclose(got.float(), expected.float(), atol=atol)


@pytest.mark.parametrize("fixture, target", [("tiny_gpt2", "c_attn"), ("tiny_llama", "q_proj")])
def test_forward_signature_is_the_wrapped_models(request, fixture, target):
    qm = get_quantum_model(request.getfixturevalue(fixture), cfg(target_modules=[target]))
    assert isinstance(qm, QuantumModel)
    for sig in (inspect.signature(qm.forward), inspect.signature(type(qm).forward)):
        assert {"input_ids", "labels"} <= set(sig.parameters)


def test_deepcopy_is_independent_and_keeps_signature(tiny_gpt2, ids):
    qm = get_quantum_model(tiny_gpt2, cfg())
    perturb(qm)
    clone = copy.deepcopy(qm)
    assert type(clone) is type(qm)
    assert {"input_ids", "labels"} <= set(inspect.signature(clone.forward).parameters)
    expected = qm(ids).logits
    assert torch.equal(clone(ids).logits, expected)
    with torch.no_grad():
        for p in clone.parameters():
            if p.requires_grad:
                p.add_(1.0)
    assert torch.equal(qm(ids).logits, expected)
    assert not torch.equal(clone(ids).logits, expected)


def test_circuit_summary_after_merge_raises(tiny_gpt2):
    qm = get_quantum_model(tiny_gpt2, QuantumAdapterConfig(target_modules=["c_attn"], rank=2, depth=2))
    qm.merge_and_unload()
    with pytest.raises(ValueError, match="no adapter layers.*before merging"):
        qm.circuit_summary()


def test_pickle_round_trip_keeps_outputs_and_signature(tiny_gpt2, ids):
    qm = get_quantum_model(tiny_gpt2, cfg(modules_to_save=["score"]))
    perturb(qm)
    expected = qm(ids).logits
    clone = pickle.loads(pickle.dumps(qm))
    assert type(clone).__name__ == type(qm).__name__ and isinstance(clone, QuantumModel)
    assert {"input_ids", "labels"} <= set(inspect.signature(clone.forward).parameters)
    assert clone.adapted_modules == qm.adapted_modules and clone.quantum_config == qm.quantum_config
    assert torch.equal(clone(ids).logits, expected)
    assert {n for n, p in clone.named_parameters() if p.requires_grad} == {
        n for n, p in qm.named_parameters() if p.requires_grad}


def test_torch_save_and_load(tiny_gpt2, ids, tmp_path):
    qm = get_quantum_model(tiny_gpt2, cfg())
    perturb(qm)
    torch.save(qm, tmp_path / "model.pt")
    loaded = torch.load(tmp_path / "model.pt", weights_only=False)
    assert torch.equal(loaded(ids).logits, qm(ids).logits)


def test_merge_refuses_a_layer_with_tied_weights(ids):
    torch.manual_seed(0)
    lm = GPT2LMHeadModel(GPT2Config(n_layer=1, n_embd=32, n_head=2, n_positions=32, vocab_size=64))
    assert lm.lm_head.weight is lm.transformer.wte.weight
    qm = get_quantum_model(lm, cfg(target_modules=["c_attn", "lm_head"]))
    perturb(qm)
    wte = lm.transformer.wte.weight.clone()
    with pytest.raises(ValueError, match="lm_head.*tied"):
        qm.merge_and_unload()
    assert len(qm.adapter_layers()) == 2  # nothing was merged
    assert torch.equal(lm.transformer.wte.weight, wte)
