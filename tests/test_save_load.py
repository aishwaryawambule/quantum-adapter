import json
import os

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import (
    GPT2Config,
    GPT2ForSequenceClassification,
    RobertaConfig,
    RobertaForSequenceClassification,
)

from quantum_adapter import QuantumAdapterConfig, QuantumModel, __version__, get_quantum_model
from quantum_adapter import model as model_module

METHODS = ["amplitude", "unitary", "tensor"]
pytestmark = pytest.mark.filterwarnings("ignore:method=.* is experimental:UserWarning")


def fresh(seed=0, n_embd=32):
    torch.manual_seed(seed)
    cfg = GPT2Config(n_layer=2, n_embd=n_embd, n_head=2, n_positions=32, vocab_size=64, num_labels=2, pad_token_id=0)
    return GPT2ForSequenceClassification(cfg).eval()


def trained(method):
    qm = get_quantum_model(fresh(), QuantumAdapterConfig(
        target_modules=["c_attn"], method=method, rank=2, depth=2, tensor_axes=2, modules_to_save=["score"],
        experimental=method != "amplitude"))  # unitary and tensor are experimental
    with torch.no_grad():
        for p in qm.parameters():
            if p.requires_grad:
                p.add_(0.3 * torch.randn_like(p))
    return qm


@pytest.mark.parametrize("method", METHODS)
def test_round_trip_gives_identical_logits(tmp_path, ids, method):
    qm = trained(method)
    expected = qm(ids).logits
    qm.save_pretrained(tmp_path)
    loaded = QuantumModel.from_pretrained(fresh(), tmp_path)
    assert torch.equal(loaded(ids).logits, expected)


def test_files_are_small_and_versioned(tmp_path):
    qm = trained("amplitude")
    qm.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "adapter_config.json").read_text())
    assert saved["library_version"] == __version__ and saved["format_version"] == 1
    assert saved["layer_shapes"] == {"transformer.h.0.attn.c_attn": [32, 96], "transformer.h.1.attn.c_attn": [32, 96]}
    keys = set(load_file(tmp_path / "adapter_model.safetensors"))
    assert keys == set(qm.trainable_state_dict())
    assert all(".adapter." in k or k.startswith("score.") for k in keys)


def test_loaded_model_is_frozen_unless_trainable(tmp_path):
    trained("amplitude").save_pretrained(tmp_path)
    assert not any(p.requires_grad for p in QuantumModel.from_pretrained(fresh(), tmp_path).parameters())
    assert any(p.requires_grad for p in QuantumModel.from_pretrained(fresh(), tmp_path, is_trainable=True).parameters())


def test_wrong_base_model_is_rejected(tmp_path):
    trained("amplitude").save_pretrained(tmp_path)
    message = "saved for a model with different layers.*saved \\[32, 96\\], this model \\[64, 192\\]"
    with pytest.raises(ValueError, match=message):
        QuantumModel.from_pretrained(fresh(n_embd=64), tmp_path)


def test_loads_from_hub_repo_id(tmp_path, monkeypatch, ids):
    qm = trained("unitary")
    qm.save_pretrained(tmp_path)
    calls = {}

    def fake_download(repo_id, **kw):
        calls["repo_id"], calls["kw"] = repo_id, kw
        return str(tmp_path)

    monkeypatch.setattr(model_module, "snapshot_download", fake_download)
    loaded = QuantumModel.from_pretrained(fresh(), "someone/adapter", revision="v1")
    assert calls["repo_id"] == "someone/adapter" and calls["kw"]["revision"] == "v1"
    assert torch.equal(loaded(ids).logits, qm(ids).logits)


def test_push_to_hub_uploads_both_files(monkeypatch):
    uploaded = {}

    class FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, repo_id, private, exist_ok):
            uploaded["private"] = private

        def upload_folder(self, repo_id, folder_path, commit_message):
            uploaded["files"] = sorted(os.listdir(folder_path))

            class Info:
                commit_url = f"https://huggingface.co/{repo_id}/commit/abc"
            return Info()

    monkeypatch.setattr(model_module, "HfApi", FakeApi)
    url = trained("amplitude").push_to_hub("someone/adapter")
    assert uploaded == {"private": True, "files": ["adapter_config.json", "adapter_model.safetensors"]}
    assert url.endswith("/commit/abc")


def _untouched(model, flags):
    assert not any(isinstance(m, model_module.AdapterLayer) for m in model.modules())
    assert {n: p.requires_grad for n, p in model.named_parameters()} == flags


def test_wrong_shape_load_leaves_model_untouched(tmp_path):
    trained("amplitude").save_pretrained(tmp_path)
    other = fresh(n_embd=64)
    flags = {n: p.requires_grad for n, p in other.named_parameters()}
    with pytest.raises(ValueError, match="different layers"):
        QuantumModel.from_pretrained(other, tmp_path)
    _untouched(other, flags)
    cfg = QuantumAdapterConfig(target_modules=["c_attn"], method="amplitude", rank=2, depth=2)
    assert get_quantum_model(other, cfg).adapter_layers()


def test_wrong_shape_is_reported_even_if_construction_would_fail(tmp_path):
    qm = get_quantum_model(fresh(), QuantumAdapterConfig(
        target_modules=["c_attn"], method="amplitude", rank=2, depth=2, qubits=8))
    qm.save_pretrained(tmp_path)
    other = fresh(n_embd=64)
    with pytest.raises(ValueError, match="different layers"):
        QuantumModel.from_pretrained(other, tmp_path)
    assert not any(isinstance(m, model_module.AdapterLayer) for m in other.modules())


def test_missing_weights_file_leaves_model_untouched(tmp_path):
    trained("amplitude").save_pretrained(tmp_path)
    (tmp_path / "adapter_model.safetensors").unlink()
    other = fresh()
    other.transformer.wte.requires_grad_(False)  # user-frozen parameters must stay frozen
    flags = {n: p.requires_grad for n, p in other.named_parameters()}
    with pytest.raises(FileNotFoundError):
        QuantumModel.from_pretrained(other, tmp_path)
    _untouched(other, flags)


def test_bad_tensor_size_leaves_model_untouched(tmp_path):
    trained("amplitude").save_pretrained(tmp_path)
    weights = tmp_path / "adapter_model.safetensors"
    state = load_file(weights)
    key = next(k for k in state if k.startswith("score."))
    state[key] = torch.zeros(7)
    save_file(state, weights)
    other = fresh()
    other.transformer.wte.requires_grad_(False)
    flags = {n: p.requires_grad for n, p in other.named_parameters()}
    before = {n: p.clone() for n, p in other.state_dict().items()}
    with pytest.raises(ValueError, match="has shape"):
        QuantumModel.from_pretrained(other, tmp_path)
    _untouched(other, flags)
    assert all(torch.equal(v, other.state_dict()[n]) for n, v in before.items())


def test_weight_key_mismatch_leaves_model_untouched(tmp_path):
    trained("amplitude").save_pretrained(tmp_path)
    weights = tmp_path / "adapter_model.safetensors"
    state = load_file(weights)
    state.pop(next(k for k in state if k.startswith("score.")))
    save_file(state, weights)
    other = fresh()
    flags = {n: p.requires_grad for n, p in other.named_parameters()}
    with pytest.raises(ValueError, match="Adapter weights don't match.*missing"):
        QuantumModel.from_pretrained(other, tmp_path)
    _untouched(other, flags)


def tiny_roberta():
    torch.manual_seed(0)
    cfg = RobertaConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=40, num_labels=2)
    return RobertaForSequenceClassification(cfg).eval()


def test_nested_head_is_saved_and_reloaded(tmp_path, ids):
    # RoBERTa's head is classifier.dense + classifier.out_proj: every parameter that trains must be saved.
    qm = get_quantum_model(tiny_roberta(), QuantumAdapterConfig(
        target_modules=["query", "value"], rank=2, depth=2, modules_to_save=["classifier"]))
    trainable = {n.removeprefix("base_model.") for n, p in qm.named_parameters() if p.requires_grad}
    assert {"classifier.dense.weight", "classifier.out_proj.weight"} <= trainable
    assert trainable <= set(qm.trainable_state_dict())
    with torch.no_grad():
        for p in qm.parameters():
            if p.requires_grad:
                p.add_(0.3 * torch.randn_like(p))
    expected = qm(ids).logits
    qm.save_pretrained(tmp_path)
    other = tiny_roberta()
    with torch.no_grad():
        for p in other.classifier.parameters():
            p.normal_()  # a different random head: the loaded one must replace it
    loaded = QuantumModel.from_pretrained(other, tmp_path)
    assert torch.equal(loaded(ids).logits, expected)


@pytest.mark.parametrize("save", [["c_attn", "score"], ["attn"]])
def test_modules_to_save_overlapping_a_target_is_rejected(save):
    with pytest.raises(ValueError, match="modules_to_save.*target"):
        get_quantum_model(fresh(), QuantumAdapterConfig(target_modules=["c_attn"], rank=2, depth=2,
                                                       modules_to_save=save))


@pytest.mark.parametrize("onto", ["base_model", "wrapper"])
def test_load_onto_adapted_model_raises_and_keeps_its_adapters(tmp_path, ids, onto):
    trained("amplitude").save_pretrained(tmp_path)
    qm = trained("unitary")
    flags = {n: p.requires_grad for n, p in qm.named_parameters()}
    expected = qm(ids).logits
    with pytest.raises(ValueError, match="already has quantum adapters"):
        QuantumModel.from_pretrained(qm.base_model if onto == "base_model" else qm, tmp_path)
    assert len(qm.adapter_layers()) == 2
    assert {n: p.requires_grad for n, p in qm.named_parameters()} == flags
    assert torch.equal(qm(ids).logits, expected)


def test_restore_unwraps_only_the_layers_it_is_given():
    qm = trained("amplitude")
    model = qm.base_model
    flags = {id(p): p.requires_grad for p in model.parameters()}
    model_module._restore(model, flags, ["transformer.h.0.attn.c_attn"])
    assert not isinstance(model.transformer.h[0].attn.c_attn, model_module.AdapterLayer)
    assert isinstance(model.transformer.h[1].attn.c_attn, model_module.AdapterLayer)


def _edit_config(path, **changes):
    meta = json.loads((path / "adapter_config.json").read_text())
    for k, v in changes.items():
        if v is None:
            meta.pop(k)
        else:
            meta[k] = v
    (path / "adapter_config.json").write_text(json.dumps(meta))


def test_saved_config_has_format_version(tmp_path):
    trained("amplitude").save_pretrained(tmp_path)
    assert json.loads((tmp_path / "adapter_config.json").read_text())["format_version"] == 1


def test_config_without_format_version_loads_as_version_1(tmp_path, ids):
    qm = trained("unitary")
    qm.save_pretrained(tmp_path)
    _edit_config(tmp_path, format_version=None)
    assert torch.equal(QuantumModel.from_pretrained(fresh(), tmp_path)(ids).logits, qm(ids).logits)


def test_newer_format_version_is_rejected_untouched(tmp_path):
    trained("amplitude").save_pretrained(tmp_path)
    _edit_config(tmp_path, format_version=2)
    other = fresh()
    with pytest.raises(ValueError, match="format_version 2.*upgrade"):
        QuantumModel.from_pretrained(other, tmp_path)
    assert not any(isinstance(m, model_module.AdapterLayer) for m in other.modules())


def test_experimental_override_loads_a_later_demoted_method(tmp_path, ids, monkeypatch):
    qm = trained("amplitude")
    qm.save_pretrained(tmp_path)  # saved while amplitude was stable (experimental=False)
    monkeypatch.setattr(model_module, "EXPERIMENTAL", frozenset({"amplitude"}))
    with pytest.raises(ValueError, match="experimental=True"):
        QuantumModel.from_pretrained(fresh(), tmp_path)
    with pytest.warns(UserWarning, match="experimental"):
        loaded = QuantumModel.from_pretrained(fresh(), tmp_path, experimental=True)
    assert loaded.quantum_config.experimental is True
    assert torch.equal(loaded(ids).logits, qm(ids).logits)


@pytest.mark.parametrize("name", ["./missing", "/no/such/adapter", "~/no/such", "a/b/c", "adapter.safetensors"])
def test_missing_local_path_is_not_sent_to_the_hub(tmp_path, monkeypatch, name):
    def boom(*a, **k):
        raise AssertionError("must not contact the Hub")

    monkeypatch.setattr(model_module, "snapshot_download", boom)
    if name == "adapter.safetensors":
        (tmp_path / name).write_text("x")
        monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="No adapter directory at"):
        QuantumModel.from_pretrained(fresh(), name)
