import pytest

from quantum_adapter import QuantumAdapterConfig


def test_defaults():
    c = QuantumAdapterConfig(target_modules=["c_attn"])
    assert (c.method, c.rank, c.qubits, c.depth, c.entanglement) == ("amplitude", 4, "auto", 10, "ring")
    assert c.modules_to_save == [] and c.experimental is False


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"method": "magic"}, "method must be one of"),
        ({"rank": 0}, "rank must be a positive integer"),
        ({"depth": -1}, "depth must be a positive integer"),
        ({"entanglement": "star"}, "entanglement must be one of"),
        ({"qubits": "many"}, "qubits must be 'auto' or a positive integer"),
        ({"qubits": 25}, "max_qubits=20"),
        ({"tensor_axes": 1}, "tensor_axes must be between 2 and 8"),
        ({"target_modules": []}, "target_modules must name at least one layer"),
    ],
)
def test_rejects_bad_values(kwargs, message):
    args = {"target_modules": ["c_attn"], **kwargs}
    with pytest.raises(ValueError, match=message):
        QuantumAdapterConfig(**args)


def test_dict_round_trip():
    c = QuantumAdapterConfig(
        target_modules=["q_proj", "v_proj"], method="unitary", rank=8, modules_to_save=["score"]
    )
    assert QuantumAdapterConfig.from_dict(c.to_dict()) == c


def test_from_dict_ignores_library_version_but_rejects_unknown_keys():
    d = QuantumAdapterConfig(target_modules=["c_attn"]).to_dict()
    assert QuantumAdapterConfig.from_dict({**d, "library_version": "9.9.9"}).target_modules == ["c_attn"]
    with pytest.raises(ValueError, match="newer quantum-adapter"):
        QuantumAdapterConfig.from_dict({**d, "future_option": 1})


def test_regex_target_modules_kept_as_string():
    assert QuantumAdapterConfig(target_modules=r".*\.attn\.c_attn").target_modules == r".*\.attn\.c_attn"
