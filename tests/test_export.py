import math

import numpy as np
import pytest
import torch

qiskit = pytest.importorskip("qiskit")
from qiskit.quantum_info import Statevector

from quantum_adapter import QuantumAdapterConfig, get_quantum_model
from quantum_adapter.export import to_qiskit
from quantum_adapter.methods import CircuitSpec
from quantum_adapter.sim import basis_states, run_ansatz

pytestmark = pytest.mark.filterwarnings("ignore:method=.* is experimental:UserWarning")


@pytest.mark.parametrize("pattern", ["ring", "line", "full"])
@pytest.mark.parametrize("gate", ["cx", "cz"])
@pytest.mark.parametrize("final", [False, True])
def test_simulator_matches_qiskit(pattern, gate, final):
    n = 5
    theta = torch.rand(3, n, dtype=torch.float64) * 2 * math.pi
    qc = to_qiskit(CircuitSpec("t", n, theta, pattern, gate, final))
    ours = run_ansatz(basis_states(4, n, dtype=torch.float64), theta, n,
                      entanglement=pattern, gate=gate, final_rotation=final).numpy()
    for j in range(4):
        sv = Statevector.from_int(j, 2**n).evolve(qc).data
        assert np.allclose(sv.imag, 0, atol=1e-10)
        assert np.allclose(sv.real, ours[j], atol=1e-10)


def test_model_export(tiny_gpt2):
    qm = get_quantum_model(tiny_gpt2, QuantumAdapterConfig(target_modules=["c_attn"], method="unitary", rank=2, depth=2,
                                                                          experimental=True))
    circuits = qm.export_circuits()
    assert list(circuits) == ["transformer.h.0.attn.c_attn", "transformer.h.1.attn.c_attn"]
    u, v = circuits["transformer.h.0.attn.c_attn"]
    assert (u.num_qubits, v.num_qubits) == (5, 7)       # 32 -> 5 qubits, 96 -> 7 qubits


def test_tensor_method_has_nothing_to_export(tiny_gpt2):
    qm = get_quantum_model(tiny_gpt2, QuantumAdapterConfig(target_modules=["c_attn"], method="tensor", tensor_axes=2,
                                                                          experimental=True))
    with pytest.raises(ValueError, match="quantum-inspired"):
        qm.export_circuits()


def test_export_after_merge_raises(tiny_gpt2):
    qm = get_quantum_model(tiny_gpt2, QuantumAdapterConfig(target_modules=["c_attn"], rank=2, depth=2))
    qm.merge_and_unload()
    with pytest.raises(ValueError, match="no adapter layers.*before merging"):
        qm.export_circuits()
