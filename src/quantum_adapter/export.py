"""Turn trained circuits into Qiskit circuits (needs the [qiskit] extra)."""

from __future__ import annotations

from quantum_adapter.methods import CircuitSpec
from quantum_adapter.sim import entangling_pairs


def to_qiskit(spec: CircuitSpec):
    try:
        from qiskit import QuantumCircuit
    except ImportError as e:
        raise ImportError("Qiskit export needs the optional extra: pip install 'quantum-adapter[qiskit]'") from e
    n = spec.n_qubits
    qc = QuantumCircuit(n, name=spec.name)

    def wire(q: int) -> int:          # our qubit 0 is the most significant bit; Qiskit's is the least
        return n - 1 - q

    pairs = entangling_pairs(n, spec.entanglement)
    layers = spec.theta.shape[0]
    for k in range(layers):
        for q in range(n):
            qc.ry(float(spec.theta[k, q]), wire(q))
        if not (spec.final_rotation and k == layers - 1):
            for a, b in pairs:
                (qc.cx if spec.gate == "cx" else qc.cz)(wire(a), wire(b))
    return qc
