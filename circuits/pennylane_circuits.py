"""
PennyLane variational quantum circuits.
Strongly entangling layers, data re-uploading, and hardware-efficient ansatze.
SDKs: PennyLane, PennyLane-Lightning (GPU), NumPy
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import Optional, List, Tuple, Callable


def make_device(n_qubits: int, backend: str = "lightning.qubit") -> qml.Device:
    """
    Create a PennyLane device.
    Backends: 'default.qubit', 'lightning.qubit' (fast CPU), 'lightning.gpu' (cuQuantum)
    """
    if backend == "lightning.gpu":
        try:
            return qml.device("lightning.gpu", wires=n_qubits)
        except Exception:
            print("[PennyLane] lightning.gpu unavailable, falling back to lightning.qubit")
            return qml.device("lightning.qubit", wires=n_qubits)
    return qml.device(backend, wires=n_qubits)


def strongly_entangling_circuit(n_qubits: int, n_layers: int, device: qml.Device):
    """
    Strongly entangling layers ansatz — standard VQC for classification.
    Parameters: (n_layers, n_qubits, 3) rotation angles
    """
    @qml.qnode(device, interface="torch", diff_method="adjoint")
    def circuit(inputs, weights):
        # Data encoding: angle encoding
        for i in range(n_qubits):
            qml.RX(inputs[i % len(inputs)], wires=i)
            qml.RY(inputs[(i + 1) % len(inputs)], wires=i)

        # Variational layers
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

        # Measurement: expectation values of Pauli-Z on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def data_reuploading_circuit(n_qubits: int, n_layers: int, device: qml.Device):
    """
    Data re-uploading circuit — re-encodes input at each layer.
    Universal approximator for quantum machine learning.
    """
    @qml.qnode(device, interface="torch", diff_method="adjoint")
    def circuit(inputs, weights):
        for layer in range(n_layers):
            # Re-upload data
            for i in range(n_qubits):
                qml.RX(inputs[i % len(inputs)] * weights[layer, i, 0], wires=i)
                qml.RY(inputs[(i + 1) % len(inputs)] * weights[layer, i, 1], wires=i)
                qml.RZ(weights[layer, i, 2], wires=i)

            # Entanglement
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            if n_qubits > 1:
                qml.CNOT(wires=[n_qubits - 1, 0])

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def hardware_efficient_ansatz(n_qubits: int, n_layers: int, device: qml.Device):
    """
    Hardware-efficient ansatz matching native gate sets of superconducting QPUs.
    Uses RY + CNOT — minimal gate depth for real hardware.
    """
    @qml.qnode(device, interface="torch", diff_method="parameter-shift")
    def circuit(inputs, weights):
        # Initial Hadamard layer
        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        for layer in range(n_layers):
            # Parameterized rotations
            for i in range(n_qubits):
                qml.RY(weights[layer, i], wires=i)

            # Linear entanglement
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Data re-encoding every other layer
            if layer % 2 == 0:
                for i in range(n_qubits):
                    qml.RX(inputs[i % len(inputs)], wires=i)

        return qml.probs(wires=range(min(n_qubits, 4)))

    return circuit


def qaoa_maxcut_circuit(n_nodes: int, edges: List[Tuple[int, int]], n_layers: int, device: qml.Device):
    """
    QAOA circuit for MaxCut problem.
    Alternates between cost (problem) and mixer (driver) unitaries.
    """
    @qml.qnode(device, interface="torch")
    def circuit(gamma, beta):
        # Initial superposition
        for i in range(n_nodes):
            qml.Hadamard(wires=i)

        for p in range(n_layers):
            # Cost unitary: ZZ interaction for each edge
            for u, v in edges:
                qml.ZZPhaseShift(2 * gamma[p], wires=[u, v])

            # Mixer unitary: X rotations
            for i in range(n_nodes):
                qml.RX(2 * beta[p], wires=i)

        # Measure cost Hamiltonian expectation
        cost_obs = qml.Hamiltonian(
            [-0.5 for _ in edges] + [0.5 * len(edges)],
            [qml.PauliZ(u) @ qml.PauliZ(v) for u, v in edges] + [qml.Identity(0)],
        )
        return qml.expval(cost_obs)

    return circuit


def draw_circuit(circuit_fn, inputs, weights, n_qubits: int):
    """Print circuit diagram."""
    print(qml.draw(circuit_fn)(inputs, weights))
