"""
Quantum nn.Module layers for PyTorch.
Drop-in quantum layers that plug into any classical neural network.
SDKs: PennyLane, PyTorch
"""
import torch
import torch.nn as nn
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import Optional, Callable, List
from dataclasses import dataclass


@dataclass
class QuantumLayerConfig:
    n_qubits: int = 4
    n_layers: int = 2
    ansatz: str = "strongly_entangling"   # "strongly_entangling", "data_reuploading", "hardware_efficient"
    backend: str = "lightning.qubit"       # "default.qubit", "lightning.qubit", "lightning.gpu"
    diff_method: str = "adjoint"           # "adjoint", "parameter-shift", "backprop"
    output_dim: Optional[int] = None       # None = n_qubits


class QuantumLayer(nn.Module):
    """
    A variational quantum circuit as a PyTorch nn.Module.
    Input: classical feature vector (pre-processed to n_qubits dims)
    Output: expectation values of Pauli-Z on each qubit (or probs)

    Usage:
        layer = QuantumLayer(QuantumLayerConfig(n_qubits=4, n_layers=2))
        out = layer(x)  # x: (batch, n_qubits)
    """

    def __init__(self, config: QuantumLayerConfig = None):
        super().__init__()
        self.cfg = config or QuantumLayerConfig()
        n_q = self.cfg.n_qubits
        n_l = self.cfg.n_layers

        # Device
        try:
            self.dev = qml.device(self.cfg.backend, wires=n_q)
        except Exception:
            self.dev = qml.device("default.qubit", wires=n_q)

        # Weight shapes per ansatz
        if self.cfg.ansatz == "strongly_entangling":
            weight_shape = (n_l, n_q, 3)
            self._circuit = self._build_strongly_entangling()
        elif self.cfg.ansatz == "data_reuploading":
            weight_shape = (n_l, n_q, 3)
            self._circuit = self._build_data_reuploading()
        else:  # hardware_efficient
            weight_shape = (n_l, n_q)
            self._circuit = self._build_hardware_efficient()

        # Trainable parameters
        self.weights = nn.Parameter(
            torch.randn(*weight_shape, dtype=torch.float32) * 0.1
        )
        out_dim = self.cfg.output_dim or n_q
        self.post_linear = nn.Linear(n_q, out_dim) if out_dim != n_q else None

        total_params = self.weights.numel()
        print(f"[QuantumLayer] {self.cfg.ansatz} | {n_q} qubits | {n_l} layers | {total_params} params")

    def _build_strongly_entangling(self):
        @qml.qnode(self.dev, interface="torch", diff_method=self.cfg.diff_method)
        def circuit(inputs, weights):
            for i in range(self.cfg.n_qubits):
                qml.RX(inputs[i], wires=i)
            qml.StronglyEntanglingLayers(weights, wires=range(self.cfg.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.cfg.n_qubits)]
        return circuit

    def _build_data_reuploading(self):
        n_q, n_l = self.cfg.n_qubits, self.cfg.n_layers
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, weights):
            for layer in range(n_l):
                for i in range(n_q):
                    qml.RX(inputs[i] * weights[layer, i, 0], wires=i)
                    qml.RY(inputs[i] * weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)
                for i in range(n_q - 1):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]
        return circuit

    def _build_hardware_efficient(self):
        n_q, n_l = self.cfg.n_qubits, self.cfg.n_layers
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, weights):
            for i in range(n_q):
                qml.Hadamard(wires=i)
            for layer in range(n_l):
                for i in range(n_q):
                    qml.RY(weights[layer, i], wires=i)
                for i in range(n_q - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(n_q):
                    qml.RX(inputs[i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, n_qubits) — input features normalized to [-pi, pi]
        Returns: (batch_size, output_dim)
        """
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            out = self._circuit(x[i], self.weights)
            outputs.append(torch.stack(out))
        out = torch.stack(outputs)  # (batch, n_qubits)

        if self.post_linear is not None:
            out = self.post_linear(out)
        return out


class HybridQuantumClassifier(nn.Module):
    """
    Classical feature extractor + quantum layer + classical head.
    Plug-and-play quantum ML model for tabular or image data.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        n_qubits: int = 4,
        n_layers: int = 2,
        ansatz: str = "strongly_entangling",
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_qubits = n_qubits

        # Classical pre-processing
        self.pre = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_qubits),
            nn.Tanh(),  # Normalize to [-1, 1] for angle encoding
        )

        # Quantum layer
        self.q_layer = QuantumLayer(QuantumLayerConfig(
            n_qubits=n_qubits,
            n_layers=n_layers,
            ansatz=ansatz,
        ))

        # Classical head
        self.head = nn.Sequential(
            nn.Linear(n_qubits, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)               # (B, n_qubits)
        x = x * np.pi                  # Scale to angle range
        x = self.q_layer(x)           # (B, n_qubits)
        return self.head(x)            # (B, n_classes)
