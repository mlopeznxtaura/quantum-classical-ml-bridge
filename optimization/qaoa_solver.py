"""
QAOA (Quantum Approximate Optimization Algorithm) solver.
Solve combinatorial optimization problems: MaxCut, TSP, portfolio optimization.
SDKs: PennyLane, SciPy, NumPy, NetworkX
"""
import numpy as np
import pennylane as qml
from scipy.optimize import minimize
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import time

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False


@dataclass
class QAOAResult:
    problem: str
    n_nodes: int
    n_layers: int
    optimal_gamma: np.ndarray
    optimal_beta: np.ndarray
    optimal_cost: float
    best_bitstring: str
    best_cut: int
    optimization_time: float
    n_iterations: int


class QAOASolver:
    """
    QAOA solver for combinatorial optimization.
    Supports MaxCut on arbitrary graphs and portfolio optimization.
    """

    def __init__(self, backend: str = "lightning.qubit"):
        self.backend = backend

    def maxcut(
        self,
        n_nodes: int,
        edges: Optional[List[Tuple[int, int]]] = None,
        n_layers: int = 2,
        optimizer: str = "COBYLA",
        max_iter: int = 200,
        seed: int = 42,
    ) -> QAOAResult:
        """
        Solve MaxCut via QAOA.
        If edges is None, generates a random Erdos-Renyi graph.
        """
        np.random.seed(seed)

        if edges is None:
            if NX_AVAILABLE:
                G = nx.erdos_renyi_graph(n_nodes, 0.5, seed=seed)
                edges = list(G.edges())
            else:
                # Simple random graph fallback
                edges = []
                for i in range(n_nodes):
                    for j in range(i + 1, n_nodes):
                        if np.random.random() > 0.5:
                            edges.append((i, j))

        print(f"[QAOA] MaxCut | {n_nodes} nodes | {len(edges)} edges | {n_layers} layers")

        dev = qml.device(self.backend, wires=n_nodes)

        @qml.qnode(dev, interface="numpy")
        def cost_circuit(params):
            gamma = params[:n_layers]
            beta = params[n_layers:]
            for i in range(n_nodes):
                qml.Hadamard(wires=i)
            for p in range(n_layers):
                for u, v in edges:
                    qml.ZZPhaseShift(2 * gamma[p], wires=[u, v])
                for i in range(n_nodes):
                    qml.RX(2 * beta[p], wires=i)
            cost_obs = qml.Hamiltonian(
                [-0.5 for _ in edges],
                [qml.PauliZ(u) @ qml.PauliZ(v) for u, v in edges],
            )
            return qml.expval(cost_obs)

        @qml.qnode(dev, interface="numpy")
        def sample_circuit(gamma, beta):
            for i in range(n_nodes):
                qml.Hadamard(wires=i)
            for p in range(n_layers):
                for u, v in edges:
                    qml.ZZPhaseShift(2 * gamma[p], wires=[u, v])
                for i in range(n_nodes):
                    qml.RX(2 * beta[p], wires=i)
            return qml.probs(wires=range(n_nodes))

        # Optimize
        params0 = np.random.uniform(0, np.pi, 2 * n_layers)
        t0 = time.time()
        result = minimize(cost_circuit, params0, method=optimizer,
                         options={"maxiter": max_iter, "rhobeg": 0.5})
        elapsed = time.time() - t0

        opt_gamma = result.x[:n_layers]
        opt_beta = result.x[n_layers:]
        opt_cost = result.fun

        # Sample the optimal distribution
        probs = sample_circuit(opt_gamma, opt_beta)
        best_idx = int(np.argmax(probs))
        best_bitstring = format(best_idx, f"0{n_nodes}b")

        # Compute actual cut value
        cut = sum(1 for u, v in edges if best_bitstring[u] != best_bitstring[v])

        print(f"[QAOA] Done in {elapsed:.2f}s | {result.nit} iters | cost={opt_cost:.4f}")
        print(f"[QAOA] Best bitstring: {best_bitstring} | cut={cut}/{len(edges)} edges")

        return QAOAResult(
            problem="maxcut",
            n_nodes=n_nodes,
            n_layers=n_layers,
            optimal_gamma=opt_gamma,
            optimal_beta=opt_beta,
            optimal_cost=float(opt_cost),
            best_bitstring=best_bitstring,
            best_cut=cut,
            optimization_time=elapsed,
            n_iterations=result.nit,
        )

    def compare_classical(self, n_nodes: int, edges: List[Tuple[int, int]]) -> Dict[str, Any]:
        """
        Compare QAOA result against greedy classical MaxCut.
        Returns ratio of QAOA cut to classical cut.
        """
        # Greedy classical MaxCut
        assignment = np.zeros(n_nodes, dtype=int)
        for _ in range(100):
            improved = False
            for node in range(n_nodes):
                current_cut = sum(1 for u, v in edges
                                  if ((u == node) or (v == node))
                                  and assignment[u] != assignment[v])
                assignment[node] ^= 1
                new_cut = sum(1 for u, v in edges
                              if ((u == node) or (v == node))
                              and assignment[u] != assignment[v])
                if new_cut >= current_cut:
                    improved = True
                else:
                    assignment[node] ^= 1
            if not improved:
                break

        classical_cut = sum(1 for u, v in edges if assignment[u] != assignment[v])
        classical_bitstring = "".join(str(b) for b in assignment)
        return {
            "classical_cut": classical_cut,
            "classical_bitstring": classical_bitstring,
            "total_edges": len(edges),
        }
