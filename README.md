# Quantum-Classical ML Bridge

Cluster 09 of the NextAura 500 SDKs / 25 Clusters project.

Hybrid quantum-classical models for optimization problems classical ML can't crack. Variational quantum circuits as drop-in PyTorch layers.

## Architecture

- PennyLane variational quantum circuits as PyTorch nn.Module layers
- Qiskit for gate-level circuit construction and IBM backend execution
- Cirq for Google quantum hardware circuits
- cuQuantum for GPU-accelerated quantum circuit simulation
- JAX + Flax for differentiable quantum-classical hybrid training
- Ray for distributed hyperparameter search over quantum circuits
- DSPy for quantum-augmented LLM reasoning pipelines
- W&B + MLflow for experiment tracking
- DuckDB + Polars for results analytics

## SDKs Used

PennyLane SDK, Qiskit SDK, Cirq SDK, cuQuantum, PyTorch, JAX, Flax, NumPy, SciPy, SymPy, Weights & Biases, MLflow, Ray SDK, FastAPI, Polars, DuckDB, Hugging Face Hub, Pydantic, Prometheus Client, DSPy

## Quickstart

```bash
pip install -r requirements.txt
python main.py --mode vqc --qubits 4 --layers 3 --epochs 50
python main.py --mode qaoa --problem maxcut --nodes 8
python main.py --mode hybrid --dataset iris
```

## Structure

```
circuits/      PennyLane + Qiskit + Cirq circuit definitions
layers/        Quantum nn.Module layers for PyTorch and JAX/Flax
training/      VQC trainer, QAOA solver, hybrid model trainer
optimization/  Quantum-classical optimization routines
benchmarks/    Classical vs quantum performance comparison
api/           FastAPI serving for quantum inference
main.py        Entry point
```
