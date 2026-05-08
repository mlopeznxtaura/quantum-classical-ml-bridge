"""
quantum-classical-ml-bridge — Entry Point

Hybrid quantum-classical ML: VQC classification, QAOA optimization, benchmarks.

Usage:
  python main.py --mode vqc --dataset iris --qubits 4 --layers 2 --epochs 50
  python main.py --mode qaoa --problem maxcut --nodes 8 --layers 3
  python main.py --mode benchmark --dataset iris
"""
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Quantum-Classical ML Bridge")
    parser.add_argument("--mode", required=True,
                        choices=["vqc", "qaoa", "benchmark"],
                        help="Operation mode")
    parser.add_argument("--dataset", default="iris",
                        choices=["iris", "breast_cancer", "moons"])
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--ansatz", default="strongly_entangling",
                        choices=["strongly_entangling", "data_reuploading", "hardware_efficient"])
    parser.add_argument("--problem", default="maxcut", choices=["maxcut"])
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--backend", default="lightning.qubit",
                        help="PennyLane backend: default.qubit, lightning.qubit, lightning.gpu")
    parser.add_argument("--output", default="./results")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("  Quantum-Classical ML Bridge")
    print(f"  Mode: {args.mode.upper()} | Backend: {args.backend}")
    print("=" * 60)

    if args.mode == "vqc":
        from training.vqc_trainer import VQCTrainer, VQCTrainConfig
        cfg = VQCTrainConfig(
            dataset=args.dataset,
            n_qubits=args.qubits,
            n_layers=args.layers,
            ansatz=args.ansatz,
            epochs=args.epochs,
        )
        trainer = VQCTrainer(cfg)
        result = trainer.train()
        print(f"
Done. Best val accuracy: {result['best_val_acc']:.4f}")

    elif args.mode == "qaoa":
        from optimization.qaoa_solver import QAOASolver
        solver = QAOASolver(backend=args.backend)
        if args.problem == "maxcut":
            result = solver.maxcut(n_nodes=args.nodes, n_layers=args.layers)
            classical = solver.compare_classical(args.nodes, [])
            print(f"
QAOA cut: {result.best_cut} edges")
            print(f"Bitstring: {result.best_bitstring}")
            print(f"Time: {result.optimization_time:.2f}s")

    elif args.mode == "benchmark":
        from benchmarks.classical_vs_quantum import ClassicalVsQuantumBenchmark
        bench = ClassicalVsQuantumBenchmark(output_dir=args.output)
        print(f"
Classical models on {args.dataset}:")
        bench.run_classical(args.dataset)
        print(f"
VQC ({args.qubits}q, {args.layers}L) on {args.dataset}:")
        bench.run_quantum(args.dataset, args.qubits, args.layers)
        df = bench.save_results()
        bench.print_summary(df)


if __name__ == "__main__":
    main()
