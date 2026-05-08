"""
Classical vs Quantum performance benchmarks.
Compare VQC against SVM, MLP, and logistic regression on the same tasks.
SDKs: PennyLane, PyTorch, scikit-learn, Polars, DuckDB
"""
import time
import numpy as np
import polars as pl
import duckdb
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_iris, load_breast_cancer, make_moons


@dataclass
class BenchmarkResult:
    model_name: str
    dataset: str
    accuracy: float
    f1: float
    train_time_sec: float
    n_params: int
    cv_mean: float
    cv_std: float


CLASSICAL_MODELS = {
    "SVM (RBF)": lambda: SVC(kernel="rbf", C=10, gamma="scale", probability=True),
    "SVM (Linear)": lambda: SVC(kernel="linear", C=1, probability=True),
    "MLP (64-32)": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    "LogisticReg": lambda: LogisticRegression(max_iter=1000, random_state=42),
}

DATASETS = {
    "iris": lambda: load_iris(return_X_y=True),
    "breast_cancer": lambda: load_breast_cancer(return_X_y=True),
    "moons": lambda: make_moons(n_samples=500, noise=0.1, random_state=42),
}


class ClassicalVsQuantumBenchmark:
    """
    Side-by-side benchmark: classical ML vs VQC hybrid on same datasets.
    Results stored in DuckDB and exported as Polars DataFrames.
    """

    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db = duckdb.connect(str(self.output_dir / "results.db"))
        self._setup_db()
        self.results: List[BenchmarkResult] = []

    def _setup_db(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_results (
                model_name VARCHAR,
                dataset VARCHAR,
                accuracy DOUBLE,
                f1 DOUBLE,
                train_time_sec DOUBLE,
                n_params INTEGER,
                cv_mean DOUBLE,
                cv_std DOUBLE,
                run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def run_classical(self, dataset_name: str = "iris") -> List[BenchmarkResult]:
        """Benchmark all classical models on a dataset."""
        X, y = DATASETS[dataset_name]()
        scaler = StandardScaler()
        X = scaler.fit_transform(X.astype(np.float32))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = []
        for name, factory in CLASSICAL_MODELS.items():
            model = factory()
            t0 = time.time()
            model.fit(X_train, y_train)
            elapsed = time.time() - t0

            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")

            cv_scores = cross_val_score(factory(), X, y, cv=5, scoring="accuracy")

            n_params = 0
            if hasattr(model, "coef_"):
                n_params = model.coef_.size
            elif hasattr(model, "coefs_"):
                n_params = sum(c.size for c in model.coefs_)

            r = BenchmarkResult(
                model_name=name,
                dataset=dataset_name,
                accuracy=float(acc),
                f1=float(f1),
                train_time_sec=float(elapsed),
                n_params=n_params,
                cv_mean=float(cv_scores.mean()),
                cv_std=float(cv_scores.std()),
            )
            results.append(r)
            print(f"  {name:20s} | acc={acc:.3f} | f1={f1:.3f} | {elapsed:.2f}s | {n_params} params")

        self.results.extend(results)
        return results

    def run_quantum(self, dataset_name: str = "iris", n_qubits: int = 4, n_layers: int = 2) -> BenchmarkResult:
        """Benchmark VQC hybrid on same dataset."""
        from training.vqc_trainer import VQCTrainer, VQCTrainConfig
        cfg = VQCTrainConfig(
            dataset=dataset_name,
            n_qubits=n_qubits,
            n_layers=n_layers,
            epochs=30,
            wandb_project=None,
        )
        trainer = VQCTrainer(cfg)

        import wandb
        wandb.init(mode="disabled")
        t0 = time.time()
        result = trainer.train(run_name=f"vqc-bench-{dataset_name}")
        elapsed = time.time() - t0
        wandb.finish()

        model = result["model"]
        n_params = sum(p.numel() for p in model.parameters())
        best_acc = result["best_val_acc"]

        r = BenchmarkResult(
            model_name=f"VQC ({n_qubits}q, {n_layers}L)",
            dataset=dataset_name,
            accuracy=float(best_acc),
            f1=float(best_acc),
            train_time_sec=float(elapsed),
            n_params=n_params,
            cv_mean=float(best_acc),
            cv_std=0.0,
        )
        self.results.append(r)
        print(f"  {'VQC':20s} | acc={best_acc:.3f} | {elapsed:.2f}s | {n_params} params")
        return r

    def save_results(self) -> pl.DataFrame:
        """Save all results to DuckDB and return as Polars DataFrame."""
        for r in self.results:
            self.db.execute(
                "INSERT INTO benchmark_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                [r.model_name, r.dataset, r.accuracy, r.f1,
                 r.train_time_sec, r.n_params, r.cv_mean, r.cv_std]
            )

        df_pd = self.db.execute("SELECT * FROM benchmark_results ORDER BY accuracy DESC").fetchdf()
        df = pl.from_pandas(df_pd)
        df.write_parquet(str(self.output_dir / "benchmark_results.parquet"))
        return df

    def print_summary(self, df: pl.DataFrame):
        print("
" + "=" * 70)
        print("Benchmark Summary")
        print("=" * 70)
        print(df.select(["model_name", "dataset", "accuracy", "f1", "train_time_sec", "n_params"]))
