"""
Variational Quantum Circuit trainer.
Trains hybrid quantum-classical models end-to-end with PyTorch + W&B.
SDKs: PennyLane, PyTorch, Weights & Biases, MLflow, scikit-learn
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

import wandb
import mlflow
import mlflow.pytorch
from sklearn.datasets import load_iris, load_breast_cancer, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from layers.quantum_layer import HybridQuantumClassifier


@dataclass
class VQCTrainConfig:
    dataset: str = "iris"               # iris, breast_cancer, moons
    n_qubits: int = 4
    n_layers: int = 2
    ansatz: str = "strongly_entangling"
    hidden_dim: int = 64
    learning_rate: float = 0.01
    epochs: int = 50
    batch_size: int = 16
    test_size: float = 0.2
    seed: int = 42
    wandb_project: str = "quantum-classical-ml-bridge"
    device: str = "cpu"                 # Quantum sims are CPU-bound


DATASETS = {
    "iris": lambda: (load_iris(return_X_y=True)),
    "breast_cancer": lambda: (load_breast_cancer(return_X_y=True)),
    "moons": lambda: (make_moons(n_samples=500, noise=0.1, random_state=42)),
}


def load_dataset(name: str, test_size: float = 0.2, seed: int = 42) -> Tuple:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASETS.keys())}")
    X, y = DATASETS[name]()
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    y = y.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test


class VQCTrainer:
    """
    End-to-end trainer for hybrid quantum-classical classifiers.
    Logs all metrics to W&B and MLflow simultaneously.
    """

    def __init__(self, config: VQCTrainConfig = None):
        self.cfg = config or VQCTrainConfig()
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    def train(self, run_name: Optional[str] = None) -> Dict[str, Any]:
        cfg = self.cfg
        run_name = run_name or f"vqc-{cfg.ansatz}-{cfg.n_qubits}q-{cfg.dataset}"

        # Data
        X_train, X_test, y_train, y_test = load_dataset(cfg.dataset, cfg.test_size, cfg.seed)
        n_classes = len(np.unique(y_train))
        input_dim = X_train.shape[1]

        print(f"[VQC] Dataset: {cfg.dataset} | {input_dim}D -> {n_classes} classes")
        print(f"[VQC] {cfg.n_qubits} qubits | {cfg.n_layers} layers | ansatz: {cfg.ansatz}")

        # Model
        model = HybridQuantumClassifier(
            input_dim=input_dim,
            n_classes=n_classes,
            n_qubits=cfg.n_qubits,
            n_layers=cfg.n_layers,
            ansatz=cfg.ansatz,
            hidden_dim=cfg.hidden_dim,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

        # W&B
        wb_run = wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            config=vars(cfg),
        )

        mlflow.set_experiment(cfg.wandb_project)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(vars(cfg))

            history = {"train_loss": [], "train_acc": [], "val_acc": []}
            best_val_acc = 0.0

            for epoch in range(cfg.epochs):
                model.train()
                epoch_loss, epoch_preds, epoch_labels = [], [], []
                t0 = time.time()

                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()

                    epoch_loss.append(loss.item())
                    epoch_preds.extend(logits.argmax(dim=1).tolist())
                    epoch_labels.extend(y_batch.tolist())

                train_loss = np.mean(epoch_loss)
                train_acc = accuracy_score(epoch_labels, epoch_preds)

                # Validation
                model.eval()
                with torch.no_grad():
                    val_logits = model(torch.tensor(X_test))
                    val_preds = val_logits.argmax(dim=1).numpy()
                val_acc = accuracy_score(y_test, val_preds)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), f"checkpoints/{run_name}_best.pt")

                elapsed = time.time() - t0
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                history["val_acc"].append(val_acc)

                metrics = {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "epoch_time_sec": elapsed,
                }
                wandb.log(metrics, step=epoch)
                mlflow.log_metrics(metrics, step=epoch)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}/{cfg.epochs} | loss={train_loss:.4f} | "
                          f"train_acc={train_acc:.3f} | val_acc={val_acc:.3f} | {elapsed:.1f}s")

            print(f"
[VQC] Best val accuracy: {best_val_acc:.4f}")
            mlflow.log_metric("best_val_acc", best_val_acc)
            mlflow.pytorch.log_model(model, "model")

        wb_run.finish()
        return {"best_val_acc": best_val_acc, "history": history, "model": model}
