from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import joblib
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .dataset_builder_lstm import load_and_process_data


def set_seed(seed: int) -> None:

    """
    Set random seeds for reproducibility across all libraries.
    ----------------------------------------------------------

    Sets seeds for:
    - PyTorch (CPU and CUDA)
    - NumPy
    - Python's random module (implicitly)

    Args:
        seed: Integer seed value for reproducibility

    Returns:
        None
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    
    """
    Compute comprehensive classification metrics from model outputs.
    ---------------------------------------------------------------

    Metrics calculated:
    - accuracy: Overall accuracy
    - balanced_accuracy: Accuracy adjusted for class imbalance
    - f1_macro: Macro-averaged F1-score (unweighted)
    - f1_weighted: Weighted F1-score (by class support)
    - precision: Binary precision for positive class
    - recall: Binary recall for positive class
    - auc: ROC AUC score (requires both classes in targets)
    - brier_score: Brier score for probability calibration

    Args:
        logits: Raw model outputs (n_samples, n_classes)
        targets: Ground truth labels (n_samples,)

    Returns:
        Dict with 8 evaluation metrics
    """

    preds = logits.argmax(dim=-1).detach().cpu().numpy()
    true = targets.detach().cpu().numpy()

    probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

    acc = accuracy_score(true, preds)
    bal_acc = balanced_accuracy_score(true, preds)
    f1_macro = f1_score(true, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(true, preds, average="weighted", zero_division=0)
    precision, recall, _, _ = precision_recall_fscore_support(
        true, preds, average="binary", zero_division=0
    )

    if len(np.unique(true)) > 1:
        auc = roc_auc_score(true, probs)
        brier = brier_score_loss(true, probs)
    else:
        auc = 0.0
        brier = 0.0

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(auc),
        "brier_score": float(brier),
    }


def _run_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_mode: bool,
    max_grad_norm: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    
    """
    Run a single training or validation epoch.
    ------------------------------------------

    Operations:
    - Iterate over all batches in the DataLoader
    - Flatten link-level data for per-link prediction
    - Compute loss and predictions
    - Update weights (if train_mode=True) with gradient clipping
    - Aggregate metrics across all samples

    Args:
        loader: DataLoader with batched samples
        model: Neural network model (DelayRNN)
        criterion: Loss function (e.g., CrossEntropyLoss)
        optimizer: Optimizer for weight updates
        device: Device for computation (cuda/cpu)
        train_mode: If True, enable training and backpropagation
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)

    Returns:
        Tuple of (average_loss, metrics_dict)
    """

    if train_mode:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_samples = 0
    logits_collect = []
    target_collect = []

    with torch.set_grad_enabled(train_mode):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            batch_size, n_links, n_features = x_batch.shape
            x_flat = x_batch.view(batch_size * n_links, n_features)
            y_flat = y_batch.view(batch_size * n_links)

            logits = model(x_flat)
            loss = criterion(logits, y_flat)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            total_loss += float(loss.item()) * int(y_flat.numel())
            total_samples += int(y_flat.numel())
            logits_collect.append(logits.detach())
            target_collect.append(y_flat.detach())

    if not logits_collect:
        return 0.0, {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    stacked_logits = torch.cat(logits_collect, dim=0)
    stacked_targets = torch.cat(target_collect, dim=0)
    metrics = _compute_metrics(stacked_logits, stacked_targets)
    avg_loss = total_loss / max(total_samples, 1)
    metrics["loss"] = avg_loss
    return avg_loss, metrics


def _make_class_weights(labels: torch.Tensor) -> torch.Tensor:

    """
    Compute class weights from flattened labels tensor.
    ---------------------------------------------------

    Strategy:
    - Count occurrences of each class
    - Compute inverse frequency weights: weight[c] = n_total / (n_classes * count[c])
    - Handle edge cases with zero counts using torch.isfinite check

    Args:
        labels: Integer labels tensor of any shape (will be flattened)

    Returns:
        Tensor with weight for each class (shape: [n_classes])
    """

    flattened = labels.reshape(-1)
    counts = torch.bincount(flattened, minlength=2).float()
    weights = counts.sum() / (counts * len(counts))
    weights = torch.where(torch.isfinite(weights), weights, torch.ones_like(weights))
    return weights


def _compute_sample_weights(labels: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    
    """
    Compute weight for each sample based on proportion of high-delay links.
    -----------------------------------------------------------------------

    Strategy:
    - Samples with more high-delay links get higher weights
    - Helps balance sampling for multi-link outputs
    - Weight formula: weight = 1.0 + alpha * (ratio of class-1 links)

    Args:
        labels: Labels tensor (n_samples, n_links)
        alpha: Scaling factor for weight computation (default: 2.0, range: 1.0-5.0)

    Returns:
        Sample weights tensor (n_samples,)
    """

    high_delay_ratio = labels.float().mean(dim=1)  
    sample_weights = 1.0 + alpha * high_delay_ratio
    
    return sample_weights


class DelayRNN(nn.Module):
    """
    Simple LSTM/GRU classifier over per-link feature vectors.
    -----------------------------------------------
    Each link window is treated as a 1-step sequence of features, mirroring
    the aggregated-window representation used in the GNN pipeline.
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        num_layers: int = 1,
        rnn_type: str = "lstm",
        dropout: float = 0.0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        rnn_type = rnn_type.lower()
        if rnn_type not in {"lstm", "gru"}:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")

        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


def train_model(args: argparse.Namespace) -> None:
    
    """
    Train LSTM/GRU model for multi-link delay classification.
    ---------------------------------------------------------

    Training pipeline:
    1. Load dataset (from joblib cache or process from CSVs)
    2. Temporal split into train/val/test sets
    3. Compute class weights for imbalanced data
    4. Setup weighted random sampling for balanced batches
    5. Initialize model, optimizer, scheduler, and criterion
    6. Training loop with validation and early stopping
    7. Save best model, training history, and test metrics

    Model architecture:
    - DelayRNN: LSTM or GRU with configurable layers
    - Per-link classification (treats each link independently)

    Advanced techniques:
    - Label smoothing for regularization
    - Weighted sampling based on high-delay link proportion
    - Gradient clipping for stability
    - Learning rate scheduling (plateau or cosine)
    - Early stopping based on validation F1-macro

    Args:
        args: Namespace with all training configuration parameters

    Returns:
        None (saves model artifacts to disk)
    """
    device = torch.device(args.device)
    set_seed(args.seed)
    
    if getattr(args, "dataset_joblib", None) is not None and args.dataset_joblib is not None:
        print(f"[INFO] Loading pre-built LSTM dataset from {args.dataset_joblib}")
        dataset_artifact = joblib.load(args.dataset_joblib)
        X_np = dataset_artifact["X"]
        y_np = dataset_artifact["y"]
        links = dataset_artifact["links"]
        stamps = dataset_artifact["timestamps"]
        print(f"[INFO] Loaded: {X_np.shape[0]} samples, {X_np.shape[2]} features per link")
    else:
        X_np, y_np, links, stamps = load_and_process_data(
            data_dir=args.data_dir,
            window_size=args.window_size,
            horizon_minutes=args.horizon_minutes,
            limit_samples=args.limit_samples,
            delay_threshold=args.delay_threshold,
            delay_percentile=args.delay_percentile,
            column_delay=args.column_delay,
            links=args.links,
        )

    n_samples, n_links, n_features = X_np.shape

    stamps_array = np.array(stamps)
    order = np.argsort(stamps_array)
    n_train = int(n_samples * args.train_ratio)
    n_val = int(n_samples * args.val_ratio)
    n_test = n_samples - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split sizes with {n_samples} samples: "
            f"train_ratio={args.train_ratio}, val_ratio={args.val_ratio}"
        )

    train_indices = order[:n_train]
    val_indices = order[n_train : n_train + n_val]
    test_indices = order[n_train + n_val :]

    X_train_t = torch.from_numpy(X_np[train_indices]).float()
    y_train_t = torch.from_numpy(y_np[train_indices]).long()

    X_val_t = torch.from_numpy(X_np[val_indices]).float()
    y_val_t = torch.from_numpy(y_np[val_indices]).long()

    X_test_t = torch.from_numpy(X_np[test_indices]).float()
    y_test_t = torch.from_numpy(y_np[test_indices]).long()

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    class_weights = _make_class_weights(y_train_t).to(device)

    sample_weight_alpha = getattr(args, "sample_weight_alpha", 2.0)
    sample_weights = _compute_sample_weights(y_train_t, alpha=sample_weight_alpha)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = DelayRNN(
        in_features=n_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
        num_classes=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    label_smoothing = getattr(args, "label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    
    scheduler = None
    scheduler_type = getattr(args, "scheduler", "none")
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=getattr(args, "scheduler_factor", 0.5),
            patience=getattr(args, "scheduler_patience", 5),
            verbose=True,
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=getattr(args, "scheduler_t0", 10),
            T_mult=1,
            eta_min=args.lr * 0.01,
            verbose=True,
        )
    
    max_grad_norm = getattr(args, "max_grad_norm", 1.0)
    patience = getattr(args, "patience", 0)
    patience_counter = 0
    
    best_state = None
    best_metric = -1.0
    best_epoch = 0
    training_history = []
    
    for epoch in range(1, args.epochs + 1):
        _, train_metrics = _run_epoch(train_loader, model, criterion, optimizer, device, train_mode=True, max_grad_norm=max_grad_norm)
        _, val_metrics = _run_epoch(val_loader, model, criterion, optimizer, device, train_mode=False)
        val_f1 = val_metrics.get("f1_macro", 0.0)
        
        training_history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        })
        
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(val_f1)
            elif scheduler_type == "cosine":
                scheduler.step()
        
        if val_f1 > best_metric:
            best_metric = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
        else:
            patience_counter += 1
        
        print(
            f"Epoch {epoch:03d} | Train Loss {train_metrics['loss']:.4f} | "
            f"Train F1 {train_metrics['f1_macro']:.4f} | Val F1 {val_metrics['f1_macro']:.4f} | "
            f"Val BalAcc {val_metrics['balanced_accuracy']:.4f} | LR {optimizer.param_groups[0]['lr']:.2e}"
        )
        
        if patience > 0 and patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs (patience={patience})")
            break
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    _, test_metrics = _run_epoch(test_loader, model, criterion, optimizer, device, train_mode=False)
    print("Test metrics:", json.dumps(test_metrics, indent=2))
    artifact = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": {
            "links": links,
            "num_features": n_features,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "rnn_type": args.rnn_type,
            "dropout": args.dropout,
            "window_size": args.window_size,
            "horizon_minutes": args.horizon_minutes,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "timestamp_range": {
                "start": stamps[0].isoformat(),
                "end": stamps[-1].isoformat(),
                "count": len(stamps),
            },
            "best_val_f1": best_metric,
            "best_epoch": best_epoch,
            "total_epochs": len(training_history),
            "scheduler": getattr(args, "scheduler", "none"),
            "early_stopped": len(training_history) < args.epochs,
            "training_history": training_history,
            "test_metrics": test_metrics,
        },
    }
    torch.save(artifact, args.model_path)
    print(f"Model saved to {args.model_path}")

    joblib_path = args.model_path.with_suffix(".joblib")
    joblib.dump(artifact, joblib_path)
    print(f"Joblib model saved to {joblib_path}")

    history_json = {
        "config": {
            "data_dir": str(args.data_dir),
            "links": args.links,
            "window_size": args.window_size,
            "horizon_minutes": args.horizon_minutes,
            "limit_samples": args.limit_samples,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "epochs": args.epochs,
            "device": args.device,
            "seed": args.seed,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "rnn_type": args.rnn_type,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "label_smoothing": getattr(args, "label_smoothing", 0.0),
            "sample_weight_alpha": getattr(args, "sample_weight_alpha", 2.0),
            "max_grad_norm": getattr(args, "max_grad_norm", 1.0),
            "scheduler": getattr(args, "scheduler", "none"),
            "scheduler_factor": getattr(args, "scheduler_factor", 0.5),
            "scheduler_patience": getattr(args, "scheduler_patience", 5),
            "scheduler_t0": getattr(args, "scheduler_t0", 10),
            "patience": getattr(args, "patience", 0),
            "delay_threshold": getattr(args, "delay_threshold", None),
            "delay_percentile": getattr(args, "delay_percentile", 85.0),
            "column_delay": getattr(args, "column_delay", "Atraso"),
            "model_path": str(args.model_path),
        },
        "history": training_history,
        "test_metrics": test_metrics,
        "best_epoch": best_epoch,
        "best_val_f1": best_metric,
    }
    
    history_json_path = args.model_path.with_name(args.model_path.stem + "_history.json")
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history_json, f, indent=2)
    print(f"Training history saved to {history_json_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for LSTM/GRU model training.
    ---------------------------------------------------------

    Argument groups:
    1. Data configuration:
       - data-dir: Directory with CSV files
       - links: Subset of links to process
       - dataset-joblib: Pre-built dataset cache (bypasses data loading)
       - window-size: Temporal window size
       - horizon-minutes: Prediction horizon
       - limit-samples: Sample limit for testing

    2. Training splits:
       - train-ratio: Proportion for training (default: 0.7)
       - val-ratio: Proportion for validation (default: 0.15)

    3. Model architecture:
       - hidden-size: Hidden units in RNN layers (default: 64)
       - num-layers: Number of RNN layers (default: 2)
       - rnn-type: 'lstm' or 'gru' (default: 'lstm')
       - dropout: Dropout rate (default: 0.2)

    4. Optimization:
       - epochs: Training epochs (default: 50)
       - batch-size: Batch size (default: 32)
       - lr: Learning rate (default: 1e-3)
       - weight-decay: L2 regularization (default: 1e-4)

    5. Advanced techniques:
       - label-smoothing: Smoothing factor (default: 0.0)
       - sample-weight-alpha: Sample weighting factor (default: 2.0)
       - max-grad-norm: Gradient clipping threshold (default: 1.0)
       - scheduler: LR scheduler type ('none', 'plateau', 'cosine')
       - patience: Early stopping patience (default: 0, disabled)

    6. Delay classification:
       - delay-threshold: Fixed threshold in ms (optional)
       - delay-percentile: Percentile threshold (default: 85.0)
       - column-delay: Column name (default: 'Atraso')

    7. Output:
       - device: 'cuda' or 'cpu' (auto-detected)
       - model-path: Output path for trained model
       - seed: Random seed for reproducibility (default: 42)

    Returns:
        Namespace with all parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train an LSTM/GRU model to classify link delay levels",
    )

    parser.add_argument("--data-dir", type=Path, default=Path("datasets_generated"))
    parser.add_argument("--links", nargs="*", default=None, help="Subset of links to use (default: all)")
    parser.add_argument(
        "--dataset-joblib",
        type=Path,
        default=None,
        help="Optional path to a pre-built LSTM dataset (.joblib). If set, data-dir is ignored.",
    )
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--horizon-minutes", type=int, default=60)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--rnn-type", type=str, default="lstm", choices=["lstm", "gru"])
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing factor (default: 0.0)")
    parser.add_argument("--sample-weight-alpha", type=float, default=2.0,
                        help="Sample weight scaling factor (default: 2.0)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping (default: 1.0)")
    parser.add_argument("--scheduler", type=str, default="none",
                        choices=["none", "plateau", "cosine"],
                        help="Learning rate scheduler type (default: none)")
    parser.add_argument("--scheduler-factor", type=float, default=0.5,
                        help="Factor for ReduceLROnPlateau (default: 0.5)")
    parser.add_argument("--scheduler-patience", type=int, default=5,
                        help="Patience for ReduceLROnPlateau (default: 5)")
    parser.add_argument("--scheduler-t0", type=int, default=10,
                        help="T0 for CosineAnnealingWarmRestarts (default: 10)")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (0 = disabled, default: 0)")

    parser.add_argument("--delay-threshold", type=float, default=None,
                        help="Fixed delay threshold in ms (overrides percentile if set)")
    parser.add_argument("--delay-percentile", type=float, default=85.0,
                        help="Percentile for delay threshold when no fixed threshold is given")
    parser.add_argument("--column-delay", type=str, default="Atraso")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-path", type=Path, default=Path("lstm_gru_delay_model.pt"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train_model(parse_args())
