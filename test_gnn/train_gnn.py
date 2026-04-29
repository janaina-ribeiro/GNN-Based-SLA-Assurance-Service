from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import joblib
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import Dataset, WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .dataset_builder import DelayGraphDataset, temporal_split
from .gnn_model import DelayGNN


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch, device: torch.device):
    return batch.to(device)


def _compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    preds = logits.argmax(dim=-1).detach().cpu().numpy()
    true = targets.detach().cpu().numpy()
    probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

    acc = accuracy_score(true, preds)
    balanced_acc = balanced_accuracy_score(true, preds)
    f1_macro = f1_score(true, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(true, preds, average="weighted", zero_division=0)
    precision, recall, _, _ = precision_recall_fscore_support(true, preds, average="binary", zero_division=0)

    if len(np.unique(true)) > 1:
        brier = brier_score_loss(true, probs)
    else:
        brier = 0.0
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision": float(precision),
        "recall": float(recall),
        "brier_score": float(brier),
    }


def _run_epoch(
    loader: DataLoader,
    model: DelayGNN,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_mode: bool,
    max_grad_norm: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    if train_mode:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_samples = 0
    logits_collect = []
    target_collect = []
    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            batch = _to_device(batch, device)
            logits = model(batch.x, batch.edge_index, getattr(batch, "edge_weight", None))
            loss = criterion(logits, batch.y)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            total_loss += float(loss.item()) * batch.y.numel()
            total_samples += int(batch.y.numel())
            logits_collect.append(logits)
            target_collect.append(batch.y)
    if not logits_collect:
        return 0.0, {"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0, "precision": 0.0, "recall": 0.0}
    stacked_logits = torch.cat(logits_collect, dim=0)
    stacked_targets = torch.cat(target_collect, dim=0)
    metrics = _compute_metrics(stacked_logits, stacked_targets)
    avg_loss = total_loss / max(total_samples, 1)
    metrics["loss"] = avg_loss
    return avg_loss, metrics


def _flatten_labels(dataset: DelayGraphDataset, indices) -> torch.Tensor:
    labels = dataset.labels[indices]
    return labels.view(-1)


def _make_class_weights(dataset: DelayGraphDataset, split_indices) -> torch.Tensor:
    flattened = _flatten_labels(dataset, split_indices)
    counts = torch.bincount(flattened, minlength=2).float()
    weights = counts.sum() / (counts * len(counts))
    weights = torch.where(torch.isfinite(weights), weights, torch.ones_like(weights))
    return weights


def _compute_sample_weights(dataset: DelayGraphDataset, split_indices, alpha: float = 2.0) -> torch.Tensor:
    """Compute weight for each sample based on proportion of high-delay links.
    
    Samples with more high-delay links get higher weights for balanced sampling.
    
    Args:
        dataset: The dataset containing labels
        split_indices: Indices of samples to compute weights for
        alpha: Scaling factor for weight computation (default: 2.0, range: 1.0-5.0)
    """
    labels = dataset.labels[split_indices]  
    high_delay_ratio = labels.float().mean(dim=1) 
    

    sample_weights = 1.0 + alpha * high_delay_ratio
    
    return sample_weights


def train_model(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    set_seed(args.seed)

    if getattr(args, "dataset_joblib", None) is not None and args.dataset_joblib is not None:
        dataset = DelayGraphDataset.load_joblib(args.dataset_joblib)
        print(f"[INFO] Loaded pre-built dataset from {args.dataset_joblib}")
    else:
        dataset = DelayGraphDataset(
            data_dir=args.data_dir,
            links=args.links,
            window_size=args.window_size,
            horizon_minutes=args.horizon_minutes,
            min_corr=args.min_corr,
            limit_samples=args.limit_samples,
        )
    split = temporal_split(dataset, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    train_indices = np.array(split.train.indices)  # type: ignore[attr-defined]

    class_weights = _make_class_weights(
        dataset=dataset,
        split_indices=train_indices,
    ).to(device)

    sample_weight_alpha = getattr(args, "sample_weight_alpha", 2.0)
    sample_weights = _compute_sample_weights(dataset, train_indices, alpha=sample_weight_alpha)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    train_loader = DataLoader(split.train, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(split.val, batch_size=args.batch_size)
    test_loader = DataLoader(split.test, batch_size=args.batch_size)
    model = DelayGNN(
        in_channels=dataset.num_node_features,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        num_classes=2,
        conv_type=args.conv_type,
        dropout=args.dropout,
        gat_heads=args.gat_heads,
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
            "links": dataset.link_order,
            "num_node_features": dataset.num_node_features,
            "hidden_channels": args.hidden_channels,
            "num_layers": args.num_layers,
            "conv_type": args.conv_type,
            "gat_heads": args.gat_heads,
            "dropout": args.dropout,
            "window_size": args.window_size,
            "horizon_minutes": args.horizon_minutes,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "min_corr": args.min_corr,
            "timestamp_range": {
                "start": dataset.sample_timestamps[0].isoformat(),
                "end": dataset.sample_timestamps[-1].isoformat(),
                "count": len(dataset.sample_timestamps),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GNN to classify link delay levels")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets_generated"))
    parser.add_argument("--links", nargs="*", default=["ac-am", "ac-ap", "ac-ba", "ac-ce"])
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--horizon-minutes", type=int, default=60)
    parser.add_argument("--min-corr", type=float, default=0.3)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument(
        "--dataset-joblib",
        type=Path,
        default=None,
        help="Optional path to a pre-built DelayGraphDataset (.joblib). If set, data-dir and CSVs are ignored.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--conv-type", type=str, default="gat")
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-path", type=Path, default=Path("test_gnn_delay_model.pt"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train_model(parse_args())