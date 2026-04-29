from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

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
from torch.utils.data import DataLoader, TensorDataset


def _compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute classification metrics from logits and targets.
    -------------------------------------------------------

    Metrics calculated:
    - accuracy: Overall accuracy
    - balanced_accuracy: Accuracy adjusted for class imbalance
    - f1_macro: Macro-averaged F1-score
    - f1_weighted: Weighted F1-score by class support
    - precision: Binary precision for positive class
    - recall: Binary recall for positive class
    - auc: ROC AUC score (if both classes present)
    - brier_score: Brier score for probability calibration

    Args:
        logits: Raw model outputs (n_samples, n_classes)
        targets: Ground truth labels (n_samples,)

    Returns:
        Dict with 8 evaluation metrics
    """
    # Move to CPU numpy
    preds = logits.argmax(dim=-1).detach().cpu().numpy()
    true = targets.detach().cpu().numpy()

    # Class probabilities for positive class (needed for AUC)
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


class DelayRNN(nn.Module):
    """Simple LSTM/GRU classifier over per-link feature vectors.

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
        # x: (batch_nodes, in_features)
        x = x.unsqueeze(1)  # (batch_nodes, seq_len=1, in_features)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


def predict(args: argparse.Namespace) -> None:
    """
    Run prediction on a pre-built dataset using a trained model.
    ------------------------------------------------------------

    Prediction pipeline:
    1. Load trained model (.pt or .joblib) and extract metadata
    2. Load pre-built dataset (.joblib) with features and labels
    3. Validate feature dimensions match model expectations
    4. Run inference on all samples in batches
    5. Compute comprehensive evaluation metrics
    6. Optionally save results to JSON file

    Output metrics:
    - Same as training: accuracy, balanced_accuracy, F1, precision, recall, AUC, Brier
    - Printed to console and optionally saved to file

    Args:
        args: Namespace with prediction configuration (model_path, dataset_joblib, etc.)

    Returns:
        Dict with computed metrics
    """
    device = torch.device(args.device)
    
    # Load trained model
    print(f"[INFO] Loading model from {args.model_path}")
    if args.model_path.suffix == ".joblib":
        model_artifact = joblib.load(args.model_path)
    else:
        model_artifact = torch.load(args.model_path, map_location=device)
    
    metadata = model_artifact.get("metadata", {})
    
    # Extract model configuration from metadata
    n_features = metadata.get("num_features")
    hidden_size = metadata.get("hidden_size")
    num_layers = metadata.get("num_layers")
    rnn_type = metadata.get("rnn_type", "lstm")
    dropout = metadata.get("dropout", 0.0)
    
    if n_features is None or hidden_size is None or num_layers is None:
        raise ValueError(
            "Model metadata incomplete. Required: num_features, hidden_size, num_layers"
        )
    
    # Initialize model architecture
    model = DelayRNN(
        in_features=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        rnn_type=rnn_type,
        dropout=dropout,
        num_classes=2,
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(model_artifact["model_state_dict"])
    model.eval()
    print(f"[INFO] Model loaded successfully")
    print(f"[INFO] Model config: {rnn_type.upper()}, hidden={hidden_size}, layers={num_layers}")
    
    # Load dataset
    print(f"[INFO] Loading dataset from {args.dataset_joblib}")
    dataset_artifact = joblib.load(args.dataset_joblib)
    X_np = dataset_artifact["X"]
    y_np = dataset_artifact["y"]
    links = dataset_artifact.get("links", [])
    timestamps = dataset_artifact.get("timestamps", [])
    
    n_samples, n_links, n_features_data = X_np.shape
    print(f"[INFO] Dataset loaded: {n_samples} samples, {n_links} links, {n_features_data} features")
    
    # Validate feature dimensions
    if n_features_data != n_features:
        raise ValueError(
            f"Feature dimension mismatch: model expects {n_features}, "
            f"dataset has {n_features_data}"
        )
    
    # Convert to tensors
    X_tensor = torch.from_numpy(X_np).float()
    y_tensor = torch.from_numpy(y_np).long()
    
    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    import time
    # Run prediction
    print("[INFO] Running predictions...")
    logits_collect = []
    target_collect = []
    start_time = time.time()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # x_batch: (batch_size, n_links, n_features)
            # y_batch: (batch_size, n_links)
            batch_size, n_links_batch, n_features_batch = x_batch.shape
            x_flat = x_batch.view(batch_size * n_links_batch, n_features_batch)
            y_flat = y_batch.view(batch_size * n_links_batch)
            logits = model(x_flat)
            logits_collect.append(logits.detach())
            target_collect.append(y_flat.detach())
    elapsed_time = time.time() - start_time
    # Concatenate all predictions
    stacked_logits = torch.cat(logits_collect, dim=0)
    stacked_targets = torch.cat(target_collect, dim=0)
    # Compute metrics
    metrics = _compute_metrics(stacked_logits, stacked_targets)
    metrics["prediction_time_seconds"] = elapsed_time
    # Print results
    print("\n" + "="*60)
    print("PREDICTION METRICS")
    print("="*60)
    print(json.dumps(metrics, indent=2))
    print(f"Prediction time (s): {elapsed_time:.4f}")
    print("="*60 + "\n")
    # Save results if output path specified
    if args.output_path:
        results = {
            "model_path": str(args.model_path),
            "dataset_path": str(args.dataset_joblib),
            "num_samples": n_samples,
            "num_links": n_links,
            "num_features": n_features,
            "links": links,
            "timestamp_info": {
                "count": len(timestamps),
                "start": timestamps[0].isoformat() if timestamps else None,
                "end": timestamps[-1].isoformat() if timestamps else None,
            } if timestamps else None,
            "metrics": metrics,
            "model_metadata": metadata,
            "prediction_time_seconds": elapsed_time,
        }
        output_path = Path(args.output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Results saved to {output_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for LSTM/GRU model prediction.
    -----------------------------------------------------------

    Required arguments:
    - model-path: Path to trained model file (.pt or .joblib)
    - dataset-joblib: Path to pre-built dataset (.joblib)

    Optional arguments:
    - batch-size: Batch size for inference (default: 32)
    - device: Computation device (default: auto-detect cuda/cpu)
    - output-path: Path to save JSON results (default: None, print only)

    Returns:
        Namespace with all parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run predictions using a trained LSTM/GRU delay classifier"
    )
    
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model (.pt or .joblib file)",
    )
    parser.add_argument(
        "--dataset-joblib",
        type=Path,
        required=True,
        help="Path to pre-built LSTM dataset (.joblib file)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for prediction (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run predictions on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to save prediction results as JSON",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())
