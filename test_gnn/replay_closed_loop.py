from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from .dataset_builder import DelayGraphDataset
from .gnn_model import DelayGNN
from .policy_engine import PolicyEngine


def _parse_thresholds(raw: str) -> Dict[int, float]:
    """Parse thresholds in the format: 3:0.55,10:0.45,30:0.35"""
    result: Dict[int, float] = {}
    chunks = [c.strip() for c in raw.split(",") if c.strip()]
    for chunk in chunks:
        h, thr = chunk.split(":", 1)
        result[int(h)] = float(thr)
    return result


def _resolve_timestamp(value) -> pd.Timestamp:
    if isinstance(value, (list, tuple)):
        value = value[0]
    if isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value)


def _load_artifact(model_path: Path, device: torch.device) -> dict:
    artifact = torch.load(model_path, map_location=device)
    if "model_state_dict" not in artifact or "metadata" not in artifact:
        raise RuntimeError(f"Invalid model artifact: {model_path}")
    return artifact


def _build_model(artifact: dict, device: torch.device) -> DelayGNN:
    meta = artifact["metadata"]
    model = DelayGNN(
        in_channels=int(meta["num_node_features"]),
        hidden_channels=int(meta["hidden_channels"]),
        num_layers=int(meta["num_layers"]),
        num_classes=2,
        conv_type=str(meta["conv_type"]),
        dropout=float(meta["dropout"]),
        gat_heads=int(meta.get("gat_heads", 2)),
    )
    model.load_state_dict(artifact["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _run_single_horizon(
    *,
    horizon: int,
    dataset: DelayGraphDataset,
    model: DelayGNN,
    device: torch.device,
    policy: PolicyEngine,
) -> List[dict]:
    records: List[dict] = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    link_order = dataset.link_order

    with torch.no_grad():
        for data in loader:
            timestamp_t = _resolve_timestamp(data.timestamp)
            predicted_violation_ts = timestamp_t + timedelta(minutes=horizon)

            edge_weight = getattr(data, "edge_weight", None)
            edge_weight = edge_weight.to(device) if edge_weight is not None else None

            logits = model(data.x.to(device), data.edge_index.to(device), edge_weight)
            probs = torch.softmax(logits, dim=-1).cpu()
            risk_high = probs[:, 1]
            y_true = data.y.cpu()

            for link_idx, link in enumerate(link_order):
                risk = float(risk_high[link_idx].item())
                actual_violation = int(y_true[link_idx].item())

                decision = policy.decide(
                    timestamp=timestamp_t.to_pydatetime(),
                    link=link,
                    horizon_minutes=horizon,
                    risk=risk,
                )

                lead_time_minutes = horizon if (decision.alert_triggered == 1 and actual_violation == 1) else None

                records.append(
                    {
                        "timestamp": timestamp_t.isoformat(),
                        "predicted_violation_timestamp": predicted_violation_ts.isoformat(),
                        "link": link,
                        "horizon_minutes": int(horizon),
                        "pred_risk": risk,
                        "actual_violation": actual_violation,
                        "alert_triggered": int(decision.alert_triggered),
                        "decision_reason": decision.reason,
                        "lead_time_minutes": lead_time_minutes,
                    }
                )
    return records


def run_replay(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    thresholds = _parse_thresholds(args.thresholds)

    per_horizon_args = {
        3: (args.model_h3, args.dataset_h3),
        10: (args.model_h10, args.dataset_h10),
        30: (args.model_h30, args.dataset_h30),
    }

    all_records: List[dict] = []
    for horizon, (model_path, dataset_path) in per_horizon_args.items():
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found for h={horizon}: {model_path}")
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found for h={horizon}: {dataset_path}")

        artifact = _load_artifact(model_path, device)
        dataset = DelayGraphDataset.load_joblib(dataset_path)

        meta_horizon = int(artifact.get("metadata", {}).get("horizon_minutes", horizon))
        if meta_horizon != horizon:
            raise ValueError(
                f"Model horizon mismatch: expected h={horizon}, artifact has h={meta_horizon}"
            )

        policy = PolicyEngine(
            thresholds={horizon: thresholds[horizon]},
            cooldown_minutes=args.cooldown_minutes,
            max_alerts_per_hour=args.max_alerts_per_hour,
        )

        model = _build_model(artifact, device)
        records = _run_single_horizon(
            horizon=horizon,
            dataset=dataset,
            model=model,
            device=device,
            policy=policy,
        )
        all_records.extend(records)

    frame = pd.DataFrame.from_records(all_records)
    if frame.empty:
        raise RuntimeError("Replay generated no records")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["timestamp", "horizon_minutes", "link"]).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)

    summary = {
        "records": int(len(frame)),
        "horizons": sorted(frame["horizon_minutes"].unique().tolist()),
        "links": int(frame["link"].nunique()),
        "alerts": int(frame["alert_triggered"].sum()),
        "violations": int(frame["actual_violation"].sum()),
        "thresholds": thresholds,
        "cooldown_minutes": int(args.cooldown_minutes),
        "max_alerts_per_hour": args.max_alerts_per_hour,
    }

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(f"Replay saved to {args.output_csv}")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop replay for SLA risk events (h=3,10,30).")

    parser.add_argument(
        "--model-h3",
        type=Path,
        default=Path("results_artifacts/best_model_output/sage_h3min.pt"),
    )
    parser.add_argument(
        "--model-h10",
        type=Path,
        default=Path("results_artifacts/best_model_output/sage_h10min.pt"),
    )
    parser.add_argument(
        "--model-h30",
        type=Path,
        default=Path("results_artifacts/best_model_output/sage_h30min.pt"),
    )

    parser.add_argument(
        "--dataset-h3",
        type=Path,
        default=Path("results_artifacts/datasets/gnn_dataset_w6_h3min_fixed.joblib"),
    )
    parser.add_argument(
        "--dataset-h10",
        type=Path,
        default=Path("results_artifacts/datasets/gnn_dataset_w6_h10min.joblib"),
    )
    parser.add_argument(
        "--dataset-h30",
        type=Path,
        default=Path("results_artifacts/datasets/gnn_dataset_w6_h30min.joblib"),
    )

    parser.add_argument(
        "--thresholds",
        type=str,
        default="3:0.55,10:0.45,30:0.35",
        help="Per-horizon risk thresholds. Format: 3:0.55,10:0.45,30:0.35",
    )
    parser.add_argument("--cooldown-minutes", type=int, default=10)
    parser.add_argument("--max-alerts-per-hour", type=int, default=None)

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results_artifacts/replay_events.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("results_artifacts/replay_summary.json"),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_replay(parse_args())
