from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) <= 1:
        return 0.0
    return float(roc_auc_score(y_true, y_prob))


def _safe_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(brier_score_loss(y_true, y_prob))


def _metrics_for_group(df: pd.DataFrame, min_actionable_lead: int) -> Dict[str, float]:
    y_true = df["actual_violation"].to_numpy(dtype=int)
    y_prob = df["pred_risk"].to_numpy(dtype=float)
    y_alert = df["alert_triggered"].to_numpy(dtype=int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_alert,
        average="binary",
        zero_division=0,
    )

    tp = int(((y_alert == 1) & (y_true == 1)).sum())
    fp = int(((y_alert == 1) & (y_true == 0)).sum())
    fn = int(((y_alert == 0) & (y_true == 1)).sum())

    actionable_alerts = int(
        (
            (df["alert_triggered"] == 1)
            & (df["lead_time_minutes"].fillna(-1) >= min_actionable_lead)
        ).sum()
    )

    lead_times = df.loc[
        (df["alert_triggered"] == 1) & (df["actual_violation"] == 1),
        "lead_time_minutes",
    ].dropna()

    return {
        "records": int(len(df)),
        "violations": int(y_true.sum()),
        "alerts": int(y_alert.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_alert": float(precision),
        "recall_alert": float(recall),
        "f1_alert": float(f1),
        "auc_risk": _safe_auc(y_true, y_prob),
        "brier_risk": _safe_brier(y_true, y_prob),
        "actionable_alerts": actionable_alerts,
        "actionable_rate_over_alerts": float(actionable_alerts / max(int(y_alert.sum()), 1)),
        "lead_time_mean": float(lead_times.mean()) if not lead_times.empty else 0.0,
        "lead_time_p50": float(lead_times.quantile(0.5)) if not lead_times.empty else 0.0,
        "lead_time_p90": float(lead_times.quantile(0.9)) if not lead_times.empty else 0.0,
    }


def _calibration_table(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    work = df[["pred_risk", "actual_violation"]].copy()
    work["bin"] = pd.cut(
        work["pred_risk"],
        bins=np.linspace(0.0, 1.0, n_bins + 1),
        include_lowest=True,
    )
    grouped = work.groupby("bin", observed=False).agg(
        count=("actual_violation", "size"),
        mean_pred=("pred_risk", "mean"),
        obs_freq=("actual_violation", "mean"),
    )
    grouped = grouped.reset_index()
    grouped["calibration_error_abs"] = (grouped["mean_pred"] - grouped["obs_freq"]).abs()
    return grouped


def run(args: argparse.Namespace) -> None:
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Replay file not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    required = {
        "timestamp",
        "link",
        "horizon_minutes",
        "pred_risk",
        "actual_violation",
        "alert_triggered",
        "lead_time_minutes",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in replay CSV: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["horizon_minutes"] = df["horizon_minutes"].astype(int)

    rows: List[Dict[str, float]] = []

    overall = _metrics_for_group(df, min_actionable_lead=args.min_actionable_lead)
    overall["horizon_minutes"] = -1
    rows.append(overall)

    for horizon in sorted(df["horizon_minutes"].unique()):
        group = df[df["horizon_minutes"] == horizon]
        metrics = _metrics_for_group(group, min_actionable_lead=args.min_actionable_lead)
        metrics["horizon_minutes"] = int(horizon)
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)
    args.output_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.output_metrics_csv, index=False)

    calibration = _calibration_table(df, n_bins=args.calibration_bins)
    args.output_calibration_csv.parent.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(args.output_calibration_csv, index=False)

    summary = {
        "input_csv": str(args.input_csv),
        "records": int(len(df)),
        "horizons": sorted(df["horizon_minutes"].unique().tolist()),
        "links": int(df["link"].nunique()),
        "min_actionable_lead": int(args.min_actionable_lead),
        "metrics_overall": metrics_df[metrics_df["horizon_minutes"] == -1].to_dict(orient="records")[0],
        "metrics_by_horizon": metrics_df[metrics_df["horizon_minutes"] != -1].to_dict(orient="records"),
    }

    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Metrics CSV: {args.output_metrics_csv}")
    print(f"Calibration CSV: {args.output_calibration_csv}")
    print(f"Summary JSON: {args.output_summary_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate closed-loop replay outputs.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results_artifacts/replay_events.csv"),
    )
    parser.add_argument(
        "--output-metrics-csv",
        type=Path,
        default=Path("results_artifacts/experiment_a_metrics.csv"),
    )
    parser.add_argument(
        "--output-calibration-csv",
        type=Path,
        default=Path("results_artifacts/experiment_a_calibration.csv"),
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=Path("results_artifacts/experiment_a_summary.json"),
    )
    parser.add_argument("--min-actionable-lead", type=int, default=3)
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
