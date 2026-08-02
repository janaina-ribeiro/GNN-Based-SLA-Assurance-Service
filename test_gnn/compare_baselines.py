from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


@dataclass
class Counts:
    records: int = 0
    violations: int = 0
    alerts: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    actionable_alerts: int = 0


@dataclass
class MetricsRow:
    baseline: str
    horizon_minutes: int
    records: int
    violations: int
    alerts: int
    tp: int
    fp: int
    fn: int
    precision_alert: float
    recall_alert: float
    f1_alert: float
    alert_rate: float
    actionable_alerts: int
    actionable_rate_over_alerts: float


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def _counts_to_metrics(baseline: str, horizon: int, counts: Counts) -> MetricsRow:
    precision = counts.tp / counts.alerts if counts.alerts > 0 else 0.0
    recall = counts.tp / counts.violations if counts.violations > 0 else 0.0
    f1 = _f1(precision, recall)
    alert_rate = counts.alerts / counts.records if counts.records > 0 else 0.0
    actionable_rate = (
        counts.actionable_alerts / counts.alerts if counts.alerts > 0 else 0.0
    )

    return MetricsRow(
        baseline=baseline,
        horizon_minutes=horizon,
        records=counts.records,
        violations=counts.violations,
        alerts=counts.alerts,
        tp=counts.tp,
        fp=counts.fp,
        fn=counts.fn,
        precision_alert=precision,
        recall_alert=recall,
        f1_alert=f1,
        alert_rate=alert_rate,
        actionable_alerts=counts.actionable_alerts,
        actionable_rate_over_alerts=actionable_rate,
    )


def _update_counts(
    counts: Counts,
    y_true: pd.Series,
    y_alert: pd.Series,
    actionable_mask: pd.Series,
) -> None:
    y_true_int = y_true.astype(int)
    y_alert_int = y_alert.astype(int)

    counts.records += int(len(y_true_int))
    counts.violations += int(y_true_int.sum())
    counts.alerts += int(y_alert_int.sum())

    tp_mask = (y_alert_int == 1) & (y_true_int == 1)
    fp_mask = (y_alert_int == 1) & (y_true_int == 0)
    fn_mask = (y_alert_int == 0) & (y_true_int == 1)

    counts.tp += int(tp_mask.sum())
    counts.fp += int(fp_mask.sum())
    counts.fn += int(fn_mask.sum())
    counts.actionable_alerts += int(actionable_mask.sum())


def _init_bucket() -> Dict[str, Counts]:
    return {
        "gnn_proactive": Counts(),
        "no_action": Counts(),
        "reactive": Counts(),
    }


def _iter_chunks(path: Path, chunk_size: int) -> Iterable[pd.DataFrame]:
    usecols = [
        "horizon_minutes",
        "actual_violation",
        "alert_triggered",
        "lead_time_minutes",
    ]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size):
        yield chunk


def run(args: argparse.Namespace) -> None:
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Replay events file not found: {args.input_csv}")

    horizons = [3, 10, 30]
    by_horizon: Dict[int, Dict[str, Counts]] = {h: _init_bucket() for h in horizons}
    overall: Dict[str, Counts] = _init_bucket()

    for chunk in _iter_chunks(args.input_csv, args.chunk_size):
        chunk["horizon_minutes"] = chunk["horizon_minutes"].astype(int)
        chunk["actual_violation"] = chunk["actual_violation"].astype(int)
        chunk["alert_triggered"] = chunk["alert_triggered"].astype(int)

        for h in horizons:
            view = chunk[chunk["horizon_minutes"] == h]
            if view.empty:
                continue

            y_true = view["actual_violation"]

            y_alert_gnn = view["alert_triggered"]
            actionable_gnn = (
                (view["alert_triggered"] == 1)
                & (view["lead_time_minutes"].fillna(-1) >= args.min_actionable_lead)
            )
            _update_counts(by_horizon[h]["gnn_proactive"], y_true, y_alert_gnn, actionable_gnn)
            _update_counts(overall["gnn_proactive"], y_true, y_alert_gnn, actionable_gnn)

            y_alert_no = pd.Series(0, index=view.index)
            actionable_no = pd.Series(False, index=view.index)
            _update_counts(by_horizon[h]["no_action"], y_true, y_alert_no, actionable_no)
            _update_counts(overall["no_action"], y_true, y_alert_no, actionable_no)

            y_alert_reactive = y_true
            if args.min_actionable_lead <= 0:
                actionable_reactive = y_alert_reactive == 1
            else:
                actionable_reactive = pd.Series(False, index=view.index)
            _update_counts(
                by_horizon[h]["reactive"], y_true, y_alert_reactive, actionable_reactive
            )
            _update_counts(overall["reactive"], y_true, y_alert_reactive, actionable_reactive)

    rows: List[MetricsRow] = []
    for baseline_name, counts in overall.items():
        rows.append(_counts_to_metrics(baseline_name, -1, counts))

    for h in horizons:
        for baseline_name, counts in by_horizon[h].items():
            rows.append(_counts_to_metrics(baseline_name, h, counts))

    df = pd.DataFrame([asdict(r) for r in rows])
    df = df.sort_values(["horizon_minutes", "baseline"]).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    summary = {
        "input_csv": str(args.input_csv),
        "min_actionable_lead": int(args.min_actionable_lead),
        "chunk_size": int(args.chunk_size),
        "baselines": ["gnn_proactive", "no_action", "reactive"],
        "horizons": horizons,
        "overall": df[df["horizon_minutes"] == -1].to_dict(orient="records"),
        "by_horizon": {
            str(h): df[df["horizon_minutes"] == h].to_dict(orient="records")
            for h in horizons
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Baseline comparison CSV: {args.output_csv}")
    print(f"Baseline comparison JSON: {args.output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare operational baselines using replay event logs."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results_artifacts/replay_events.csv"),
        help="Replay event log generated by replay_closed_loop.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results_artifacts/baseline_comparison.csv"),
        help="Output table with baseline metrics.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results_artifacts/baseline_comparison.json"),
        help="Output summary JSON.",
    )
    parser.add_argument(
        "--min-actionable-lead",
        type=int,
        default=3,
        help="Minimum lead-time to count an alert as actionable.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500000,
        help="Chunk size for processing large replay CSV files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
