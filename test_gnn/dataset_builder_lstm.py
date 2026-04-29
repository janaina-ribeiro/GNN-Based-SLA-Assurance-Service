import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import logging
from datetime import datetime
import joblib

SCRIPT_DIR = Path(__file__).parent
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_logger():

    """
    Setup logger with file and console handlers for dataset building.
    -------------------------------------------------------------

    Configuration:
    - File handler: DEBUG level with timestamp and full context
    - Console handler: INFO level with simplified format
    - Log files stored in: logs/dataset_builder_lstm_<timestamp>.log

    Returns:
        Logger instance configured for both file and console output
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"dataset_builder_lstm_{timestamp}.log"
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

def infer_step_minutes(timestamps: pd.Series) -> float:

    """
    Infer the sampling frequency (step) from timestamps.
    ---------------------------------------------------

    Strategy:
    - Computes differences between consecutive timestamps
    - Returns the median difference in minutes
    - Fallback: 5 minutes if timestamps are empty or uniform
    - Ensures minimum value of 1 minute

    Args:
        timestamps: Series of pandas Timestamps

    Returns:
        Inferred step size in minutes (float)
    """

    if timestamps.empty:
        return 5.0
    ordered = timestamps.sort_values().drop_duplicates()
    deltas = ordered.diff().dropna().dt.total_seconds() / 60.0
    if deltas.empty:
        return 5.0
    return max(float(np.median(deltas)), 1.0)

def extract_path_features(df: pd.DataFrame) -> Dict[str, float]:

    """
    Extract traceroute path features from the DataFrame.
    ---------------------------------------------------

    Extracted features:
    - avg_hops: average number of hops across all measurements
    - min_hops: minimum number of hops observed
    - max_hops: maximum number of hops observed
    - std_hops: standard deviation of the number of hops

    Args:
        df: DataFrame with traceroute data (must contain 'Num_Hops' column)

    Returns:
        Dict with the 4 traceroute features (returns zeros if data is invalid)
    """

    if df.empty or "Num_Hops" not in df.columns:
        return {"avg_hops": 0.0, "min_hops": 0.0, "max_hops": 0.0, "std_hops": 0.0}
    hops = df["Num_Hops"].fillna(0).astype("float32")
    if hops.empty or hops.isna().all():
        return {"avg_hops": 0.0, "min_hops": 0.0, "max_hops": 0.0, "std_hops": 0.0}
    avg_hops = float(hops.mean())
    min_hops = float(hops.min())
    max_hops = float(hops.max())
    std_hops = float(hops.std()) if len(hops) > 1 else 0.0
    return {
        "avg_hops": avg_hops,
        "min_hops": min_hops,
        "max_hops": max_hops,
        "std_hops": std_hops,
    }

def resample_frame(df: pd.DataFrame, freq_minutes: int, delay_threshold: Optional[float] = None, column_delay: str = "Atraso") -> pd.DataFrame:
    
    """
    Resample DataFrame to uniform frequency and interpolate missing values.
    ----------------------------------------------------------------------

    Processing steps:
    - Convert timestamps to datetime and sort temporally
    - Resample to specified frequency using mean aggregation
    - Interpolate delay and hops columns (linear + forward/backward fill)
    - Optionally create binary 'high_delay' target based on threshold
    - Drop rows with all-NaN delay values

    Args:
        df: Input DataFrame with 'Timestamp' and delay columns
        freq_minutes: Target resampling frequency in minutes
        delay_threshold: Optional threshold for binary classification (ms)
        column_delay: Name of the delay column (default: 'Atraso')

    Returns:
        Resampled DataFrame with uniform time intervals and interpolated values
    """

    frame = df.copy()
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], utc=False)
    frame = frame.sort_values("Timestamp")
    frame = frame.set_index("Timestamp")
    rule = f"{int(freq_minutes)}min"
    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    agg = frame[numeric_cols].resample(rule).mean()
    if column_delay in agg.columns:
        agg[column_delay] = agg[column_delay].interpolate(method="linear", limit_direction="both")
        agg[column_delay] = agg[column_delay].ffill().bfill()
    if "Num_Hops" in agg.columns:
        agg["Num_Hops"] = agg["Num_Hops"].interpolate(method="linear", limit_direction="both")
        agg["Num_Hops"] = agg["Num_Hops"].ffill().bfill()
    if delay_threshold is not None and column_delay in agg.columns:
        agg["high_delay"] = (agg[column_delay] > delay_threshold).astype(int)
    agg = agg.dropna(subset=[column_delay], how="all")
    agg = agg.reset_index()
    return agg

def build_feature_vector(delay_window: pd.Series, hops_window: Optional[pd.Series] = None, global_topo_features: Optional[Dict[str, float]] = None) -> np.ndarray:
    
    """
    Build feature vector from delay window and topology information.
    ----------------------------------------------------------------

    Delay features (always included):
    - mean_delay: average delay in the window
    - std_delay: standard deviation of delay
    - max_delay: maximum delay observed
    - min_delay: minimum delay observed
    - last_delay: most recent delay value

    Topology features (optional, 4 features):
    - avg_hops, min_hops, max_hops, std_hops
    - Uses hops_window if provided, otherwise global_topo_features

    Args:
        delay_window: Series of delay measurements in the temporal window
        hops_window: Optional series of hop counts in the same window
        global_topo_features: Optional dict with global topology statistics

    Returns:
        Feature vector as numpy array (5 or 9 features depending on topology data)
    """

    values = delay_window.to_numpy(dtype=np.float32)
    mean_delay = float(np.nanmean(values))
    std_delay = float(np.nanstd(values)) if len(values) > 1 else 0.0
    max_delay = float(np.nanmax(values))
    min_delay = float(np.nanmin(values))
    last_delay = float(values[-1]) if not np.isnan(values[-1]) else mean_delay
    delay_features = np.array([mean_delay, std_delay, max_delay, min_delay, last_delay], dtype=np.float32)
    if hops_window is not None and len(hops_window) > 0:
        hops_values = hops_window.to_numpy(dtype=np.float32)
        hops_values = hops_values[~np.isnan(hops_values)]
        if len(hops_values) > 0:
            avg_hops = float(np.mean(hops_values))
            min_hops = float(np.min(hops_values))
            max_hops = float(np.max(hops_values))
            std_hops = float(np.std(hops_values)) if len(hops_values) > 1 else 0.0
        else:
            avg_hops = min_hops = max_hops = std_hops = 0.0
        topo_array = np.array([avg_hops, min_hops, max_hops, std_hops], dtype=np.float32)
        feat = np.concatenate([delay_features, topo_array])
    elif global_topo_features is not None:
        topo_array = np.array([
            global_topo_features.get("avg_hops", 0.0),
            global_topo_features.get("min_hops", 0.0),
            global_topo_features.get("max_hops", 0.0),
            global_topo_features.get("std_hops", 0.0),
        ], dtype=np.float32)
        feat = np.concatenate([delay_features, topo_array])
    else:
        feat = delay_features
    return feat.astype(np.float32)

def load_and_process_data(
    data_dir: Path | str,
    window_size: int = 3,
    horizon_minutes: int = 15,
    limit_samples: Optional[int] = None,
    delay_threshold: Optional[float] = None,
    delay_percentile: float = 85.0,
    column_delay: str = "Atraso",
    links: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[pd.Timestamp]]:
    
    """
    Load and process multi-link delay data into windowed samples for LSTM/GRU.
    --------------------------------------------------------------------------

    Processing pipeline:
    1. Load CSV files for each link (delay + traceroute data)
    2. Extract global topology features per link
    3. Infer sampling frequency and resample to uniform intervals
    4. Determine delay threshold (percentile or fixed value)
    5. Merge all links into single temporal DataFrame
    6. Create sliding windows with feature vectors per link
    7. Generate binary labels based on future delay threshold

    Output format:
    - X: (n_samples, n_links, n_features) - feature tensor
    - y: (n_samples, n_links) - binary labels (0: normal, 1: high delay)
    - links: List of link identifiers
    - timestamps: List of target timestamps for each sample

    Args:
        data_dir: Directory containing dataset_*_links_hops.csv files
        window_size: Number of past timesteps for feature extraction
        horizon_minutes: Minutes ahead for label prediction
        limit_samples: Optional limit on number of samples to generate
        delay_threshold: Fixed threshold in ms (overrides percentile)
        delay_percentile: Percentile for dynamic threshold (default: 85.0)
        column_delay: Name of delay column in CSVs (default: 'Atraso')
        links: Optional subset of links to process (default: all)

    Returns:
        Tuple of (feature_array, label_array, link_names, timestamps)
    """

    root = Path(data_dir)

 
    if links is None:
        link_files = sorted(root.glob("dataset_*_links_hops.csv"))
        links = [f.name[len("dataset_") : -len("_links_hops.csv")] for f in link_files]
    else:
        links = list(links)
        link_files = []
        for link in links:
            file_path = root / f"dataset_{link}_links_hops.csv"
            if not file_path.exists():
                raise FileNotFoundError(f"Missing file for link {link}: {file_path}")
            link_files.append(file_path)

    logger.info(f"Found {len(links)} links: {links}")
    frames = {}
    for link, file_path in zip(links, link_files):
        df = pd.read_csv(file_path, parse_dates=["Timestamp"])
        if column_delay not in df.columns:
            raise ValueError(f"Column '{column_delay}' not found in {file_path}")
        if "Num_Hops" not in df.columns and "Total_Hops" not in df.columns:
            raise ValueError(f"Column 'Num_Hops' or 'Total_Hops' not found in {file_path}")
        if "Total_Hops" in df.columns and "Num_Hops" not in df.columns:
            df = df.rename(columns={"Total_Hops": "Num_Hops"})
        if "Path_IPs" not in df.columns:
            raise ValueError(f"Column 'Path_IPs' not found in {file_path}")
        required_cols = ["Timestamp", column_delay, "Num_Hops", "Path_IPs"]
        df = df[[col for col in required_cols if col in df.columns]]
        df = df.dropna(subset=["Timestamp", column_delay])
        frames[link] = df
    topo_features = {link: extract_path_features(df) for link, df in frames.items()}
    freq_minutes = int(round(np.median([infer_step_minutes(f["Timestamp"]) for f in frames.values()]))) or 5
    logger.info(f"Detected frequency: {freq_minutes} minutes")
    if delay_threshold is None:
        all_delays = pd.concat([df[column_delay] for df in frames.values()], ignore_index=True)
        delay_threshold = float(all_delays.quantile(delay_percentile / 100.0))
        logger.info(f"Delay threshold (percentile {delay_percentile}%): {delay_threshold:.2f} ms")
    else:
        logger.info(f"Using fixed delay threshold: {delay_threshold:.2f} ms")
    resampled = {link: resample_frame(df, freq_minutes, delay_threshold, column_delay) for link, df in frames.items()}
    delay_frames = []
    hops_frames = []
    for link, frame in resampled.items():
        clean_frame = frame[["Timestamp", column_delay]].copy()
        ren = clean_frame.rename(columns={column_delay: f"delay_{link}"})
        ren = ren.set_index("Timestamp")
        delay_frames.append(ren)
        if "Num_Hops" in frame.columns:
            hops_frame = frame[["Timestamp", "Num_Hops"]].copy()
            hops_ren = hops_frame.rename(columns={"Num_Hops": f"hops_{link}"})
            hops_ren = hops_ren.set_index("Timestamp")
            hops_frames.append(hops_ren)
    merged = pd.concat(delay_frames, axis=1, join="outer")
    if hops_frames:
        hops_merged = pd.concat(hops_frames, axis=1, join="outer")
        merged = pd.concat([merged, hops_merged], axis=1)
    merged = merged.sort_index()
    merged = merged.ffill().bfill()
    merged = merged.interpolate(method="time", limit_direction="both")
    target_data = {}
    for link in links:
        delay_col = f"delay_{link}"
        target_data[f"target_{link}"] = (merged[delay_col] > delay_threshold).astype(int)
    target_df = pd.DataFrame(target_data, index=merged.index)
    merged = pd.concat([merged, target_df], axis=1)
    merged = merged.fillna(0)
    merged = merged.reset_index()
    delay_cols = [f"delay_{link}" for link in links]
    hops_cols = [f"hops_{link}" for link in links]
    target_cols = [f"target_{link}" for link in links]
    available_hops_cols = {col: col in merged.columns for col in hops_cols}
    features = []
    labels = []
    stamps = []
    total_rows = len(merged)
    min_required = window_size + int(round(horizon_minutes / freq_minutes)) + 1
    for idx in range(window_size, total_rows - int(round(horizon_minutes / freq_minutes))):
        window_start = idx - window_size
        window_end = idx
        target_pos = idx + int(round(horizon_minutes / freq_minutes)) - 1
        if target_pos >= total_rows:
            break
        window_data = merged.iloc[window_start:window_end]
        target_row = merged.iloc[target_pos]
        node_feat_list = []
        for link_idx, delay_col in enumerate(delay_cols):
            hops_col = hops_cols[link_idx]
            delay_window = pd.Series(window_data[delay_col].values)
            hops_window = None
            if available_hops_cols.get(hops_col, False):
                hops_window = pd.Series(window_data[hops_col].values)
            global_topo = topo_features.get(links[link_idx])
            feat_vector = build_feature_vector(delay_window, hops_window, global_topo)
            node_feat_list.append(feat_vector)
        node_feat = np.stack(node_feat_list)
        node_label = target_row[target_cols].to_numpy(dtype=np.int64)
        features.append(node_feat)
        labels.append(node_label)
        stamps.append(pd.Timestamp(target_row["Timestamp"]))
        if limit_samples and len(features) >= limit_samples:
            break
    feature_array = np.stack(features)
    label_array = np.stack(labels)
    return feature_array, label_array, links, stamps

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Process network delay datasets into windowed samples for LSTM/GRU training.",
        epilog="""
IMPORTANT: For fair comparison with GNN models, ensure you use the same parameters:
  - column_delay: Use 'Atraso' (same as GNN's dataset_builder.py)
  - delay_percentile: Use 85.0 (same as GNN default)
  - window_size: Use 6 (same as GNN's train_params_best.py)
  
Example for generating all horizons:
  python dataset_builder_lstm.py --data_dir datasets_generated --generate_all_horizons --column_delay Atraso
        """
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing dataset_*_links_hops.csv files")
    parser.add_argument("--window_size", type=int, default=6, help="Sliding window size (default: 6, same as GNN)")
    parser.add_argument("--horizon_minutes", type=int, default=15, help="Prediction horizon in minutes")
    parser.add_argument("--limit_samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--delay_threshold", type=float, default=None, help="Fixed delay threshold (ms)")
    parser.add_argument("--delay_percentile", type=float, default=85.0, help="Percentile for delay threshold (default: 85.0, same as GNN)")
    parser.add_argument("--column_delay", type=str, default="Atraso", help="Name of delay column (default: 'Atraso', same as GNN)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for samples (if not set, uses data_dir)")
    parser.add_argument("--generate_all_horizons", action="store_true", help="Generate datasets for all horizons (1, 3, 5, 10, 30 min)")
    args = parser.parse_args()
    
    horizons = [1, 3, 5, 10, 30] if args.generate_all_horizons else [args.horizon_minutes]
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for horizon in horizons:
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating dataset for horizon {horizon} minutes")
        logger.info(f"{'='*60}")
        
        X, y, links, stamps = load_and_process_data(
            args.data_dir,
            window_size=args.window_size,
            horizon_minutes=horizon,
            limit_samples=args.limit_samples,
            delay_threshold=args.delay_threshold,
            delay_percentile=args.delay_percentile,
            column_delay=args.column_delay,
        )
        
        joblib_path = output_dir / f"lstm_dataset_w{args.window_size}_h{horizon}min.joblib"
        dataset_artifact = {
            "X": X,
            "y": y,
            "links": links,
            "timestamps": stamps,
            "metadata": {
                "window_size": args.window_size,
                "horizon_minutes": horizon,
                "delay_percentile": args.delay_percentile,
                "column_delay": args.column_delay,
                "num_features": X.shape[2],
                "num_links": X.shape[1],
                "num_samples": X.shape[0],
            }
        }
        joblib.dump(dataset_artifact, joblib_path)
        logger.info(f"Saved joblib to {joblib_path}")
        
        npz_file = output_dir / f"windowed_samples_w{args.window_size}_h{horizon}_p{int(args.delay_percentile)}.npz"
        np.savez_compressed(npz_file, X=X, y=y, links=links, timestamps=[str(ts) for ts in stamps])
        logger.info(f"Saved npz to {npz_file}")
        logger.info(f"Shape - X: {X.shape}, y: {y.shape}, links: {len(links)}, timestamps: {len(stamps)}")
    
    logger.info("All datasets generated successfully!")
    logger.info(f"Output directory: {output_dir}")
