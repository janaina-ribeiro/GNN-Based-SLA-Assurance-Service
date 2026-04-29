

import argparse
from pathlib import Path
from datetime import datetime

import joblib
from .dataset_builder_lstm import load_and_process_data


def build_and_save_dataset(
    data_dir: Path,
    output_path: Path,
    window_size: int = 6,
    horizon_minutes: int = 30,
    links: list = None,
    limit_samples: int = None,
    delay_threshold: float = None,
    delay_percentile: float = 85.0,
    column_delay: str = "Atraso",
) -> None:
    
    """
    Build and cache LSTM/GRU dataset to avoid repeated preprocessing.
    ----------------------------------------------------------------

    This function pre-processes the dataset and saves it to a joblib file,
    which can be loaded directly during training or optimization, significantly
    reducing startup time.

    Saved artifact contains:
    - X: Feature array (n_samples, n_links, n_features)
    - y: Label array (n_samples, n_links)
    - links: List of link identifiers
    - timestamps: List of sample timestamps
    - Metadata: window_size, horizon_minutes, thresholds, etc.

    Args:
        data_dir: Path to directory with dataset CSV files
        output_path: Path for output joblib file
        window_size: Temporal window size for features
        horizon_minutes: Prediction horizon in minutes
        links: Optional list of specific links to process
        limit_samples: Optional limit on number of samples
        delay_threshold: Fixed delay threshold in ms (optional)
        delay_percentile: Percentile for dynamic threshold (default: 85.0)
        column_delay: Name of delay column (default: 'Atraso')

    Returns:
        None (saves dataset to output_path)
    """
    
    print("BUILDING LSTM/GRU DATASET CACHE")
    print(f"Data directory: {data_dir}")
    print(f"Window size: {window_size}")
    print(f"Horizon: {horizon_minutes} minutes")
    print(f"Delay threshold: {delay_threshold if delay_threshold else f'{delay_percentile}th percentile'}")
    print(f"Links: {'All' if links is None else links}")
    print(f"Sample limit: {'None (all data)' if limit_samples is None else limit_samples}")
    print(f"Output: {output_path}")
    print("\nThis may take several minutes...\n")
    
    start_time = datetime.now()
    
    X_np, y_np, links_list, timestamps = load_and_process_data(
        data_dir=data_dir,
        window_size=window_size,
        horizon_minutes=horizon_minutes,
        limit_samples=limit_samples,
        delay_threshold=delay_threshold,
        delay_percentile=delay_percentile,
        column_delay=column_delay,
        links=links,
    )
    
    dataset_artifact = {
        "X": X_np,
        "y": y_np,
        "links": links_list,
        "timestamps": timestamps,
        "window_size": window_size,
        "horizon_minutes": horizon_minutes,
        "delay_threshold": delay_threshold,
        "delay_percentile": delay_percentile,
        "column_delay": column_delay,
    }
    
    joblib.dump(dataset_artifact, output_path)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("LSTM/GRU DATASET CACHE BUILT SUCCESSFULLY!")
    print(f"Output file: {output_path}")
    print(f"File size: {output_path.stat().st_size / (1024**2):.2f} MB")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print(f"Number of samples: {X_np.shape[0]}")
    print(f"Number of links: {X_np.shape[1]}")
    print(f"Features per link: {X_np.shape[2]}")
    print("\nUsage with optimization:")
    print(f"  python -m test_gnn.run_optimization_lstm --dataset-joblib {output_path}")


def main():

    """
    Main entry point for building and caching LSTM/GRU datasets.
    ------------------------------------------------------------

    Parses command-line arguments and builds a pre-processed dataset
    that can be reused during training and optimization, avoiding
    repeated data loading and preprocessing overhead.

    Command-line arguments:
    - --data-dir: Directory with input CSV files
    - --output: Output path for cached dataset (.joblib)
    - --window-size: Temporal window size (default: 6)
    - --horizon-minutes: Prediction horizon (default: 30)
    - --delay-threshold: Fixed delay threshold in ms (optional)
    - --delay-percentile: Percentile for dynamic threshold (default: 85.0)
    - --column-delay: Delay column name (default: 'Atraso')
    - --links: Specific links to include (default: all)
    - --limit-samples: Sample limit (default: None)

    Returns:
        None (saves dataset to file)
    """
    
    parser = argparse.ArgumentParser(
        description="Pre-build dataset for LSTM/GRU model optimization"
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets_generated"),
        help="Directory containing CSV files (default: datasets_generated)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_lstm_cache.joblib"),
        help="Output path for cached dataset (default: dataset_lstm_cache.joblib)"
    )
    
    parser.add_argument(
        "--window-size",
        type=int,
        default=6,
        help="Window size for temporal aggregation (default: 6)"
    )
    
    parser.add_argument(
        "--horizon-minutes",
        type=int,
        default=30,
        help="Prediction horizon in minutes (default: 30)"
    )
    
    parser.add_argument(
        "--delay-threshold",
        type=float,
        default=None,
        help="Fixed delay threshold in ms (default: None, uses percentile)"
    )
    
    parser.add_argument(
        "--delay-percentile",
        type=float,
        default=85.0,
        help="Percentile for delay threshold (default: 85.0)"
    )
    
    parser.add_argument(
        "--column-delay",
        type=str,
        default="Atraso",
        help="Column name for delay measurements (default: Atraso)"
    )
    
    parser.add_argument(
        "--links",
        nargs="*",
        default=None,
        help="Specific links to include (default: all links)"
    )
    
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Limit number of samples (default: None, use all data)"
    )
    
    args = parser.parse_args()
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    build_and_save_dataset(
        data_dir=args.data_dir,
        output_path=args.output,
        window_size=args.window_size,
        horizon_minutes=args.horizon_minutes,
        delay_threshold=args.delay_threshold,
        delay_percentile=args.delay_percentile,
        column_delay=args.column_delay,
        links=args.links,
        limit_samples=args.limit_samples,
    )


if __name__ == "__main__":
    main()
