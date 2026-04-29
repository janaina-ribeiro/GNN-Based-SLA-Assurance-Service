import argparse
import sys
from pathlib import Path

import torch
from .hyperparameter_optimizer_lstm import HyperparameterOptimizerLSTM
from optimization_analysis import OptimizationAnalyzer

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR))


def run_full_optimization(dataset_joblib: Path = None) -> None:
    """
    Run full hyperparameter optimization for LSTM/GRU models using Optuna.
    ----------------------------------------------------------------------

    Optimization process:
    1. Configure base parameters and search space
    2. Run Optuna trials with TPE sampler and Median pruner
    3. Find best hyperparameters based on validation F1-macro
    4. Train final model with best parameters
    5. Generate analysis plots and reports

    Optimized hyperparameters:
    - Architecture: hidden_size, num_layers, rnn_type (LSTM/GRU)
    - Regularization: dropout, weight_decay, label_smoothing
    - Training: lr, batch_size, scheduler type and parameters
    - Sampling: sample_weight_alpha for imbalanced data
    - Gradient: max_grad_norm for clipping

    Args:
        dataset_joblib: Optional path to pre-built dataset cache (recommended)
                       If None, datasets will be built from scratch (slower)

    Returns:
        None (saves optimized model and analysis to optimization_results_lstm/)
    """
    print("=" * 60)
    print("LSTM/GRU OPTIMIZATION")
    print("=" * 60)
    
    if dataset_joblib is not None:
        print(f" Using pre-built dataset: {dataset_joblib}")
        print("This will SKIP dataset construction and use cached data.")
    else:
        print(" No dataset cache provided. Will construct dataset from scratch (SLOW).")
        print("Consider building a dataset first: python -m test_gnn.build_dataset_lstm")
    
    print("This optimization may take several hours.")
    print("Configure the parameters as needed.\n")

    data_dir = PROJECT_ROOT / "datasets_generated"
    output_dir = SCRIPT_DIR / "optimization_results_lstm"

    base_args = argparse.Namespace(
        data_dir=data_dir,
        links=None,
        window_size=6,
        horizon_minutes=30,
        limit_samples=None,
        dataset_joblib=dataset_joblib,
        train_ratio=0.7,
        val_ratio=0.15,
        delay_threshold=None,
        delay_percentile=85.0,
        column_delay="Atraso",
        epochs=12,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        hidden_size=64,
        num_layers=2,
        rnn_type="lstm",
        dropout=0.2,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=16,
        label_smoothing=0.1,
        sample_weight_alpha=2.0,
        max_grad_norm=1.0,
        scheduler="plateau",
        scheduler_factor=0.5,
        scheduler_patience=5,
        scheduler_t0=10,
        patience=15,
        model_path=output_dir / "temp_model.pt",
    )

    optimizer = HyperparameterOptimizerLSTM(
        base_args=base_args,
        n_trials=15,
        timeout=14400,
        study_name="full_lstm_gru_optimization",
        metric="val_f1_macro",
        pruner_patience=5,
    )

    try:
        best_params, best_value = optimizer.optimize()
        study_path = optimizer.save_study()
        best_model_path = optimizer.train_best_model()

        print("\nAnalyzing results...")
        analyzer = OptimizationAnalyzer(study_path)
        analyzer.print_summary()
        analyzer.analyze_top_trials(10)

        analysis_dir = SCRIPT_DIR / "optimization_results_lstm" / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        history_path = analysis_dir / "optimization_history.png"
        params_path = analysis_dir / "parameter_distributions.png"

        analyzer.plot_optimization_history(history_path)
        analyzer.plot_parameter_distributions(params_path)

        report_path = analysis_dir / "analysis_report.json"
        analyzer.export_summary_report(report_path)

        print("\n" + "=" * 60)
        print("LSTM/GRU OPTIMIZATION COMPLETED!")
        print("=" * 60)
        print(f"Best model: {best_model_path}")
        
        history_json_path = best_model_path.with_name(best_model_path.stem + "_history.json")
        if history_json_path.exists():
            print(f"Training history: {history_json_path}")
        
        print(f"Study saved: {study_path}")
        print(f"Report: {report_path}")
        print(f"Plots: {analysis_dir}")

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        print("Saving progress...")
        optimizer.save_study()

        if len(optimizer.study.trials) > 0:
            print("Training the best model found so far...")
            try:
                best_model_path = optimizer.train_best_model()
                print(f"Best model saved at: {best_model_path}")
                
                history_json_path = best_model_path.with_name(best_model_path.stem + "_history.json")
                if history_json_path.exists():
                    print(f"Training history: {history_json_path}")
            except Exception as e:
                print(f"Error saving model: {e}")

    except Exception as e:
        print(f"Error during optimization: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LSTM/GRU hyperparameter optimization (full mode only)")
    parser.add_argument(
        "--dataset-joblib",
        type=Path,
        default=None,
        help="Path to pre-built LSTM dataset (.joblib). If provided, will skip dataset construction and use this instead."
    )
    args = parser.parse_args()
    
    if args.dataset_joblib:
        print(f"Using pre-built dataset: {args.dataset_joblib}")
    
    run_full_optimization(dataset_joblib=args.dataset_joblib)
