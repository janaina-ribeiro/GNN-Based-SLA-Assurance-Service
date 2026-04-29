import argparse
import sys
from pathlib import Path

import torch
from .hyperparameter_optimizer import HyperparameterOptimizer
from optimization_analysis import OptimizationAnalyzer

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR))


def run_full_optimization(dataset_joblib: Path = None) -> None:
    print("=" * 60)
    print("FULL OPTIMIZATION")
    print("=" * 60)
    print("This optimization may take several hours.")
    print("Configure the parameters as needed.\n")

    data_dir = PROJECT_ROOT / "datasets_generated"
    output_dir = SCRIPT_DIR / "optimization_results"

    base_args = argparse.Namespace(
        data_dir=data_dir,
        links=None,
        window_size=6,
        horizon_minutes=30,
        min_corr=0.3,
        limit_samples=None,
        dataset_joblib=dataset_joblib,
        train_ratio=0.7,
        val_ratio=0.15,
        epochs=12,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        hidden_channels=64,
        num_layers=2,
        conv_type="gat",
        dropout=0.2,
        gat_heads=4,
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

    optimizer = HyperparameterOptimizer(
        base_args=base_args,
        n_trials=15,
        timeout=14400,
        study_name="full_gnn_optimization",
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

        analysis_dir = SCRIPT_DIR / "optimization_results" / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        history_path = analysis_dir / "optimization_history.png"
        params_path = analysis_dir / "parameter_distributions.png"

        analyzer.plot_optimization_history(history_path)
        analyzer.plot_parameter_distributions(params_path)

        report_path = analysis_dir / "analysis_report.json"
        analyzer.export_summary_report(report_path)

        print("\n" + "=" * 60)
        print("FULL OPTIMIZATION COMPLETED!")
        print("=" * 60)
        print(f"Best model: {best_model_path}")
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
            except Exception as e:
                print(f"Error saving model: {e}")

    except Exception as e:
        print(f"Error during optimization: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GNN hyperparameter optimization (full mode only)")
    parser.add_argument(
        "--dataset-joblib",
        type=Path,
        default=None,
        help="Path to pre-built dataset (.joblib). If provided, will skip dataset construction and use this instead."
    )
    args = parser.parse_args()
    
    if args.dataset_joblib:
        print(f"Using pre-built dataset: {args.dataset_joblib}")
    
    run_full_optimization(dataset_joblib=args.dataset_joblib)
