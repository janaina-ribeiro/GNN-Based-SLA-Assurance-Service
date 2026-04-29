from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import Trial
from train_lstm_gru import train_model as train_single_model


class HyperparameterOptimizerLSTM:
    """
    Hyperparameter optimizer using Optuna for the DelayRNN (LSTM/GRU) model.
    -------------------------------------------------------------------------
    Supports single-objective and multi-objective optimization.

    Available metrics:
    - val_f1_macro: Macro F1-score (recommended for imbalance)
    - val_balanced_accuracy: Balanced accuracy
    - val_brier_score: Brier score (minimize)
    - test_f1_macro: Macro F1-score on test
    - test_precision: Precision on test
    - test_recall: Recall on test
    - test_auc: ROC AUC score
    """

    def __init__(
        self,
        base_args: argparse.Namespace,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
        direction: str = "maximize",
        metric: str = "val_f1_macro",
        pruner_patience: int = 5,
        sampler_seed: int = 42,
    ):
        self.base_args = base_args
        self.n_trials = n_trials
        self.timeout = timeout
        self.direction = direction
        self.metric = metric

        if study_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            study_name = f"lstm_gru_optimization_{timestamp}"

        self.study_name = study_name

        sampler = TPESampler(seed=sampler_seed, n_startup_trials=10)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=pruner_patience)

        self.study = optuna.create_study(
            study_name=study_name, direction=direction, sampler=sampler, pruner=pruner
        )

        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """
        Configure logging for optimization process.
        -------------------------------------------

        Setup:
        - File handler: Saves all optimization logs to file
        - Console handler: Displays progress to user
        - Log directory: optimization_logs/ under model output directory

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"optuna_{self.study_name}")
        logger.setLevel(logging.INFO)

        if logger.handlers:
            return logger

        log_dir = self.base_args.model_path.parent / "optimization_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{self.study_name}.log"

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def suggest_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for LSTM/GRU model optimization.
        ---------------------------------------------------------

        Hyperparameter search space:
        - window_size: [4, 6, 8, 10, 12] (if not using pre-built dataset)
        - hidden_size: [32, 64, 128, 256]
        - num_layers: [1, 2, 3]
        - rnn_type: ['lstm', 'gru']
        - dropout: [0.1, 0.5] (step 0.1)
        - lr: [1e-4, 1e-2] (log scale)
        - weight_decay: [1e-6, 1e-3] (log scale)
        - batch_size: [8, 16, 32, 64]
        - label_smoothing: [0.0, 0.2] (step 0.05)
        - sample_weight_alpha: [1.0, 5.0] (step 0.5)
        - max_grad_norm: [0.5, 1.0, 2.0, 5.0]
        - scheduler: ['none', 'plateau', 'cosine']
        - scheduler_factor: [0.3, 0.7] (for plateau)
        - scheduler_patience: [3, 8] (for plateau)
        - scheduler_t0: [5, 15] (for cosine)

        Args:
            trial: Optuna trial object for suggesting hyperparameters

        Returns:
            Dict with suggested hyperparameters
        """
        
        # Se dataset pré-construído, usa window_size fixo; caso contrário, otimiza
        if getattr(self.base_args, "dataset_joblib", None) is not None:
            window_size = self.base_args.window_size
        else:
            window_size = trial.suggest_categorical("window_size", [4, 6, 8, 10, 12])

        # Architecture parameters (different from GNN)
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        rnn_type = trial.suggest_categorical("rnn_type", ["lstm", "gru"])

        # Regularization
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)

        # Optimization parameters
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])

        # Advanced techniques (same as GNN)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2, step=0.05)
        sample_weight_alpha = trial.suggest_float("sample_weight_alpha", 1.0, 5.0, step=0.5)
        max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.5, 1.0, 2.0, 5.0])

        # Scheduler configuration
        scheduler = trial.suggest_categorical("scheduler", ["none", "plateau", "cosine"])
        
        scheduler_factor = 0.5
        scheduler_patience = 5
        scheduler_t0 = 10

        if scheduler == "plateau":
            scheduler_factor = trial.suggest_float("scheduler_factor", 0.3, 0.7, step=0.1)
            scheduler_patience = trial.suggest_int("scheduler_patience", 3, 8)
        elif scheduler == "cosine":
            scheduler_t0 = trial.suggest_int("scheduler_t0", 5, 15)

        return {
            "window_size": window_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "rnn_type": rnn_type,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "label_smoothing": label_smoothing,
            "sample_weight_alpha": sample_weight_alpha,
            "max_grad_norm": max_grad_norm,
            "scheduler": scheduler,
            "scheduler_factor": scheduler_factor,
            "scheduler_patience": scheduler_patience,
            "scheduler_t0": scheduler_t0,
        }

    def objective(self, trial: Trial) -> float:
        """
        Objective function for Optuna optimization.
        -------------------------------------------

        Execution flow:
        1. Suggest hyperparameters using trial object
        2. Configure trial arguments with suggested parameters
        3. Train model with reduced epochs (max 20) for efficiency
        4. Enable early stopping (patience=10)
        5. Extract objective metric from training results
        6. Clean up temporary model files

        Supported metrics:
        - val_f1_macro: Validation F1-macro (default)
        - val_balanced_accuracy: Validation balanced accuracy
        - test_f1_macro: Test F1-macro
        - test_balanced_accuracy: Test balanced accuracy
        - test_precision: Test precision
        - test_recall: Test recall
        - test_auc: Test ROC AUC
        - test_brier_score: Test Brier score (negated for minimization)

        Args:
            trial: Optuna trial object

        Returns:
            Objective value (metric score to maximize/minimize)
        """
        suggested_params = self.suggest_hyperparameters(trial)

        trial_args = argparse.Namespace(**vars(self.base_args))

        for key, value in suggested_params.items():
            setattr(trial_args, key, value)

        temp_dir = self.base_args.model_path.parent / "temp_models"
        temp_dir.mkdir(parents=True, exist_ok=True)
        trial_model_path = temp_dir / f"trial_{trial.number}_model.pt"

        trial_args.model_path = trial_model_path

        original_epochs = trial_args.epochs
        trial_args.epochs = min(20, original_epochs)

        trial_args.patience = 10

        try:
            train_logger = logging.getLogger("lstm_gru_training")
            original_level = train_logger.level
            train_logger.setLevel(logging.WARNING)

            self.logger.info(
                f"Trial {trial.number}: Testing parameters {suggested_params}"
            )

            train_single_model(trial_args)

            artifact = torch.load(trial_model_path, map_location="cpu")
            metadata = artifact["metadata"]

            if self.metric == "val_f1_macro":
                objective_value = metadata["best_val_f1"]
            elif self.metric == "val_balanced_accuracy":
                history = metadata.get("training_history", [])
                best_epoch = metadata.get("best_epoch", 0)
                if history and best_epoch > 0 and best_epoch <= len(history):
                    best_epoch_data = history[best_epoch - 1]
                    objective_value = best_epoch_data["val"].get("balanced_accuracy", 0.0)
                else:
                    objective_value = (
                        history[-1]["val"].get("balanced_accuracy", 0.0)
                        if history
                        else 0.0
                    )
            elif self.metric == "test_f1_macro":
                objective_value = metadata["test_metrics"]["f1_macro"]
            elif self.metric == "test_balanced_accuracy":
                objective_value = metadata["test_metrics"].get("balanced_accuracy", 0.0)
            elif self.metric == "test_precision":
                objective_value = metadata["test_metrics"]["precision"]
            elif self.metric == "test_recall":
                objective_value = metadata["test_metrics"]["recall"]
            elif self.metric == "test_auc":
                objective_value = metadata["test_metrics"].get("auc", 0.0)
            elif self.metric == "test_brier_score":
                objective_value = -metadata["test_metrics"].get("brier_score", 1.0)
            else:
                objective_value = metadata["best_val_f1"]

            test_metrics = metadata["test_metrics"]
            self.logger.info(
                f"Trial {trial.number} finished: {self.metric}={objective_value:.4f} | "
                f"Test F1={test_metrics['f1_macro']:.4f} | "
                f"Test BalAcc={test_metrics.get('balanced_accuracy', 0):.4f} | "
                f"Test Precision={test_metrics.get('precision', 0):.4f} | "
                f"Test Recall={test_metrics.get('recall', 0):.4f} | "
                f"Test AUC={test_metrics.get('auc', 0):.4f} | "
                f"Epoch={metadata.get('best_epoch', 0)}"
            )

            train_logger.setLevel(original_level)

            trial_model_path.unlink(missing_ok=True)

            return objective_value

        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {str(e)}")

            train_logger.setLevel(original_level)

            trial_model_path.unlink(missing_ok=True)

            return -1.0 if self.direction == "maximize" else 1000.0

    def optimize(self) -> Tuple[Dict[str, Any], float]:
        """
        Run the full hyperparameter optimization process.
        -------------------------------------------------

        Process:
        1. Log optimization configuration
        2. Run Optuna study with specified trials and timeout
        3. Track best parameters and value
        4. Log summary of optimization results

        Returns:
            Tuple of (best_parameters_dict, best_objective_value)
        """
        self.logger.info("=" * 60)
        self.logger.info(f"STARTING LSTM/GRU OPTIMIZATION: {self.study_name}")
        self.logger.info("=" * 60)
        self.logger.info(f"Number of trials: {self.n_trials}")
        self.logger.info(
            f"Timeout: {self.timeout}s" if self.timeout else "Timeout: None"
        )
        self.logger.info(f"Objective metric: {self.metric} ({self.direction})")
        self.logger.info(f"Dataset: {self.base_args.data_dir}")

        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )

        best_params = self.study.best_params
        best_value = self.study.best_value

        self.logger.info("=" * 60)
        self.logger.info("OPTIMIZATION COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Best value: {best_value:.4f}")
        self.logger.info(f"Number of completed trials: {len(self.study.trials)}")
        self.logger.info("Best parameters:")
        for key, value in best_params.items():
            self.logger.info(f"  {key}: {value}")

        return best_params, best_value

    def save_study(self, save_path: Optional[Path] = None) -> Path:
        """
        Save optimization study results to JSON file.
        ----------------------------------------------

        Saved information:
        - Study metadata (name, direction, metric)
        - Best trial details (value, parameters, state)
        - All trial results with parameters and outcomes
        - Timestamps and execution information

        Args:
            save_path: Optional custom path (default: optimization_results/<study_name>_study.json)

        Returns:
            Path where study was saved
        """
        if save_path is None:
            save_dir = self.base_args.model_path.parent / "optimization_results"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{self.study_name}_study.json"

        study_data = {
            "study_name": self.study_name,
            "direction": self.direction,
            "metric": self.metric,
            "n_trials_completed": len(self.study.trials),
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "best_trial": {
                "number": self.study.best_trial.number,
                "value": self.study.best_trial.value,
                "params": self.study.best_trial.params,
                "state": str(self.study.best_trial.state),
            },
            "all_trials": [
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": str(trial.state),
                    "datetime_start": trial.datetime_start.isoformat()
                    if trial.datetime_start
                    else None,
                    "datetime_complete": trial.datetime_complete.isoformat()
                    if trial.datetime_complete
                    else None,
                }
                for trial in self.study.trials
            ],
            "base_config": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in vars(self.base_args).items()
            },
            "optimization_timestamp": datetime.now().isoformat(),
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(study_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Study saved at: {save_path}")
        return save_path

    def train_best_model(self, save_path: Optional[Path] = None) -> Path:
        """
        Train final model using best hyperparameters from optimization.
        ---------------------------------------------------------------

        Process:
        1. Create new arguments namespace with best parameters
        2. Use full epoch count (not limited like in trials)
        3. Enable early stopping with patience=10
        4. Train and save final model

        Args:
            save_path: Optional custom path (default: optimized_models/best_lstm_gru_model_<timestamp>.pt)

        Returns:
            Path to saved best model
        """
        if save_path is None:
            save_dir = self.base_args.model_path.parent / "optimized_models"
            save_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = save_dir / f"best_lstm_gru_model_{timestamp}.pt"

        best_args = argparse.Namespace(**vars(self.base_args))

        for key, value in self.study.best_params.items():
            setattr(best_args, key, value)

        best_args.model_path = save_path

        best_args.patience = 10

        self.logger.info("=" * 60)
        self.logger.info("TRAINING FINAL LSTM/GRU MODEL WITH BEST PARAMETERS")
        self.logger.info("=" * 60)
        self.logger.info(f"Saving at: {save_path}")

        train_single_model(best_args)

        self.logger.info(f"Final model trained and saved at: {save_path}")
        return save_path


def parse_optimization_args() -> argparse.Namespace:
    """
    Parse command-line arguments for LSTM/GRU hyperparameter optimization.
    ----------------------------------------------------------------------

    Arguments:
    - Data configuration: data_dir, links, window_size, horizon_minutes
    - Dataset caching: dataset_joblib (optional pre-built dataset)
    - Training splits: train_ratio, val_ratio
    - Delay threshold: delay_threshold, delay_percentile, column_delay
    - Optimization: n_trials, timeout, metric, pruner_patience
    - Model defaults: hidden_size, num_layers, rnn_type, dropout
    - Training defaults: epochs, lr, weight_decay, batch_size
    - Output: model_path, device, seed

    Returns:
        Namespace with all parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for LSTM/GRU", parents=[], add_help=True
    )

    parser.add_argument("--data-dir", type=Path, default=Path("datasets_generated"))
    parser.add_argument("--links", nargs="*", default=None)
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--horizon-minutes", type=int, default=30)
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=2000,
        help="Sample limit for optimization (speeds up trials)",
    )

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)

    parser.add_argument(
        "--delay-threshold", 
        type=float, 
        default=None,
        help="Fixed delay threshold in ms (overrides percentile if set)"
    )
    parser.add_argument(
        "--delay-percentile", 
        type=float, 
        default=85.0,
        help="Percentile for delay threshold when no fixed threshold is given"
    )
    parser.add_argument("--column-delay", type=str, default="Atraso")

    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Epochs for final model (trials use fewer)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--n-trials", type=int, default=50, help="Number of trials for optimization"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds (None = no limit)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val_f1_macro",
        choices=[
            "val_f1_macro",
            "val_balanced_accuracy",
            "test_f1_macro",
            "test_balanced_accuracy",
            "test_precision",
            "test_recall",
            "test_auc",
            "test_brier_score",
        ],
        help="Metric to optimize",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Study name (None = auto-generated)",
    )
    parser.add_argument(
        "--pruner-patience", type=int, default=5, help="Pruner patience"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("optimization_results_lstm"),
        help="Directory to save optimization results",
    )

    return parser.parse_args()


def main():
    args = parse_optimization_args()

    base_args = argparse.Namespace(
        data_dir=args.data_dir,
        links=args.links,
        window_size=args.window_size,
        horizon_minutes=args.horizon_minutes,
        limit_samples=args.limit_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        delay_threshold=args.delay_threshold,
        delay_percentile=args.delay_percentile,
        column_delay=args.column_delay,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
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
        scheduler="none",
        scheduler_factor=0.5,
        scheduler_patience=5,
        scheduler_t0=10,
        patience=0,
        model_path=args.output_dir / "temp_model.pt",
        dataset_joblib=None,
    )

    optimizer = HyperparameterOptimizerLSTM(
        base_args=base_args,
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=args.study_name,
        metric=args.metric,
        pruner_patience=args.pruner_patience,
    )

    try:
        best_params, best_value = optimizer.optimize()

        study_path = optimizer.save_study()

        best_model_path = optimizer.train_best_model()

        print("=" * 60)
        print("OPTIMIZATION SUCCESSFULLY COMPLETED!")
        print("=" * 60)
        print(f"Best {args.metric}: {best_value:.4f}")
        print(f"Study saved at: {study_path}")
        print(f"Best model saved at: {best_model_path}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        print("Saving progress...")
        optimizer.save_study()

    except Exception as e:
        print(f"Error during optimization: {e}")
        raise


if __name__ == "__main__":
    main()
