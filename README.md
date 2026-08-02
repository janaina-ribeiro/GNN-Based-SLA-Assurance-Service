# GNN-Based-SLA-Assurance-Service

This repository contains experiments, datasets, training scripts, and evaluation pipelines used to build a GNN-based SLA assurance service for Internet Service Providers.

Dataset source (2023 delay + traceroute):
https://ieee-dataport.org/documents/datasets-delay-traceroute-2023

## End-to-End Workflow

1. Prepare environment and dependencies
2. Build graph datasets (horizons 3, 10, 30 minutes)
3. Train GraphSAGE models
4. Run closed-loop replay
5. Evaluate replay outputs (metrics + calibration)

## 1) Environment Setup

Use Python 3.12 and create/activate a virtual environment.

PowerShell (Windows):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
python -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you want GPU training, install PyTorch with CUDA and compatible PyG wheels (example for CUDA 12.4):

```powershell
python -m pip uninstall -y torch torchvision torchaudio torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
python -m pip install torch-geometric==2.7.0
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 2) Build Graph Datasets (3, 10, 30 min)

The builder is `test_gnn.dataset_builder` and writes `.joblib` graph datasets.

```powershell
python -m test_gnn.dataset_builder --data-dir datasets_generated --window-size 6 --horizon-minutes 3  --min-corr 0.3 --output-joblib "results_artifacts/datasets/gnn_dataset_w6_h3min_fixed.joblib"
python -m test_gnn.dataset_builder --data-dir datasets_generated --window-size 6 --horizon-minutes 10 --min-corr 0.3 --output-joblib "results_artifacts/datasets/gnn_dataset_w6_h10min.joblib"
python -m test_gnn.dataset_builder --data-dir datasets_generated --window-size 6 --horizon-minutes 30 --min-corr 0.3 --output-joblib "results_artifacts/datasets/gnn_dataset_w6_h30min.joblib"
```

To inspect existing datasets metadata:

```powershell
python -c "import joblib; from pathlib import Path; base=Path('results_artifacts/datasets'); files=sorted(base.glob('*.joblib')); print(f'Total: {len(files)}'); [print({'file':str(p),'window_size':(d:=joblib.load(p)).get('window_size'),'horizon_minutes':d.get('horizon_minutes'),'freq_minutes':d.get('freq_minutes'),'offset_steps':d.get('offset_steps'),'horizon_inferido':(d.get('freq_minutes')*d.get('offset_steps') if d.get('horizon_minutes') is None and d.get('freq_minutes') is not None and d.get('offset_steps') is not None else d.get('horizon_minutes'))}) for p in files]"
```

## 3) Train GraphSAGE Models

Training script: `test_gnn.train_gnn`.

Recommended hyperparameters from the current experiments:
- `conv_type=sage`
- `hidden_channels=256`
- `num_layers=4`
- `dropout=0.2`
- `lr=0.001951`
- `weight_decay=3.335026e-05`
- `label_smoothing=0.05`
- `sample_weight_alpha=1.5`
- `max_grad_norm=0.5`
- `scheduler=none`

Train for horizon 10 min:

```powershell
python -m test_gnn.train_gnn --data-dir datasets_generated --window-size 6 --horizon-minutes 10 --dataset-joblib "results_artifacts/datasets/gnn_dataset_w6_h10min.joblib" --strict-dataset --min-corr 0.3 --train-ratio 0.7 --val-ratio 0.15 --epochs 20 --batch-size 32 --hidden-channels 256 --num-layers 4 --conv-type sage --gat-heads 4 --dropout 0.2 --lr 0.001951 --weight-decay 3.335026e-05 --label-smoothing 0.05 --loss-type cross_entropy --focal-gamma 2.0 --sample-weight-alpha 1.5 --max-grad-norm 0.5 --scheduler none --scheduler-factor 0.5 --scheduler-patience 5 --scheduler-t0 10 --patience 8 --device cuda --seed 42 --model-path "results_artifacts/best_model_output/sage_h10min.pt" --training-log-path "results_artifacts/best_model_output/sage_h10min_log.json"
```

Train for horizon 30 min:

```powershell
python -m test_gnn.train_gnn --data-dir datasets_generated --window-size 6 --horizon-minutes 30 --dataset-joblib "results_artifacts/datasets/gnn_dataset_w6_h30min.joblib" --strict-dataset --min-corr 0.3 --train-ratio 0.7 --val-ratio 0.15 --epochs 20 --batch-size 32 --hidden-channels 256 --num-layers 4 --conv-type sage --gat-heads 4 --dropout 0.2 --lr 0.001951 --weight-decay 3.335026e-05 --label-smoothing 0.05 --loss-type cross_entropy --focal-gamma 2.0 --sample-weight-alpha 1.5 --max-grad-norm 0.5 --scheduler none --scheduler-factor 0.5 --scheduler-patience 5 --scheduler-t0 10 --patience 8 --device cuda --seed 42 --model-path "results_artifacts/best_model_output/sage_h30min.pt" --training-log-path "results_artifacts/best_model_output/sage_h30min_log.json"
```

Optional train for horizon 3 min:

```powershell
python -m test_gnn.train_gnn --data-dir datasets_generated --window-size 6 --horizon-minutes 3 --dataset-joblib "results_artifacts/datasets/gnn_dataset_w6_h3min_fixed.joblib" --strict-dataset --min-corr 0.3 --train-ratio 0.7 --val-ratio 0.15 --epochs 20 --batch-size 32 --hidden-channels 256 --num-layers 4 --conv-type sage --gat-heads 4 --dropout 0.2 --lr 0.001951 --weight-decay 3.335026e-05 --label-smoothing 0.05 --loss-type cross_entropy --focal-gamma 2.0 --sample-weight-alpha 1.5 --max-grad-norm 0.5 --scheduler none --scheduler-factor 0.5 --scheduler-patience 5 --scheduler-t0 10 --patience 8 --device cuda --seed 42 --model-path "results_artifacts/best_model_output/sage_h3min.pt" --training-log-path "results_artifacts/best_model_output/sage_h3min_log.json"
```

## 4) Closed-Loop Replay

Replay script: `test_gnn.replay_closed_loop`.

Default models/datasets are:
- `results_artifacts/best_model_output/sage_h3min.pt`
- `results_artifacts/best_model_output/sage_h10min.pt`
- `results_artifacts/best_model_output/sage_h30min.pt`
- `results_artifacts/datasets/gnn_dataset_w6_h3min_fixed.joblib`
- `results_artifacts/datasets/gnn_dataset_w6_h10min.joblib`
- `results_artifacts/datasets/gnn_dataset_w6_h30min.joblib`

Run replay:

```powershell
python -m test_gnn.replay_closed_loop --thresholds "3:0.55,10:0.45,30:0.35" --cooldown-minutes 10 --output-csv "results_artifacts/replay_events.csv" --summary-json "results_artifacts/replay_summary.json" --device cuda
```

## 5) Evaluate Replay Outputs

Evaluation script: `test_gnn.evaluate_replay`.

```powershell
python -m test_gnn.evaluate_replay --input-csv "results_artifacts/replay_events.csv" --output-metrics-csv "results_artifacts/experiment_a_metrics.csv" --output-calibration-csv "results_artifacts/experiment_a_calibration.csv" --output-summary-json "results_artifacts/experiment_a_summary.json" --min-actionable-lead 3 --calibration-bins 10
```

Generated outputs:
- `results_artifacts/replay_events.csv`
- `results_artifacts/replay_summary.json`
- `results_artifacts/experiment_a_metrics.csv`
- `results_artifacts/experiment_a_calibration.csv`
- `results_artifacts/experiment_a_summary.json`

## Notes

- If CUDA is not available in your environment, replace `--device cuda` with `--device cpu`.
- Keep `--strict-dataset` enabled to catch metadata mismatches between CLI parameters and dataset artifacts.
- The loader supports legacy joblib datasets with missing `horizon_minutes` by inferring from `freq_minutes * offset_steps`.

## Contact

- Janaina Ribeiro
- janainaribeiro780@gmail.com
- janaina.ribeiro@aluno.uece.br
