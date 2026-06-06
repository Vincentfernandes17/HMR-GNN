# MGTAB Experiment Workflow

This repository trains and evaluates heterophily-aware multi-relational GNNs for bot and stance prediction on the compact `data/MGTAB` tensor dataset.

## Environment

Install the dependencies in `requirements.txt` in a Python environment that supports PyTorch:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## One-Command Full Journal Run

From the project root, this single command runs both bot and stance tasks, all baselines, hyperparameter tuning, ablations, detailed metrics, confusion matrices, gate analysis, and paper table exports:

```powershell
.\.venv\Scripts\python.exe src\run_experiments.py
```

By default, this is equivalent to:

```powershell
.\.venv\Scripts\python.exe src\run_experiments.py --mode all --task both --seeds 42,52,62 --trials 12 --epochs 100
```

## Smoke Test

```powershell
.\.venv\Scripts\python.exe src\run_experiments.py --mode smoke --epochs 2 --patience 2 --quiet
```

## Optional Focused Runs

Baseline comparison:

```powershell
.\.venv\Scripts\python.exe src\run_experiments.py --mode baselines --task bot --seeds 42,52,62 --epochs 100
```

Hyperparameter tuning:

```powershell
.\.venv\Scripts\python.exe src\run_experiments.py --mode tune --task bot --trials 12 --epochs 100
```

Ablation study:

```powershell
.\.venv\Scripts\python.exe src\run_experiments.py --mode ablation --task bot --seeds 42,52,62 --epochs 100
```

Run a smaller full suite if compute is limited:

```powershell
.\.venv\Scripts\python.exe src\run_experiments.py --mode all --task both --seeds 42 --trials 3 --epochs 30
```

## Outputs

Runs are saved under `results/<task>/<model>/seed_<seed>_<timestamp>/` with:

- `summary.json`
- `history.csv`
- `classwise_metrics.csv`
- `subgroup_metrics.csv`
- `confusion_matrix.csv`
- `gate_statistics.csv`
- `predictions.csv`

Paper-style CSV, Markdown, and LaTeX tables are saved under `results/<task>/tables/`.

Cross-task manuscript tables and a run manifest are saved under `results/paper_tables/`.
