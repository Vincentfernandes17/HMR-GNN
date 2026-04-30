import numpy as np
import torch

from config import Config
from train import train  # kita akan modif sedikit
from baselines import MLPBaseline, RGCNBaseline
from model import HMRGNN


# =========================
# CONFIG EXPERIMENT
# =========================
SEEDS = [42, 52, 62, 72, 82]
MODELS = ["mlp", "rgcn", "hmr"]


# =========================
# RUN SINGLE MODEL
# =========================
def run_model(model_name):
    results = []

    for seed in SEEDS:
        cfg = Config()
        cfg.seed = seed

        print(f"\n=== Running {model_name.upper()} | Seed {seed} ===")

        metrics = train(cfg, model_name=model_name, return_metrics=True)
        results.append(metrics)

    return results


# =========================
# AGGREGATE RESULTS
# =========================
def aggregate(results):
    keys = results[0].keys()
    agg = {}

    for k in keys:
        values = [r[k] for r in results]
        agg[k] = {
            "mean": np.mean(values),
            "std": np.std(values)
        }

    return agg


# =========================
# LATEX TABLE GENERATOR
# =========================
def generate_latex(results_dict):
    print("\n\n=== LATEX TABLE ===\n")

    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Performance Comparison on MGTAB}")
    print("\\begin{tabular}{lcc}")
    print("\\hline")
    print("Model & Accuracy & F1-macro \\\\")
    print("\\hline")

    for model, res in results_dict.items():
        acc = res["accuracy"]
        f1 = res["f1"]

        acc_str = f"{acc['mean']:.4f} $\\pm$ {acc['std']:.4f}"
        f1_str = f"{f1['mean']:.4f} $\\pm$ {f1['std']:.4f}"

        print(f"{model.upper()} & {acc_str} & {f1_str} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


# =========================
# MAIN
# =========================
def main():
    final_results = {}

    for model in MODELS:
        results = run_model(model)

        agg = aggregate(results)

        # rename key biar konsisten
        final_results[model] = {
            "accuracy": agg["accuracy"],
            "f1": agg["f1_macro"]
        }

    generate_latex(final_results)


if __name__ == "__main__":
    main()