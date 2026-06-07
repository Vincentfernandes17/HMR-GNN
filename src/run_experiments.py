import argparse
import gc
import itertools
import os
import random
from copy import deepcopy
from dataclasses import asdict

import torch

from config import Config
from reporting import ensure_dir, mean_std, significance_table, write_json, write_table_bundle
from train import train


BASELINE_MODELS = ["mlp", "gcn", "graphsage", "gat", "rgcn", "dir_rgcn", "hmr", "hmr_full"]
ABLATION_MODELS = ["hmr", "hmr_homophily", "hmr_directional", "hmr_full"]
# Models compared under adversarial edge injection: a graph-agnostic control (mlp),
# the two strongest ungated relational baselines, and the proposed gated model.
ATTACK_MODELS = ["mlp", "rgcn", "dir_rgcn", "hmr_full"]
ATTACK_FRACTIONS = [0.0, 0.05, 0.10, 0.20]
METRIC_KEYS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
    "auc_roc",
    "sensitivity_macro",
    "specificity_macro",
]


SEARCH_SPACE = {
    "lr": [1e-4, 5e-4, 1e-3, 3e-3],
    "weight_decay": [0.0, 1e-5, 5e-4, 1e-3],
    "dropout": [0.1, 0.3, 0.5],
    "hidden_dim": [64, 128, 256],
    "num_layers": [1, 2, 3],
    "num_heads": [1, 2, 4],
    "patience": [10, 20, 30],
    "gate_temperature": [0.5, 1.0, 2.0],
    "homophily_alpha": [0.0, 0.5, 1.0, 2.0],
    "class_weight_power": [0.0, 0.5, 1.0],
    "batch_size": [0],
}


def _base_config(args):
    cfg = Config()
    cfg.data_dir = args.data_dir
    cfg.output_dir = args.output_dir
    cfg.epochs = args.epochs
    cfg.patience = args.patience
    cfg.tune_trials = args.trials
    cfg.log_gate_scores = True
    return cfg


def _set_params(cfg, params):
    for key, value in params.items():
        setattr(cfg, key, value)
    return cfg


def _metric_row(result):
    metrics = result["metrics"]
    row = {
        "model": result["model"],
        "task": result["task"],
        "seed": result["seed"],
        "best_epoch": result["best_epoch"],
        "run_dir": result["run_dir"],
    }
    for key in METRIC_KEYS:
        row[key] = metrics.get(key)
    return row


def _aggregate(rows, group_key="model"):
    groups = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)

    summary = []
    for group, group_rows in groups.items():
        out = {group_key: group, "runs": len(group_rows)}
        for key in METRIC_KEYS:
            mean, std = mean_std([row.get(key) for row in group_rows])
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std
        summary.append(out)
    return sorted(summary, key=lambda r: r.get("f1_macro_mean", 0.0), reverse=True)


def _run_many(cfg, models, seeds, output_dir, verbose):
    rows = []
    results = []
    for model_name in models:
        for seed in seeds:
            run_cfg = deepcopy(cfg)
            run_cfg.seed = seed
            run_cfg.output_dir = output_dir
            print(f"\n=== {model_name.upper()} | task={run_cfg.task} | seed={seed} ===")
            result = train(run_cfg, model_name=model_name, return_metrics=True, verbose=verbose)
            results.append(result)
            rows.append(_metric_row(result))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows, results


def run_baselines(cfg, seeds, output_dir, verbose):
    rows, results = _run_many(cfg, BASELINE_MODELS, seeds, output_dir, verbose)
    table_dir = ensure_dir(os.path.join(output_dir, cfg.task, "tables"))
    write_table_bundle(
        os.path.join(table_dir, "baseline_runs"),
        rows,
        "Per-run baseline results on MGTAB",
        "tab:baseline-runs",
    )
    write_table_bundle(
        os.path.join(table_dir, "baseline_comparison"),
        _aggregate(rows),
        "Baseline comparison on MGTAB reported as mean and standard deviation across seeds",
        "tab:baseline-comparison",
    )
    sig_rows = significance_table(rows, reference_model="hmr_full")
    if sig_rows:
        write_table_bundle(
            os.path.join(table_dir, "significance_tests"),
            sig_rows,
            "Paired significance tests of HMR-GNN (full) against each baseline across seeds. "
            "Primary test is a paired t-test; Wilcoxon signed-rank p-values are reported for "
            "reference. Significance stars: *** p<0.001, ** p<0.01, * p<0.05, ns not significant.",
            "tab:significance",
        )
    return rows, results


def run_ablation(cfg, seeds, output_dir, verbose):
    rows, results = _run_many(cfg, ABLATION_MODELS, seeds, output_dir, verbose)
    table_dir = ensure_dir(os.path.join(output_dir, cfg.task, "tables"))
    write_table_bundle(
        os.path.join(table_dir, "ablation_runs"),
        rows,
        "Per-run ablation results for the proposed HMRGNN variants",
        "tab:ablation-runs",
    )
    write_table_bundle(
        os.path.join(table_dir, "ablation_comparison"),
        _aggregate(rows),
        "Ablation study for homophily-aware gating and directional aggregation",
        "tab:ablation-comparison",
    )
    return rows, results


def _aggregate_attack(rows):
    groups = {}
    for row in rows:
        key = (row["model"], row["attack_fraction"])
        groups.setdefault(key, []).append(row)
    out = []
    for (model, frac), grp in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        entry = {"model": model, "attack_fraction": frac, "runs": len(grp)}
        for key in ("accuracy", "f1_macro"):
            mean, std = mean_std([r.get(key) for r in grp])
            entry[f"{key}_mean"] = mean
            entry[f"{key}_std"] = std
        out.append(entry)
    return out


def _attack_degradation(agg):
    by_model = {}
    fractions = sorted({r["attack_fraction"] for r in agg})
    for row in agg:
        by_model.setdefault(row["model"], {})[row["attack_fraction"]] = row["f1_macro_mean"]

    rows = []
    max_frac = max(fractions) if fractions else 0.0
    for model, fmap in by_model.items():
        row = {"model": model}
        for frac in fractions:
            row[f"f1_at_{int(round(frac * 100))}pct"] = fmap.get(frac)
        base = fmap.get(0.0)
        attacked = fmap.get(max_frac)
        if base is not None and attacked is not None:
            row["abs_drop"] = base - attacked
            row["rel_drop_pct"] = 100.0 * (base - attacked) / base if base else float("nan")
        rows.append(row)
    # Most robust models (smallest relative drop) first.
    rows.sort(key=lambda r: r.get("rel_drop_pct", float("inf")))
    return rows


def run_attack(cfg, seeds, fractions, output_dir, verbose):
    rows = []
    results = []
    for frac in fractions:
        for model_name in ATTACK_MODELS:
            for seed in seeds:
                run_cfg = deepcopy(cfg)
                run_cfg.seed = seed
                run_cfg.attack_fraction = frac
                run_cfg.output_dir = output_dir
                print(f"\n=== ATTACK {int(round(frac * 100))}% | {model_name.upper()} | seed={seed} ===")
                try:
                    result = train(
                        run_cfg,
                        model_name=model_name,
                        return_metrics=True,
                        save_outputs=False,
                        verbose=verbose,
                    )
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        print(f"[skip] OOM: {model_name} frac={frac} seed={seed}")
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise
                row = _metric_row(result)
                row["attack_fraction"] = frac
                rows.append(row)
                results.append(result)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    table_dir = ensure_dir(os.path.join(output_dir, cfg.task, "tables"))
    agg = _aggregate_attack(rows)
    write_table_bundle(
        os.path.join(table_dir, "attack_robustness"),
        agg,
        "Robustness to injected adversarial/spurious edges (label-independent random "
        "edges; accuracy and macro F1, mean over seeds, by injection intensity)",
        "tab:attack-robustness",
    )
    write_table_bundle(
        os.path.join(table_dir, "attack_degradation"),
        _attack_degradation(agg),
        "Macro F1 degradation under injected spurious edges relative to the clean "
        "graph; smaller relative drop indicates greater robustness",
        "tab:attack-degradation",
    )
    return rows, results


def _sample_search_space(cfg):
    keys = list(SEARCH_SPACE.keys())
    all_trials = [dict(zip(keys, values)) for values in itertools.product(*(SEARCH_SPACE[k] for k in keys))]
    rng = random.Random(cfg.tune_seed)
    rng.shuffle(all_trials)
    return all_trials[: cfg.tune_trials]


def run_tuning(cfg, output_dir, verbose):
    rows = []
    results = []
    table_dir = ensure_dir(os.path.join(output_dir, cfg.task, "tables"))
    for trial_idx, params in enumerate(_sample_search_space(cfg), start=1):
        trial_cfg = deepcopy(cfg)
        _set_params(trial_cfg, params)
        trial_cfg.homophily_gate = True
        trial_cfg.separate_directions = True
        trial_cfg.log_gate_scores = True
        trial_cfg.output_dir = output_dir

        print(f"\n=== TUNING TRIAL {trial_idx}/{cfg.tune_trials}: {params} ===")
        try:
            result = train(trial_cfg, model_name="hmr_full", return_metrics=True, verbose=verbose)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"[skip] trial {trial_idx} ran out of memory, skipping: {params}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
        results.append(result)
        row = {"trial": trial_idx}
        row.update(params)
        row.update(_metric_row(result))
        rows.append(row)

    if not rows:
        print("[warn] no tuning trials succeeded; skipping tuning tables")
        return rows, results, None

    best = max(rows, key=lambda row: row["f1_macro"])
    write_table_bundle(
        os.path.join(table_dir, "hyperparameter_tuning"),
        rows,
        "Hyperparameter tuning trials for the full HMRGNN model",
        "tab:hyperparameter-tuning",
    )
    write_json(os.path.join(table_dir, "best_hyperparameters.json"), best)
    return rows, results, best


def export_best_detailed_tables(best_result, output_dir):
    table_dir = ensure_dir(os.path.join(output_dir, best_result["task"], "tables"))
    model = best_result["model"]
    write_table_bundle(
        os.path.join(table_dir, f"{model}_classwise_metrics"),
        best_result["metrics"]["classwise"],
        "Class-wise performance for the selected model",
        "tab:classwise",
    )
    write_table_bundle(
        os.path.join(table_dir, f"{model}_subgroup_metrics"),
        best_result["metrics"]["subgroups"],
        "Performance by stance and bot value for the selected model",
        "tab:subgroup",
    )
    cm_rows = [
        {"true_class": idx, **{f"pred_{j}": value for j, value in enumerate(row)}}
        for idx, row in enumerate(best_result["metrics"]["confusion_matrix"])
    ]
    write_table_bundle(
        os.path.join(table_dir, f"{model}_confusion_matrix"),
        cm_rows,
        "Confusion matrix for the selected model",
        "tab:confusion-matrix",
    )


def _run_task(args, task):
    cfg = _base_config(args)
    cfg.task = task
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    verbose = not args.quiet

    ensure_dir(os.path.join(args.output_dir, task))
    write_json(os.path.join(args.output_dir, task, "experiment_config.json"), asdict(cfg))

    task_results = []
    task_rows = []

    if args.mode == "attack":
        fractions = [float(x) for x in args.attack_fractions.split(",") if x.strip()] or ATTACK_FRACTIONS
        rows, results = run_attack(cfg, seeds, fractions, args.output_dir, verbose)
        return rows, results

    if args.mode == "smoke":
        cfg.epochs = min(cfg.epochs, 2)
        cfg.patience = min(cfg.patience, 2)
        rows, results = _run_many(cfg, ["mlp", "rgcn", "hmr_full"], seeds[:1], args.output_dir, verbose)
        task_rows.extend(rows)
        task_results.extend(results)
        write_table_bundle(
            os.path.join(args.output_dir, task, "tables", "smoke_results"),
            rows,
            "Smoke-test results for key models",
            "tab:smoke-results",
        )
    else:
        if args.mode in {"baselines", "all"}:
            rows, results = run_baselines(cfg, seeds, args.output_dir, verbose)
            task_rows.extend(rows)
            task_results.extend(results)
        # Ablation runs before tuning: it is cheaper and more essential to the paper,
        # so it should complete even if the (heavier, large-config) tuning struggles.
        if args.mode in {"ablation", "all"}:
            rows, results = run_ablation(cfg, seeds, args.output_dir, verbose)
            task_rows.extend(rows)
            task_results.extend(results)
        if args.mode in {"tune", "all"}:
            rows, results, _ = run_tuning(cfg, args.output_dir, verbose)
            task_rows.extend(rows)
            task_results.extend(results)

    if task_results:
        best_result = max(task_results, key=lambda result: result["metrics"]["f1_macro"])
        export_best_detailed_tables(best_result, args.output_dir)
        write_json(os.path.join(args.output_dir, task, "best_result.json"), best_result)
        print("\n=== BEST RUN ===")
        print(f"task={task} model={best_result['model']} seed={best_result['seed']} f1_macro={best_result['metrics']['f1_macro']:.4f}")
        print(f"run_dir={best_result['run_dir']}")

    return task_rows, task_results


def _export_master_tables(all_rows, all_results, output_dir):
    master_dir = ensure_dir(os.path.join(output_dir, "paper_tables"))
    write_table_bundle(
        os.path.join(master_dir, "all_runs"),
        all_rows,
        "All MGTAB experiment runs across bot and stance tasks",
        "tab:all-runs",
    )
    write_table_bundle(
        os.path.join(master_dir, "model_task_comparison"),
        _aggregate(all_rows, group_key="model"),
        "Overall model comparison aggregated across requested tasks and seeds",
        "tab:model-task-comparison",
    )

    best_rows = []
    for task in sorted({result["task"] for result in all_results}):
        task_results = [result for result in all_results if result["task"] == task]
        best = max(task_results, key=lambda result: result["metrics"]["f1_macro"])
        row = {
            "task": task,
            "model": best["model"],
            "seed": best["seed"],
            "best_epoch": best["best_epoch"],
            "run_dir": best["run_dir"],
        }
        for key in METRIC_KEYS:
            row[key] = best["metrics"].get(key)
        best_rows.append(row)

    write_table_bundle(
        os.path.join(master_dir, "best_runs_by_task"),
        best_rows,
        "Best selected run for each prediction task",
        "tab:best-runs-by-task",
    )
    write_json(os.path.join(master_dir, "run_manifest.json"), {
        "num_runs": len(all_results),
        "tasks": sorted({result["task"] for result in all_results}),
        "models": sorted({result["model"] for result in all_results}),
        "best_runs": best_rows,
    })


def parse_args():
    parser = argparse.ArgumentParser(description="Publication-ready MGTAB experiment runner.")
    parser.add_argument("--mode", choices=["smoke", "baselines", "tune", "ablation", "attack", "all"], default="all")
    parser.add_argument("--task", choices=["bot", "stance", "both"], default="both")
    parser.add_argument("--data-dir", default="./data/MGTAB")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--seeds", default="42,52,62,72,82")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--attack-fractions", default="0.0,0.05,0.10,0.20")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    tasks = ["bot", "stance"] if args.task == "both" else [args.task]
    all_rows = []
    all_results = []

    print("\n=== MGTAB PUBLICATION EXPERIMENT SUITE ===")
    print(f"mode={args.mode} tasks={','.join(tasks)} seeds={args.seeds} epochs={args.epochs} trials={args.trials}")
    print(f"output_dir={args.output_dir}")

    for task in tasks:
        print(f"\n\n######## TASK: {task.upper()} ########")
        rows, results = _run_task(args, task)
        all_rows.extend(rows)
        all_results.extend(results)

    if all_results:
        _export_master_tables(all_rows, all_results, args.output_dir)
        print("\n=== COMPLETE ===")
        print(f"Generated per-task outputs in {args.output_dir}\\bot and {args.output_dir}\\stance")
        print(f"Generated cross-task paper tables in {args.output_dir}\\paper_tables")


if __name__ == "__main__":
    main()
