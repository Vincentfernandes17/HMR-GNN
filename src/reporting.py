import csv
import json
import math
import os
from typing import Dict, Iterable, List


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, payload) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_csv(path: str, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value):
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.4f}"
    return str(value)


def rows_to_markdown(rows: List[Dict]) -> str:
    if not rows:
        return ""
    fields = list(rows[0].keys())
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"


# Substrings of column names whose best (maximum) value should be bolded in
# LaTeX output. Higher is better for all of these.
_HIGHER_BETTER_KEYS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "sensitivity",
    "specificity",
)
# Columns that look like a metric substring but must never be bolded.
_NEVER_BOLD_KEYS = ("_std", "support", "seed", "epoch", "trial", "p_value", "params", "time")


def _is_highlightable(field: str) -> bool:
    name = field.lower()
    if any(skip in name for skip in _NEVER_BOLD_KEYS):
        return False
    return any(key in name for key in _HIGHER_BETTER_KEYS)


def _best_row_per_field(rows: List[Dict], fields: List[str]) -> Dict[str, int]:
    """For each highlightable numeric column, return the row index of the max value."""
    best = {}
    for field in fields:
        if not _is_highlightable(field):
            continue
        best_idx, best_val = None, None
        for idx, row in enumerate(rows):
            value = row.get(field)
            if not isinstance(value, (int, float)) or (isinstance(value, float) and math.isnan(value)):
                continue
            if best_val is None or value > best_val:
                best_val, best_idx = value, idx
        if best_idx is not None:
            best[field] = best_idx
    return best


def rows_to_latex(rows: List[Dict], caption: str, label: str) -> str:
    if not rows:
        return ""
    fields = list(rows[0].keys())
    colspec = "l" + "c" * (len(fields) - 1)
    best_idx = _best_row_per_field(rows, fields)
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\hline",
        " & ".join(fields) + " \\\\",
        "\\hline",
    ]
    for row_idx, row in enumerate(rows):
        cells = []
        for field in fields:
            text = _fmt(row.get(field, ""))
            if best_idx.get(field) == row_idx:
                text = f"\\textbf{{{text}}}"
            cells.append(text)
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def write_table_bundle(path_prefix: str, rows: List[Dict], caption: str, label: str) -> None:
    write_csv(path_prefix + ".csv", rows)
    ensure_dir(os.path.dirname(path_prefix))
    with open(path_prefix + ".md", "w", encoding="utf-8") as f:
        f.write(rows_to_markdown(rows))
    with open(path_prefix + ".tex", "w", encoding="utf-8") as f:
        f.write(rows_to_latex(rows, caption, label))


def _stars(p_value: float) -> str:
    if p_value is None or (isinstance(p_value, float) and math.isnan(p_value)):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def significance_table(per_run_rows, reference_model, metrics=("f1_macro", "accuracy"),
                       group_key="model", seed_key="seed"):
    """
    Paired significance tests comparing `reference_model` against every other model
    over shared seeds. Reports paired t-test (primary) and Wilcoxon signed-rank
    (secondary) p-values. Returns a list of rows ready for table export.
    """
    from scipy import stats

    # Collect {model: {seed: value}} for each metric.
    by_model = {}
    for row in per_run_rows:
        model = row.get(group_key)
        seed = row.get(seed_key)
        by_model.setdefault(model, {})[seed] = row

    if reference_model not in by_model:
        return []

    ref_runs = by_model[reference_model]
    out_rows = []
    for model, runs in by_model.items():
        if model == reference_model:
            continue
        shared_seeds = sorted(set(ref_runs) & set(runs))
        if len(shared_seeds) < 2:
            continue
        for metric in metrics:
            ref_vals = [ref_runs[s].get(metric) for s in shared_seeds]
            cmp_vals = [runs[s].get(metric) for s in shared_seeds]
            if any(v is None for v in ref_vals + cmp_vals):
                continue
            mean_ref = sum(ref_vals) / len(ref_vals)
            mean_cmp = sum(cmp_vals) / len(cmp_vals)
            try:
                t_stat, t_p = stats.ttest_rel(ref_vals, cmp_vals)
            except Exception:
                t_p = float("nan")
            try:
                _, w_p = stats.wilcoxon(ref_vals, cmp_vals)
            except Exception:
                w_p = float("nan")
            out_rows.append({
                "comparison": f"{reference_model} vs {model}",
                "metric": metric,
                "mean_proposed": mean_ref,
                "mean_baseline": mean_cmp,
                "mean_diff": mean_ref - mean_cmp,
                "n_seeds": len(shared_seeds),
                "ttest_p": float(t_p),
                "wilcoxon_p": float(w_p),
                "significance": _stars(float(t_p)),
            })
    return out_rows


def mean_std(values):
    values = [v for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(variance)
