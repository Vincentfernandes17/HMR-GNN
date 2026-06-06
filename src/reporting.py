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


def rows_to_latex(rows: List[Dict], caption: str, label: str) -> str:
    if not rows:
        return ""
    fields = list(rows[0].keys())
    colspec = "l" + "c" * (len(fields) - 1)
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
    for row in rows:
        lines.append(" & ".join(_fmt(row.get(field, "")) for field in fields) + " \\\\")
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


def mean_std(values):
    values = [v for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(variance)
