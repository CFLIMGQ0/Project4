#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

import yaml

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH_CONFIG = PROJECT_ROOT / "configs" / "task1" / "path.yaml"
TASK1_LABEL_NAMES = (
    "label_esophageal_smt",
    "label_esophageal_mucosal_or_tumor",
    "label_gastritis",
)
CHECKPOINT_ALIAS = "best_macro_f1"
EXPERIMENT_ORDER = {
    "exp_task1_ablation1_full_label_graph": 1,
    "exp_task1_ablation1_wo_label_graph": 2,
    "exp_task1_ablation1_label_self_attention": 3,
    "exp_task1_ablation1_static_gcn": 4,
    "exp_task1_ablation1_dynamic_gat": 5,
    "exp_task1_ablation1_label_transformer": 6,
    "exp_task1_ablation1_low_rank_graph": 7,
    "exp_task1_ablation1_cosine_graph": 8,
    "exp_task1_ablation1_label_mlp_mixer": 9,
    "exp_task1_ablation1_label_hypergraph": 10,
}
DISPLAY_NAMES = {
    "exp_task1_ablation1_full_label_graph": "Full Label Graph Reasoner",
    "exp_task1_ablation1_wo_label_graph": "w/o Label Graph Reasoner",
    "exp_task1_ablation1_label_self_attention": "Label Self-Attention Reasoner",
    "exp_task1_ablation1_static_gcn": "Static Co-occurrence GCN Reasoner",
    "exp_task1_ablation1_dynamic_gat": "Dynamic Label GAT Reasoner",
    "exp_task1_ablation1_label_transformer": "Label Transformer Reasoner",
    "exp_task1_ablation1_low_rank_graph": "Low-Rank Label Graph Reasoner",
    "exp_task1_ablation1_cosine_graph": "Cosine Dynamic Graph Reasoner",
    "exp_task1_ablation1_label_mlp_mixer": "Label MLP-Mixer Reasoner",
    "exp_task1_ablation1_label_hypergraph": "Label Hypergraph Reasoner",
}
SUMMARY_FIELDS = [
    "experiment_name",
    "backbone",
    "use_label_graph",
    "label_graph_type",
    "use_label_wise_attention",
    "attention_type",
    "pooling_type",
    "seed",
    "best_epoch",
    "test_macro_f1",
    "test_micro_f1",
    "test_macro_auc",
    "test_macro_prauc",
    "test_subset_accuracy",
    "test_kappa",
    "label1_f1",
    "label2_f1",
    "label3_f1",
    "checkpoint_path",
    "config_path",
]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 TASK1 消融实验测试结果。")
    parser.add_argument("--path-config", type=Path, default=DEFAULT_PATH_CONFIG, help="TASK1 path.yaml")
    parser.add_argument("--ablation-root", type=Path, default=None, help="task1_ablation 根目录")
    parser.add_argument("--output-dir", type=Path, default=None, help="结果输出目录，默认写入 task1_ablation/results")
    parser.add_argument("--selection-alias", default=CHECKPOINT_ALIAS, help="从 test_result.csv 选择的 checkpoint_alias")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML 格式错误: {path}")
    return payload


def resolve_ablation_root(path_config: Path, ablation_root: Path | None) -> Path:
    if ablation_root is not None:
        return ablation_root.expanduser().resolve()
    path_cfg = load_yaml(path_config)
    output_dir = Path(str(path_cfg["paths"]["output_dir"] if "paths" in path_cfg else path_cfg["output_dir"]))
    return (output_dir.expanduser().resolve() / "train_runs" / "task1" / "task1_ablation").resolve()


def safe_float(value: Any) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def safe_int_text(value: Any) -> str:
    try:
        return str(int(float(str(value).strip())))
    except Exception:
        return ""


def format_float(value: Any, digits: int = 6) -> str:
    numeric = safe_float(value)
    if math.isnan(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def choose_result_row(rows: list[dict[str, str]], selection_alias: str) -> dict[str, str] | None:
    for row in rows:
        if str(row.get("checkpoint_alias", "")).strip() == selection_alias:
            return row
    return rows[0] if rows else None


def infer_experiment_name(train_dir: Path, row: dict[str, str]) -> str:
    explicit_name = str(row.get("experiment_name", "")).strip()
    if explicit_name:
        return explicit_name
    match = re.match(r"^train_\d+_(.+)$", train_dir.name)
    return match.group(1) if match else train_dir.name


def base_experiment_name(experiment_name: str, row: dict[str, str]) -> str:
    explicit_group = str(row.get("seed_group_name", "")).strip()
    if explicit_group:
        return explicit_group
    return re.sub(r"_seed\d+$", "", experiment_name)


def metric_from_row(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key, "")
        if str(value).strip() != "":
            return format_float(value)
    return ""


def build_summary_row(test_csv_path: Path, selection_alias: str) -> dict[str, Any] | None:
    rows = read_csv_rows(test_csv_path)
    result_row = choose_result_row(rows, selection_alias)
    if result_row is None:
        return None

    train_dir = test_csv_path.parent
    config_path = train_dir / "config.yaml"
    experiment_name = infer_experiment_name(train_dir, result_row)
    group_name = base_experiment_name(experiment_name, result_row)
    summary_order_value = safe_float(result_row.get("summary_order", ""))
    if math.isnan(summary_order_value):
        summary_order_value = float(EXPERIMENT_ORDER.get(group_name, 99))

    label_f1_values = [
        metric_from_row(result_row, f"f1_{label_name}")
        for label_name in TASK1_LABEL_NAMES
    ]
    while len(label_f1_values) < 3:
        label_f1_values.append("")

    summary_row: dict[str, Any] = {
        "experiment_name": experiment_name,
        "backbone": str(result_row.get("backbone", "")).strip(),
        "use_label_graph": str(result_row.get("use_label_graph", "")).strip(),
        "label_graph_type": str(result_row.get("label_graph_type", "")).strip(),
        "use_label_wise_attention": str(result_row.get("use_label_wise_attention", "")).strip(),
        "attention_type": str(result_row.get("attention_type", "")).strip(),
        "pooling_type": str(result_row.get("pooling_type", "")).strip(),
        "seed": safe_int_text(result_row.get("seed", "")),
        "best_epoch": safe_int_text(result_row.get("best_epoch", "")),
        "test_macro_f1": metric_from_row(result_row, "macro_f1"),
        "test_micro_f1": metric_from_row(result_row, "micro_f1"),
        "test_macro_auc": metric_from_row(result_row, "macro_auc", "macro_roc_auc"),
        "test_macro_prauc": metric_from_row(result_row, "macro_pr_auc", "macro_ap"),
        "test_subset_accuracy": metric_from_row(result_row, "subset_accuracy"),
        "test_kappa": metric_from_row(result_row, "kappa"),
        "label1_f1": label_f1_values[0],
        "label2_f1": label_f1_values[1],
        "label3_f1": label_f1_values[2],
        "checkpoint_path": str(result_row.get("checkpoint_path", "")).strip(),
        "config_path": str(config_path.resolve()),
        "_summary_order": int(summary_order_value),
        "_summary_name": str(result_row.get("summary_name", DISPLAY_NAMES.get(group_name, experiment_name))).strip(),
        "_seed_group_name": group_name,
    }
    return summary_row


def discover_test_result_csvs(ablation_root: Path) -> list[Path]:
    if not ablation_root.is_dir():
        return []
    paths = [
        path
        for path in ablation_root.rglob("test_result.csv")
        if "results" not in path.relative_to(ablation_root).parts
    ]
    return sorted(paths, key=lambda item: str(item))


def sort_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        seed = safe_float(row.get("seed", ""))
        seed_key = seed if not math.isnan(seed) else -1
        return (
            int(row.get("_summary_order", 99)),
            str(row.get("_seed_group_name", "")),
            seed_key,
            str(row.get("experiment_name", "")),
        )

    return sorted(rows, key=key)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_ablation_markdown(path: Path, rows: list[dict[str, Any]], missing_groups: list[str]) -> None:
    lines = [
        "# TASK1 消融实验汇总",
        "",
        "| 顺序 | 实验 | seed | backbone | graph | graph type | attention | pooling | macro F1 | micro F1 | macro AUC | macro PR-AUC | subset acc | kappa |",
        "|---:|---|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.get('_summary_order', '')} | "
            f"{row.get('_summary_name', row.get('experiment_name', ''))} | "
            f"{row.get('seed', '')} | "
            f"{row.get('backbone', '')} | "
            f"{row.get('use_label_graph', '')} | "
            f"{row.get('label_graph_type', '')} | "
            f"{row.get('attention_type', '')} | "
            f"{row.get('pooling_type', '')} | "
            f"{row.get('test_macro_f1', '')} | "
            f"{row.get('test_micro_f1', '')} | "
            f"{row.get('test_macro_auc', '')} | "
            f"{row.get('test_macro_prauc', '')} | "
            f"{row.get('test_subset_accuracy', '')} | "
            f"{row.get('test_kappa', '')} |"
        )

    if missing_groups:
        lines.extend(["", "## TODO", ""])
        for group_name in missing_groups:
            lines.append(f"- 尚未发现 `{group_name}` 的 `test_result.csv`，训练完成后重新运行汇总脚本即可补齐。")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_task1_ablation_summaries(
    *,
    ablation_root: Path,
    output_dir: Path | None = None,
    selection_alias: str = CHECKPOINT_ALIAS,
) -> Path:
    output_dir = output_dir.expanduser().resolve() if output_dir else (ablation_root / "results").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    test_csv_paths = discover_test_result_csvs(ablation_root)
    iterator = tqdm(test_csv_paths, desc="汇总 TASK1 消融结果", dynamic_ncols=True) if tqdm is not None else test_csv_paths

    rows: list[dict[str, Any]] = []
    for test_csv_path in iterator:
        row = build_summary_row(test_csv_path, selection_alias)
        if row is not None:
            rows.append(row)
    rows = sort_summary_rows(rows)

    found_groups = {str(row.get("_seed_group_name", "")) for row in rows}
    missing_groups = [name for name in EXPERIMENT_ORDER if name not in found_groups]

    write_csv(output_dir / "ablation_summary.csv", rows, SUMMARY_FIELDS)
    write_ablation_markdown(output_dir / "ablation_summary.md", rows, missing_groups)

    print(f"TASK1 消融汇总目录: {output_dir}")
    print(f"- {output_dir / 'ablation_summary.csv'}")
    print(f"- {output_dir / 'ablation_summary.md'}")
    return output_dir


def main() -> None:
    args = parse_args()
    ablation_root = resolve_ablation_root(args.path_config, args.ablation_root)
    write_task1_ablation_summaries(
        ablation_root=ablation_root,
        output_dir=args.output_dir,
        selection_alias=args.selection_alias,
    )


if __name__ == "__main__":
    main()
