from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_CSV = PROJECT_ROOT / "datasets" / "valid_dicts_report_for task2.csv"
DEFAULT_DATALIST_CSV = PROJECT_ROOT / "datasets" / "task_data" / "task2" / "gastro_multilabel_task_datalist.csv"
EXPERIMENT_NAME = "key-label_basic_fields"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME

LABEL_COLUMNS = [
    "label_esophageal_smt",
    "label_esophageal_mucosal_or_tumor",
    "label_gastritis",
]

FIELD_ORDER = ["reportTitle", "age", "sex", "hp", "operationValue"]
MISSING_TOKEN = "__MISSING__"
OTHER_TOKEN = "__OTHER_LOW_FREQ__"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    source_aliases: tuple[str, ...]
    kind: str
    description: str


FIELD_SPECS = {
    "reportTitle": FieldSpec(
        name="reportTitle",
        source_aliases=("reportTitle", "report_title"),
        kind="categorical",
        description="检查标题/检查场景",
    ),
    "age": FieldSpec(
        name="age",
        source_aliases=("age",),
        kind="age_numeric",
        description="年龄，脚本会先分箱再做类别关联分析",
    ),
    "sex": FieldSpec(
        name="sex",
        source_aliases=("sex",),
        kind="categorical",
        description="性别",
    ),
    "hp": FieldSpec(
        name="hp",
        source_aliases=("hp", "hp_status"),
        kind="categorical",
        description="HP 状态",
    ),
    "operationValue": FieldSpec(
        name="operationValue",
        source_aliases=("operationValue", "openationValue", "operation_value"),
        kind="categorical",
        description="操作/检查类型；openationValue 会自动按 operationValue 兼容",
    ),
}


def configure_matplotlib_font() -> None:
    preferred_names = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Droid Sans Fallback",
        "SimHei",
        "Microsoft YaHei",
    ]
    preferred_paths = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ]

    for font_path in preferred_paths:
        if font_path.is_file():
            try:
                font_manager.fontManager.addfont(str(font_path))
            except RuntimeError:
                pass

    available_names = {font.name for font in font_manager.fontManager.ttflist}
    selected_name = next((name for name in preferred_names if name in available_names), None)
    if selected_name is not None:
        plt.rcParams["font.sans-serif"] = [selected_name, "DejaVu Sans"]
        plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


configure_matplotlib_font()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 TASK2 多模态有效键与三标签的关联")
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV, help="源报告总表 CSV")
    parser.add_argument("--datalist-csv", type=Path, default=DEFAULT_DATALIST_CSV, help="TASK2 三标签 datalist CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="实验输出目录")
    parser.add_argument("--min-category-count", type=int, default=20, help="低于该样本数的类别合并为 __OTHER_LOW_FREQ__")
    parser.add_argument("--top-categories", type=int, default=30, help="每张类别图最多展示的类别数")
    parser.add_argument("--no-plots", action="store_true", help="只输出 CSV，不生成可视化图")
    return parser.parse_args()


def progress(iterable: Any, *, total: int | None = None, desc: str = "") -> Any:
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, unit="项")
    return iterable


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).replace("\u3000", " ").strip()


def normalize_category(value: Any) -> str:
    cleaned = normalize_text(value)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned if cleaned else MISSING_TOKEN


def resolve_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def load_merged_rows(report_csv: Path, datalist_csv: Path) -> pd.DataFrame:
    if not report_csv.is_file():
        raise FileNotFoundError(f"未找到源报告总表：{report_csv}")
    if not datalist_csv.is_file():
        raise FileNotFoundError(f"未找到 TASK2 datalist：{datalist_csv}")

    report_df = pd.read_csv(report_csv, dtype=str, encoding="utf-8-sig").fillna("")
    datalist_df = pd.read_csv(datalist_csv, dtype=str, encoding="utf-8-sig").fillna("")

    if "exam_dir" not in report_df.columns:
        raise KeyError("源报告总表缺少 exam_dir 字段")
    if "exam_dir" not in datalist_df.columns:
        raise KeyError("TASK2 datalist 缺少 exam_dir 字段")

    missing_labels = [label for label in LABEL_COLUMNS if label not in datalist_df.columns]
    if missing_labels:
        raise KeyError(f"TASK2 datalist 缺少标签字段：{', '.join(missing_labels)}")

    report_keep_cols = ["exam_dir"]
    for field_name in FIELD_ORDER:
        column = resolve_column(report_df, FIELD_SPECS[field_name].source_aliases)
        if column is not None and column not in report_keep_cols:
            report_keep_cols.append(column)

    merged = datalist_df.merge(
        report_df[report_keep_cols],
        on="exam_dir",
        how="left",
        suffixes=("", "__report"),
        validate="many_to_one",
    )

    for field_name in FIELD_ORDER:
        spec = FIELD_SPECS[field_name]
        datalist_col = resolve_column(datalist_df, spec.source_aliases)
        report_col = resolve_column(report_df, spec.source_aliases)
        merged_col = field_name

        if datalist_col is not None and datalist_col in merged.columns:
            base_values = merged[datalist_col].astype(str)
            if report_col is not None:
                candidate_col = report_col if report_col not in datalist_df.columns else f"{report_col}__report"
                if candidate_col in merged.columns:
                    report_values = merged[candidate_col].astype(str)
                    base_values = base_values.where(base_values.str.strip() != "", report_values)
            merged[merged_col] = base_values
        elif report_col is not None:
            candidate_col = report_col if report_col in merged.columns else f"{report_col}__report"
            if candidate_col in merged.columns:
                merged[merged_col] = merged[candidate_col].astype(str)
            else:
                merged[merged_col] = ""
        else:
            merged[merged_col] = ""

    for label in LABEL_COLUMNS:
        merged[label] = pd.to_numeric(merged[label], errors="coerce").fillna(0).astype(int).clip(0, 1)

    return merged


def age_to_bin(value: Any) -> str:
    cleaned = normalize_text(value)
    if not cleaned:
        return MISSING_TOKEN
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    if not match:
        return MISSING_TOKEN
    age = float(match.group(0))
    if age < 30:
        return "age_<30"
    if age < 40:
        return "age_30-39"
    if age < 50:
        return "age_40-49"
    if age < 60:
        return "age_50-59"
    if age < 70:
        return "age_60-69"
    if age < 80:
        return "age_70-79"
    return "age_80+"


def encode_field(series: pd.Series, field_name: str, min_category_count: int) -> pd.Series:
    spec = FIELD_SPECS[field_name]
    if spec.kind == "age_numeric":
        encoded = series.map(age_to_bin)
    else:
        encoded = series.map(normalize_category)

    counts = encoded.value_counts(dropna=False)
    low_freq = {name for name, count in counts.items() if name != MISSING_TOKEN and count < min_category_count}
    if low_freq:
        encoded = encoded.map(lambda value: OTHER_TOKEN if value in low_freq else value)
    return encoded.astype(str)


def build_field_audit(raw_df: pd.DataFrame, encoded_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(raw_df)
    for field_name in FIELD_ORDER:
        raw_values = raw_df[field_name].map(normalize_text)
        non_missing = raw_values[raw_values != ""]
        encoded_values = encoded_df[field_name]
        rows.append(
            {
                "field": field_name,
                "description": FIELD_SPECS[field_name].description,
                "total": total,
                "non_missing": int(len(non_missing)),
                "missing": int(total - len(non_missing)),
                "missing_rate": safe_div(total - len(non_missing), total),
                "raw_unique_non_missing": int(non_missing.nunique(dropna=True)),
                "encoded_unique": int(encoded_values.nunique(dropna=True)),
                "encoded_categories": " | ".join(map(str, encoded_values.value_counts().index.tolist())),
            }
        )
    return rows


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-(probs * np.log(probs)).sum())


def cramers_v_from_contingency(contingency: pd.DataFrame, chi2: float) -> float:
    matrix = contingency.to_numpy(dtype=float)
    n = matrix.sum()
    row_count, col_count = matrix.shape
    if n <= 0 or row_count <= 1 or col_count <= 1:
        return float("nan")
    denominator = n * min(row_count - 1, col_count - 1)
    return math.sqrt(max(chi2, 0.0) / denominator) if denominator > 0 else float("nan")


def compute_relation_metrics(
    encoded_df: pd.DataFrame,
    field_name: str,
    label_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    x = encoded_df[field_name].astype(str)
    y = encoded_df[label_name].astype(int)
    contingency = pd.crosstab(x, y)
    for label_value in (0, 1):
        if label_value not in contingency.columns:
            contingency[label_value] = 0
    contingency = contingency[[0, 1]].sort_index()

    n = int(contingency.to_numpy().sum())
    categories = int(contingency.shape[0])
    label_positive = int(contingency[1].sum())
    label_negative = int(contingency[0].sum())
    overall_positive_rate = safe_div(label_positive, n)

    if categories > 1 and label_positive > 0 and label_negative > 0:
        chi2, p_value, dof, expected = chi2_contingency(contingency.to_numpy(), correction=False)
        min_expected = float(np.min(expected))
        cramers_v = cramers_v_from_contingency(contingency, float(chi2))
    else:
        chi2 = float("nan")
        p_value = float("nan")
        dof = 0
        min_expected = float("nan")
        cramers_v = float("nan")

    mi = float(mutual_info_score(x, y))
    nmi = float(normalized_mutual_info_score(x, y))
    label_entropy = entropy_from_counts(np.array([label_negative, label_positive], dtype=float))
    mi_over_label_entropy = safe_div(mi, label_entropy)

    detail_rows: list[dict[str, Any]] = []
    max_abs_diff = 0.0
    weighted_abs_diff_sum = 0.0
    lifts: list[float] = []
    for category, row in contingency.iterrows():
        negative_count = int(row[0])
        positive_count = int(row[1])
        support = negative_count + positive_count
        target_rate = safe_div(positive_count, support)
        rate_diff = target_rate - overall_positive_rate
        lift = safe_div(target_rate, overall_positive_rate)
        if not math.isnan(rate_diff):
            max_abs_diff = max(max_abs_diff, abs(rate_diff))
            weighted_abs_diff_sum += abs(rate_diff) * support
        if not math.isnan(lift):
            lifts.append(lift)
        detail_rows.append(
            {
                "field": field_name,
                "label": label_name,
                "category": category,
                "support": support,
                "negative_count": negative_count,
                "positive_count": positive_count,
                "target_rate": target_rate,
                "overall_positive_rate": overall_positive_rate,
                "target_rate_diff": rate_diff,
                "abs_target_rate_diff": abs(rate_diff) if not math.isnan(rate_diff) else float("nan"),
                "lift": lift,
            }
        )

    summary = {
        "field": field_name,
        "label": label_name,
        "n": n,
        "category_count": categories,
        "label_positive": label_positive,
        "label_negative": label_negative,
        "overall_positive_rate": overall_positive_rate,
        "chi2": float(chi2),
        "chi2_p_value": float(p_value),
        "chi2_dof": int(dof),
        "chi2_min_expected": min_expected,
        "cramers_v": cramers_v,
        "mutual_info": mi,
        "normalized_mutual_info": nmi,
        "mutual_info_over_label_entropy": mi_over_label_entropy,
        "max_abs_target_rate_diff": max_abs_diff,
        "weighted_abs_target_rate_diff": safe_div(weighted_abs_diff_sum, n),
        "max_lift": max(lifts) if lifts else float("nan"),
        "min_lift": min(lifts) if lifts else float("nan"),
    }
    return summary, detail_rows, contingency


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: Any, digits: int = 6) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "nan"
    return f"{numeric:.{digits}g}"


def plot_metric_heatmap(summary_df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    pivot = summary_df.pivot(index="field", columns="label", values=metric).reindex(index=FIELD_ORDER, columns=LABEL_COLUMNS)
    values = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)

    fig_width = max(7.0, 1.8 * len(LABEL_COLUMNS))
    fig_height = max(4.0, 0.55 * len(FIELD_ORDER) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(masked, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(LABEL_COLUMNS)))
    ax.set_xticklabels(LABEL_COLUMNS, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(FIELD_ORDER)))
    ax.set_yticklabels(FIELD_ORDER)
    ax.set_title(title)

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if not np.isfinite(value):
                text = "NA"
            else:
                text = format_number(value, digits=3)
            ax.text(col_idx, row_idx, text, ha="center", va="center", color="white", fontsize=9)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_bars(summary_df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    labels = LABEL_COLUMNS
    x = np.arange(len(FIELD_ORDER))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5))
    for label_idx, label_name in enumerate(labels):
        values = []
        for field_name in FIELD_ORDER:
            matched = summary_df[(summary_df["field"] == field_name) & (summary_df["label"] == label_name)]
            values.append(float(matched.iloc[0][metric]) if len(matched) else float("nan"))
        ax.bar(x + (label_idx - 1) * width, values, width, label=label_name)

    ax.set_xticks(x)
    ax.set_xticklabels(FIELD_ORDER, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_category_target_rates(
    detail_df: pd.DataFrame,
    *,
    output_dir: Path,
    top_categories: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for field_name in FIELD_ORDER:
        for label_name in LABEL_COLUMNS:
            subset = detail_df[(detail_df["field"] == field_name) & (detail_df["label"] == label_name)].copy()
            if subset.empty:
                continue
            subset = subset.sort_values(["support", "category"], ascending=[False, True]).head(top_categories)
            subset = subset.sort_values("target_rate", ascending=True)

            categories = subset["category"].astype(str).tolist()
            y_pos = np.arange(len(categories))
            rates = subset["target_rate"].astype(float).to_numpy()
            supports = subset["support"].astype(int).to_numpy()
            overall = float(subset["overall_positive_rate"].iloc[0])

            fig_height = max(4.5, 0.32 * len(categories) + 1.5)
            fig, ax = plt.subplots(figsize=(9, fig_height))
            ax.barh(y_pos, rates, color="#4c78a8", alpha=0.85)
            ax.axvline(overall, color="#d62728", linestyle="--", linewidth=1.4, label="overall")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(categories, fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_xlabel("target positive rate")
            ax.set_title(f"{field_name} vs {label_name}")
            ax.legend(fontsize=8)
            ax.grid(axis="x", alpha=0.25)
            for idx, (rate, support) in enumerate(zip(rates, supports)):
                ax.text(min(rate + 0.015, 0.98), idx, f"n={support}", va="center", fontsize=7)
            fig.tight_layout()
            fig.savefig(output_dir / f"{field_name}__{label_name}__target_rate.png", dpi=180)
            plt.close(fig)


def write_markdown_summary(
    *,
    output_path: Path,
    report_csv: Path,
    datalist_csv: Path,
    row_count: int,
    min_category_count: int,
    field_audit_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    top_rows = summary_df.sort_values("cramers_v", ascending=False).head(10)
    lines = [
        f"# {EXPERIMENT_NAME} 实验结果说明",
        "",
        "## 输入",
        "",
        f"- 源报告总表：`{report_csv}`",
        f"- TASK2 datalist：`{datalist_csv}`",
        f"- 纳入样本数：`{row_count}`",
        f"- 分析字段：`{', '.join(FIELD_ORDER)}`",
        f"- 标签字段：`{', '.join(LABEL_COLUMNS)}`",
        "",
        "## 方法",
        "",
        "- `age` 先按年龄段分箱，再作为类别字段参与卡方检验、Cramér's V、互信息、目标率差异和 lift。",
        f"- 非缺失样本数小于 `{min_category_count}` 的类别会合并为 `{OTHER_TOKEN}`，用于降低低频类别造成的偶然相关。",
        f"- 空值统一编码为 `{MISSING_TOKEN}`，因此缺失本身也会参与关联分析。",
        "- 每个键会分别和三个二分类标签计算关联指标。",
        "",
        "## 输出文件",
        "",
        "- `field_audit.csv`：字段缺失率、唯一值数量、合并后的类别情况。",
        "- `key_label_summary.csv`：每个键-标签组合的卡方、Cramér's V、互信息、目标率差异和 lift 汇总。",
        "- `key_label_category_detail.csv`：每个类别下的标签阳性率、目标率差异和 lift。",
        "- `contingency_tables/`：每个键-标签组合的 2 列列联表。",
        "- `figures/`：热图、柱状图和类别目标率图。",
        "",
        "## 字段审计",
        "",
        "| 字段 | 缺失率 | 原始非缺失唯一值 | 合并后类别数 |",
        "|---|---:|---:|---:|",
    ]
    for _, row in field_audit_df.iterrows():
        lines.append(
            f"| `{row['field']}` | `{float(row['missing_rate']):.2%}` | "
            f"`{int(row['raw_unique_non_missing'])}` | `{int(row['encoded_unique'])}` |"
        )

    lines.extend(
        [
            "",
            "## Cramér's V 最高的键-标签组合",
            "",
            "| 字段 | 标签 | Cramér's V | p 值 | 互信息 | 最大目标率差异 | 最大 lift |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in top_rows.iterrows():
        lines.append(
            f"| `{row['field']}` | `{row['label']}` | `{format_number(row['cramers_v'])}` | "
            f"`{format_number(row['chi2_p_value'])}` | `{format_number(row['mutual_info'])}` | "
            f"`{format_number(row['max_abs_target_rate_diff'])}` | `{format_number(row['max_lift'])}` |"
        )

    lines.extend(
        [
            "",
            "## 注意",
            "",
            "- 这些统计只能说明字段与标签有关联，不能证明因果。",
            "- `operationValue` 可能反映检查/治疗路径，若关联很强，需要进一步做置乱实验或消融实验。",
            "- 本脚本不使用 `watchResult`，避免把标签来源直接作为输入字段。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "contingency_tables").mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"

    print("=" * 72)
    print(f"实验名称：{EXPERIMENT_NAME}")
    print(f"源报告总表：{args.report_csv}")
    print(f"TASK2 datalist：{args.datalist_csv}")
    print(f"输出目录：{output_dir}")

    merged_df = load_merged_rows(args.report_csv.expanduser().resolve(), args.datalist_csv.expanduser().resolve())
    encoded_df = merged_df.copy()
    for field_name in FIELD_ORDER:
        encoded_df[field_name] = encode_field(merged_df[field_name], field_name, args.min_category_count)

    field_audit_rows = build_field_audit(merged_df, encoded_df)
    field_audit_df = pd.DataFrame(field_audit_rows)
    field_audit_df.to_csv(output_dir / "field_audit.csv", index=False, encoding="utf-8-sig")

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    combinations = [(field_name, label_name) for field_name in FIELD_ORDER for label_name in LABEL_COLUMNS]
    for field_name, label_name in progress(combinations, total=len(combinations), desc="计算键-标签关联"):
        summary, detail, contingency = compute_relation_metrics(encoded_df, field_name, label_name)
        summary_rows.append(summary)
        detail_rows.extend(detail)
        contingency.to_csv(
            output_dir / "contingency_tables" / f"{field_name}__{label_name}.csv",
            encoding="utf-8-sig",
        )

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)
    summary_df.to_csv(output_dir / "key_label_summary.csv", index=False, encoding="utf-8-sig")
    detail_df.to_csv(output_dir / "key_label_category_detail.csv", index=False, encoding="utf-8-sig")

    if not args.no_plots:
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_metric_heatmap(summary_df, "cramers_v", figures_dir / "heatmap_cramers_v.png", "Cramer's V")
        plot_metric_heatmap(
            summary_df,
            "normalized_mutual_info",
            figures_dir / "heatmap_normalized_mutual_info.png",
            "Normalized Mutual Information",
        )
        plot_metric_heatmap(
            summary_df,
            "max_abs_target_rate_diff",
            figures_dir / "heatmap_max_abs_target_rate_diff.png",
            "Max Abs Target Rate Difference",
        )
        plot_metric_heatmap(summary_df, "max_lift", figures_dir / "heatmap_max_lift.png", "Max Lift")
        plot_metric_bars(summary_df, "cramers_v", figures_dir / "bar_cramers_v_by_label.png", "Cramer's V by label")
        plot_metric_bars(
            summary_df,
            "normalized_mutual_info",
            figures_dir / "bar_normalized_mutual_info_by_label.png",
            "Normalized Mutual Information by label",
        )
        plot_category_target_rates(
            detail_df,
            output_dir=figures_dir / "category_target_rate",
            top_categories=args.top_categories,
        )

    write_markdown_summary(
        output_path=output_dir / "README.md",
        report_csv=args.report_csv.expanduser().resolve(),
        datalist_csv=args.datalist_csv.expanduser().resolve(),
        row_count=len(merged_df),
        min_category_count=args.min_category_count,
        field_audit_df=field_audit_df,
        summary_df=summary_df,
    )

    print("\n完成。主要输出：")
    print(f"- 字段审计：{output_dir / 'field_audit.csv'}")
    print(f"- 汇总指标：{output_dir / 'key_label_summary.csv'}")
    print(f"- 类别明细：{output_dir / 'key_label_category_detail.csv'}")
    print(f"- 可视化目录：{figures_dir}")
    print(f"- 结果说明：{output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
