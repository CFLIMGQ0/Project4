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

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_CSV = PROJECT_ROOT / "datasets" / "valid_dicts_report_for task2.csv"
DEFAULT_DATALIST_CSV = PROJECT_ROOT / "datasets" / "task_data" / "task2" / "gastro_multilabel_task_datalist.csv"
EXPERIMENT_NAME = "key-key_cramers_basic_fields"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME

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
        description="年龄，脚本会先分箱再作为类别字段计算 Cramér's V",
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
    parser = argparse.ArgumentParser(description="分析 TASK2 多模态有效键之间的 Cramér's V 关联")
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV, help="源报告总表 CSV")
    parser.add_argument("--datalist-csv", type=Path, default=DEFAULT_DATALIST_CSV, help="TASK2 三标签 datalist CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="实验输出目录")
    parser.add_argument("--min-category-count", type=int, default=20, help="低于该样本数的类别合并为 __OTHER_LOW_FREQ__")
    parser.add_argument("--no-plots", action="store_true", help="只输出 CSV，不生成热力图")
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

    report_keep_cols = ["exam_dir"]
    for field_name in FIELD_ORDER:
        column = resolve_column(report_df, FIELD_SPECS[field_name].source_aliases)
        if column is not None and column not in report_keep_cols:
            report_keep_cols.append(column)

    merged = datalist_df[["exam_dir"]].merge(
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

        if datalist_col is not None and datalist_col in datalist_df.columns:
            base_values = datalist_df[datalist_col].astype(str).reset_index(drop=True)
            if report_col is not None:
                candidate_col = report_col if report_col in merged.columns else f"{report_col}__report"
                if candidate_col in merged.columns:
                    report_values = merged[candidate_col].astype(str).reset_index(drop=True)
                    base_values = base_values.where(base_values.str.strip() != "", report_values)
            merged[field_name] = base_values
        elif report_col is not None and report_col in merged.columns:
            merged[field_name] = merged[report_col].astype(str)
        else:
            merged[field_name] = ""

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


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def cramers_v_from_contingency(contingency: pd.DataFrame, chi2: float) -> float:
    matrix = contingency.to_numpy(dtype=float)
    n = matrix.sum()
    row_count, col_count = matrix.shape
    if n <= 0 or row_count <= 1 or col_count <= 1:
        return float("nan")
    denominator = n * min(row_count - 1, col_count - 1)
    return math.sqrt(max(float(chi2), 0.0) / denominator) if denominator > 0 else float("nan")


def compute_pair_metrics(encoded_df: pd.DataFrame, field_a: str, field_b: str) -> tuple[dict[str, Any], pd.DataFrame]:
    contingency = pd.crosstab(encoded_df[field_a].astype(str), encoded_df[field_b].astype(str)).sort_index()
    n = int(contingency.to_numpy().sum())
    row_count, col_count = contingency.shape

    if field_a == field_b:
        chi2 = float("nan")
        p_value = float("nan")
        dof = 0
        min_expected = float("nan")
        cramers_v = 1.0
    elif row_count > 1 and col_count > 1:
        chi2, p_value, dof, expected = chi2_contingency(contingency.to_numpy(), correction=False)
        min_expected = float(np.min(expected))
        cramers_v = cramers_v_from_contingency(contingency, float(chi2))
    else:
        chi2 = float("nan")
        p_value = float("nan")
        dof = 0
        min_expected = float("nan")
        cramers_v = float("nan")

    summary = {
        "field_a": field_a,
        "field_b": field_b,
        "n": n,
        "field_a_category_count": row_count,
        "field_b_category_count": col_count,
        "chi2": float(chi2),
        "chi2_p_value": float(p_value),
        "chi2_dof": int(dof),
        "chi2_min_expected": min_expected,
        "cramers_v": cramers_v,
    }
    return summary, contingency


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


def format_number(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def build_matrix(summary_df: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.DataFrame(index=FIELD_ORDER, columns=FIELD_ORDER, dtype=float)
    for _, row in summary_df.iterrows():
        matrix.loc[row["field_a"], row["field_b"]] = float(row["cramers_v"])
    for field_name in FIELD_ORDER:
        matrix.loc[field_name, field_name] = 1.0
    return matrix


def plot_cramers_heatmap(matrix: pd.DataFrame, output_path: Path) -> None:
    values = matrix.reindex(index=FIELD_ORDER, columns=FIELD_ORDER).to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    image = ax.imshow(masked, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(np.arange(len(FIELD_ORDER)))
    ax.set_yticks(np.arange(len(FIELD_ORDER)))
    ax.set_xticklabels(FIELD_ORDER, rotation=35, ha="right")
    ax.set_yticklabels(FIELD_ORDER)
    ax.set_title("键间关联强度：Cramér's V")

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            text = format_number(value)
            color = "white" if np.isfinite(value) and value >= 0.55 else "black"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color=color, fontsize=10)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Cramér's V")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
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
    top_pairs = (
        summary_df[summary_df["field_a"] != summary_df["field_b"]]
        .sort_values("cramers_v", ascending=False)
        .head(10)
    )
    lines = [
        f"# {EXPERIMENT_NAME} 实验结果说明",
        "",
        "## 输入",
        "",
        f"- 源报告总表：`{report_csv}`",
        f"- TASK2 datalist：`{datalist_csv}`",
        f"- 纳入样本数：`{row_count}`",
        f"- 分析字段：`{', '.join(FIELD_ORDER)}`",
        "",
        "## 方法",
        "",
        "- 本实验只分析五个多模态有效键之间的关联，不使用标签字段。",
        "- `age` 先按年龄段分箱，再作为类别字段参与 Cramér's V 计算。",
        f"- 非缺失样本数小于 `{min_category_count}` 的类别会合并为 `{OTHER_TOKEN}`，降低低频类别造成的偶然相关。",
        f"- 空值统一编码为 `{MISSING_TOKEN}`，因此缺失本身也会参与键间关联分析。",
        "- 热力图固定为正方形，颜色从黄色到红色，越红表示 Cramér's V 越高。",
        "",
        "## 输出文件",
        "",
        "- `field_audit.csv`：字段缺失率、唯一值数量、合并后的类别情况。",
        "- `key_key_cramers_summary.csv`：每个键-键组合的 Cramér's V、卡方统计量和 p 值。",
        "- `key_key_cramers_matrix.csv`：Cramér's V 方阵。",
        "- `contingency_tables/`：每个键-键组合的列联表。",
        "- `figures/heatmap_cramers_v_square.png`：正方形 Cramér's V 热力图。",
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
            "## Cramér's V 最高的键对",
            "",
            "| 字段 A | 字段 B | Cramér's V | p 值 | A 类别数 | B 类别数 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in top_pairs.iterrows():
        lines.append(
            f"| `{row['field_a']}` | `{row['field_b']}` | `{format_number(row['cramers_v'], 6)}` | "
            f"`{format_number(row['chi2_p_value'], 6)}` | `{int(row['field_a_category_count'])}` | "
            f"`{int(row['field_b_category_count'])}` |"
        )

    lines.extend(
        [
            "",
            "## 注意",
            "",
            "- Cramér's V 只表示关联强度，不表示因果关系。",
            "- 如果 `reportTitle` 与 `operationValue` 关联很高，后续多模态建模建议做二者消融，避免重复输入同一检查流程信息。",
            "- `age` 在这里是分箱后的类别变量，因此结果代表年龄段与其他字段之间的关联。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    contingency_dir = output_dir / "contingency_tables"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    contingency_dir.mkdir(parents=True, exist_ok=True)

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

    pair_rows: list[dict[str, Any]] = []
    pairs = [(field_a, field_b) for field_a in FIELD_ORDER for field_b in FIELD_ORDER]
    for field_a, field_b in progress(pairs, total=len(pairs), desc="计算键-键 Cramér's V"):
        summary, contingency = compute_pair_metrics(encoded_df, field_a, field_b)
        pair_rows.append(summary)
        contingency.to_csv(contingency_dir / f"{field_a}__{field_b}.csv", encoding="utf-8-sig")

    summary_df = pd.DataFrame(pair_rows)
    summary_df.to_csv(output_dir / "key_key_cramers_summary.csv", index=False, encoding="utf-8-sig")

    matrix_df = build_matrix(summary_df)
    matrix_df.to_csv(output_dir / "key_key_cramers_matrix.csv", encoding="utf-8-sig")

    if not args.no_plots:
        plot_cramers_heatmap(matrix_df, figures_dir / "heatmap_cramers_v_square.png")

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
    print(f"- 键间汇总：{output_dir / 'key_key_cramers_summary.csv'}")
    print(f"- Cramér's V 方阵：{output_dir / 'key_key_cramers_matrix.csv'}")
    print(f"- 正方形热力图：{figures_dir / 'heatmap_cramers_v_square.png'}")
    print(f"- 结果说明：{output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
