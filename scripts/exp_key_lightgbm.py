from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = SRC_ROOT / "scripts" / "key-label_basic_fields.py"
DEFAULT_REPORT_CSV = PROJECT_ROOT / "datasets" / "valid_dicts_report_for task2.csv"
DEFAULT_DATALIST_CSV = PROJECT_ROOT / "datasets" / "task_data" / "task2" / "gastro_multilabel_task_datalist.csv"
EXPERIMENT_NAME = "exp_key_lightgbm"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME

LABEL_COLUMNS = [
    "label_esophageal_smt",
    "label_esophageal_mucosal_or_tumor",
    "label_gastritis",
]
FIELD_ORDER = ["reportTitle", "age", "sex", "hp", "operationValue"]
FIELD_TO_COLUMNS = {
    "reportTitle": ["reportTitle"],
    "age": ["age", "age_missing"],
    "sex": ["sex"],
    "hp": ["hp"],
    "operationValue": ["operationValue"],
}
MISSING_TOKEN = "__MISSING__"
OTHER_TOKEN = "__OTHER_LOW_FREQ__"


@dataclass(frozen=True)
class RunConfig:
    folds: int
    n_estimators: int
    early_stopping_rounds: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    min_child_samples: int
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    permutation_repeats: int
    min_category_count: int
    max_samples: int
    seed: int
    n_jobs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="exp_key_lightgbm：用 LightGBM 快速分析 TASK2 关键结构化字段对三标签分类的影响"
    )
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV, help="源报告总表 CSV")
    parser.add_argument("--datalist-csv", type=Path, default=DEFAULT_DATALIST_CSV, help="TASK2 三标签 datalist CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="实验输出目录")
    parser.add_argument("--folds", type=int, default=3, help="交叉验证折数；默认 3，避免运行过久")
    parser.add_argument("--n-estimators", type=int, default=80, help="每个 LightGBM 二分类器最多树数")
    parser.add_argument("--early-stopping-rounds", type=int, default=10, help="早停轮数")
    parser.add_argument("--learning-rate", type=float, default=0.06, help="学习率")
    parser.add_argument("--num-leaves", type=int, default=15, help="叶子数上限，默认较小以减少过拟合和耗时")
    parser.add_argument("--max-depth", type=int, default=4, help="树深度上限")
    parser.add_argument("--min-child-samples", type=int, default=30, help="叶节点最小样本数")
    parser.add_argument("--subsample", type=float, default=0.9, help="行采样比例")
    parser.add_argument("--colsample-bytree", type=float, default=1.0, help="列采样比例")
    parser.add_argument("--reg-lambda", type=float, default=1.0, help="L2 正则")
    parser.add_argument("--permutation-repeats", type=int, default=1, help="置乱重要性重复次数；默认 1，控制耗时")
    parser.add_argument("--min-category-count", type=int, default=20, help="低频类别合并阈值")
    parser.add_argument("--max-samples", type=int, default=0, help="可选抽样上限；0 表示使用全部样本")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    parser.add_argument("--n-jobs", type=int, default=2, help="LightGBM 线程数，默认 2，避免占满机器")
    parser.add_argument("--dry-run", action="store_true", help="只检查数据和输出字段审计，不训练 LightGBM")
    return parser.parse_args()


def load_helper_module() -> Any:
    spec = importlib.util.spec_from_file_location("key_label_basic_fields_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载字段合并辅助脚本：{HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_lightgbm() -> tuple[Any, Any, Any]:
    try:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "当前 Python 环境没有安装 lightgbm。请先安装后再运行：python -m pip install lightgbm"
        ) from exc
    return LGBMClassifier, early_stopping, log_evaluation


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
    return " ".join(cleaned.split()) if cleaned else MISSING_TOKEN


def parse_age(value: Any) -> float:
    cleaned = normalize_text(value)
    if not cleaned:
        return float("nan")
    digits = []
    dot_seen = False
    for char in cleaned:
        if char.isdigit():
            digits.append(char)
        elif char == "." and not dot_seen:
            digits.append(char)
            dot_seen = True
        elif digits:
            break
    if not digits:
        return float("nan")
    try:
        age = float("".join(digits))
    except ValueError:
        return float("nan")
    if age < 0 or age > 120:
        return float("nan")
    return age


def merge_low_frequency_categories(series: pd.Series, min_category_count: int) -> pd.Series:
    encoded = series.map(normalize_category)
    counts = encoded.value_counts(dropna=False)
    low_freq = {name for name, count in counts.items() if name != MISSING_TOKEN and count < min_category_count}
    if low_freq:
        encoded = encoded.map(lambda value: OTHER_TOKEN if value in low_freq else value)
    return encoded.astype(str)


def prepare_data(
    merged_df: pd.DataFrame,
    *,
    min_category_count: int,
    max_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    work_df = merged_df.copy()
    if max_samples > 0 and len(work_df) > max_samples:
        work_df = work_df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    features = pd.DataFrame(index=work_df.index)
    age_raw = work_df["age"].map(parse_age).astype(float)
    age_median = float(age_raw.dropna().median()) if age_raw.notna().any() else 0.0
    features["age"] = age_raw.fillna(age_median)
    features["age_missing"] = age_raw.isna().astype(int)

    categorical_features = ["reportTitle", "sex", "hp", "operationValue"]
    for field_name in categorical_features:
        features[field_name] = merge_low_frequency_categories(work_df[field_name], min_category_count).astype("category")

    labels = work_df[LABEL_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).clip(0, 1)
    groups = work_df["patient_id"].astype(str) if "patient_id" in work_df.columns else work_df["exam_dir"].astype(str)

    audit_rows = []
    for field_name in FIELD_ORDER:
        raw = work_df[field_name].map(normalize_text)
        non_missing = raw[raw != ""]
        if field_name == "age":
            encoded_unique = int(features[["age", "age_missing"]].drop_duplicates().shape[0])
        else:
            encoded_unique = int(features[field_name].nunique(dropna=False))
        audit_rows.append(
            {
                "field": field_name,
                "total": int(len(work_df)),
                "non_missing": int(len(non_missing)),
                "missing": int(len(work_df) - len(non_missing)),
                "missing_rate": float((len(work_df) - len(non_missing)) / len(work_df)) if len(work_df) else float("nan"),
                "raw_unique_non_missing": int(non_missing.nunique(dropna=True)),
                "encoded_unique": encoded_unique,
            }
        )
    field_audit = pd.DataFrame(audit_rows)
    return features, labels, groups, field_audit


def make_splits(
    y: pd.Series,
    groups: pd.Series,
    *,
    requested_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    y_array = y.to_numpy(dtype=int)
    group_array = groups.to_numpy()
    positives = int(y_array.sum())
    negatives = int(len(y_array) - positives)
    folds = max(2, min(requested_folds, positives, negatives))

    if folds >= 2:
        try:
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
            splits = list(splitter.split(np.zeros(len(y_array)), y_array, group_array))
            if splits:
                return [(train_idx, valid_idx) for train_idx, valid_idx in splits]
        except ValueError:
            pass

        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        return [(train_idx, valid_idx) for train_idx, valid_idx in splitter.split(np.zeros(len(y_array)), y_array)]

    train_idx, valid_idx = train_test_split(
        np.arange(len(y_array)),
        test_size=0.25,
        random_state=seed,
        stratify=y_array if positives > 0 and negatives > 0 else None,
    )
    return [(np.asarray(train_idx), np.asarray(valid_idx))]


def compute_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    has_two_classes = len(np.unique(y_true)) == 2
    return {
        "auc": float(roc_auc_score(y_true, prob)) if has_two_classes else float("nan"),
        "ap": float(average_precision_score(y_true, prob)) if has_two_classes else float("nan"),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)) if has_two_classes else float("nan"),
    }


def aggregate_importance(
    *,
    label_name: str,
    fold: int,
    feature_names: list[str],
    gain_values: np.ndarray,
    split_values: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    gain_by_name = dict(zip(feature_names, gain_values.tolist()))
    split_by_name = dict(zip(feature_names, split_values.tolist()))
    for field_name, columns in FIELD_TO_COLUMNS.items():
        rows.append(
            {
                "label": label_name,
                "fold": fold,
                "field": field_name,
                "gain_importance": float(sum(gain_by_name.get(column, 0.0) for column in columns)),
                "split_importance": float(sum(split_by_name.get(column, 0.0) for column in columns)),
            }
        )
    return rows


def train_one_label(
    *,
    label_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cfg: RunConfig,
    lightgbm_api: tuple[Any, Any, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    LGBMClassifier, early_stopping, log_evaluation = lightgbm_api
    metric_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    categorical_feature = ["reportTitle", "sex", "hp", "operationValue"]

    splits = make_splits(y, groups, requested_folds=cfg.folds, seed=cfg.seed)
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        x_train = x.iloc[train_idx].copy()
        x_valid = x.iloc[valid_idx].copy()
        y_train = y.iloc[train_idx].to_numpy(dtype=int)
        y_valid = y.iloc[valid_idx].to_numpy(dtype=int)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_valid)) < 2:
            continue

        positive_count = int(y_train.sum())
        negative_count = int(len(y_train) - positive_count)
        scale_pos_weight = float(negative_count / positive_count) if positive_count > 0 else 1.0

        model = LGBMClassifier(
            objective="binary",
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            max_depth=cfg.max_depth,
            min_child_samples=cfg.min_child_samples,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda,
            scale_pos_weight=scale_pos_weight,
            random_state=cfg.seed + fold,
            n_jobs=cfg.n_jobs,
            verbosity=-1,
            force_col_wise=True,
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="auc",
            categorical_feature=categorical_feature,
            callbacks=[
                early_stopping(cfg.early_stopping_rounds, verbose=False),
                log_evaluation(period=0),
            ],
        )

        prob = model.predict_proba(x_valid)[:, 1]
        metrics = compute_metrics(y_valid, prob)
        metric_rows.append(
            {
                "label": label_name,
                "fold": fold,
                "train_n": int(len(train_idx)),
                "valid_n": int(len(valid_idx)),
                "valid_positive": int(y_valid.sum()),
                "valid_positive_rate": float(y_valid.mean()),
                "best_iteration": int(getattr(model, "best_iteration_", 0) or cfg.n_estimators),
                **metrics,
            }
        )

        booster = model.booster_
        feature_names = list(booster.feature_name())
        gain_values = booster.feature_importance(importance_type="gain")
        split_values = booster.feature_importance(importance_type="split")
        importance_rows.extend(
            aggregate_importance(
                label_name=label_name,
                fold=fold,
                feature_names=feature_names,
                gain_values=gain_values,
                split_values=split_values,
            )
        )

        rng = np.random.default_rng(cfg.seed + 1000 * fold)
        for field_name, columns in FIELD_TO_COLUMNS.items():
            for repeat in range(1, cfg.permutation_repeats + 1):
                x_perm = x_valid.copy()
                for column in columns:
                    shuffled = rng.permutation(x_perm[column].to_numpy())
                    if isinstance(x_valid[column].dtype, pd.CategoricalDtype):
                        x_perm[column] = pd.Categorical(shuffled, categories=x_valid[column].cat.categories)
                    else:
                        x_perm[column] = shuffled
                perm_prob = model.predict_proba(x_perm)[:, 1]
                perm_metrics = compute_metrics(y_valid, perm_prob)
                permutation_rows.append(
                    {
                        "label": label_name,
                        "fold": fold,
                        "field": field_name,
                        "repeat": repeat,
                        "baseline_auc": metrics["auc"],
                        "permuted_auc": perm_metrics["auc"],
                        "auc_drop": metrics["auc"] - perm_metrics["auc"],
                        "baseline_ap": metrics["ap"],
                        "permuted_ap": perm_metrics["ap"],
                        "ap_drop": metrics["ap"] - perm_metrics["ap"],
                        "baseline_f1": metrics["f1"],
                        "permuted_f1": perm_metrics["f1"],
                        "f1_drop": metrics["f1"] - perm_metrics["f1"],
                    }
                )
    return metric_rows, importance_rows, permutation_rows


def summarize_results(
    metric_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_metrics = (
        metric_df.groupby("label", as_index=False)
        .agg(
            folds=("fold", "count"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            ap_mean=("ap", "mean"),
            ap_std=("ap", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            valid_positive_rate_mean=("valid_positive_rate", "mean"),
            best_iteration_mean=("best_iteration", "mean"),
        )
        .sort_values("label")
    )

    importance_summary = (
        importance_df.groupby(["label", "field"], as_index=False)
        .agg(
            gain_importance_mean=("gain_importance", "mean"),
            split_importance_mean=("split_importance", "mean"),
        )
    )
    permutation_summary = (
        permutation_df.groupby(["label", "field"], as_index=False)
        .agg(
            auc_drop_mean=("auc_drop", "mean"),
            auc_drop_std=("auc_drop", "std"),
            ap_drop_mean=("ap_drop", "mean"),
            ap_drop_std=("ap_drop", "std"),
            f1_drop_mean=("f1_drop", "mean"),
            f1_drop_std=("f1_drop", "std"),
        )
    )
    field_importance = importance_summary.merge(permutation_summary, on=["label", "field"], how="outer")
    field_importance["rank_by_auc_drop"] = (
        field_importance.groupby("label")["auc_drop_mean"].rank(method="dense", ascending=False).astype("Int64")
    )
    field_importance["rank_by_gain"] = (
        field_importance.groupby("label")["gain_importance_mean"].rank(method="dense", ascending=False).astype("Int64")
    )
    field_importance = field_importance.sort_values(["label", "rank_by_auc_drop", "rank_by_gain", "field"])
    return label_metrics, field_importance


def format_float(value: Any, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "nan"
    return f"{numeric:.{digits}f}"


def write_readme(
    *,
    output_path: Path,
    args: argparse.Namespace,
    cfg: RunConfig,
    row_count: int,
    field_audit: pd.DataFrame,
    label_metrics: pd.DataFrame | None,
    field_importance: pd.DataFrame | None,
    dry_run: bool,
) -> None:
    lines = [
        f"# {EXPERIMENT_NAME} 实验说明",
        "",
        "## 目的",
        "",
        "使用 LightGBM 对 TASK2 的 5 个结构化字段做快速机器学习重要性分析，不使用图像，也不使用 `watchResult`。",
        "",
        "## 输入字段",
        "",
        "| 字段 | 处理方式 |",
        "|---|---|",
        "| `age` | 解析为连续年龄，并加入 `age_missing` 缺失标记 |",
        "| `sex` | 类别特征，低频类别合并 |",
        "| `hp` | 类别特征，低频类别合并 |",
        "| `reportTitle` | 类别特征，低频类别合并 |",
        "| `operationValue` | 类别特征，低频类别合并 |",
        "",
        "## 标签",
        "",
        "- `label_esophageal_smt`",
        "- `label_esophageal_mucosal_or_tumor`",
        "- `label_gastritis`",
        "",
        "## 轻量运行设置",
        "",
        f"- folds：`{cfg.folds}`",
        f"- n_estimators：`{cfg.n_estimators}`",
        f"- early_stopping_rounds：`{cfg.early_stopping_rounds}`",
        f"- num_leaves / max_depth：`{cfg.num_leaves}` / `{cfg.max_depth}`",
        f"- permutation_repeats：`{cfg.permutation_repeats}`",
        f"- n_jobs：`{cfg.n_jobs}`",
        f"- 实际样本数：`{row_count}`",
        "",
        "## 输出文件",
        "",
        "- `field_audit.csv`：字段缺失率与唯一值数量。",
        "- `fold_metrics.csv`：每个标签每一折的 LightGBM 指标。",
        "- `label_metrics.csv`：每个标签的平均 AUC、AP、F1 等指标。",
        "- `lightgbm_importance_detail.csv`：每折 LightGBM gain/split 重要性。",
        "- `permutation_importance_detail.csv`：每折字段置乱后的 AUC/AP/F1 下降。",
        "- `field_importance_by_label.csv`：按标签汇总后的字段重要性结果，建议优先看 `auc_drop_mean`。",
        "",
        "## 字段审计",
        "",
        "| 字段 | 缺失率 | 原始非缺失唯一值 | 编码后唯一值 |",
        "|---|---:|---:|---:|",
    ]
    for _, row in field_audit.iterrows():
        lines.append(
            f"| `{row['field']}` | `{float(row['missing_rate']):.2%}` | "
            f"`{int(row['raw_unique_non_missing'])}` | `{int(row['encoded_unique'])}` |"
        )

    if dry_run:
        lines.extend(
            [
                "",
                "## 当前状态",
                "",
                "本次使用 `--dry-run`，只完成字段审计，没有训练 LightGBM。",
                "",
            ]
        )
    elif label_metrics is not None and field_importance is not None:
        lines.extend(
            [
                "",
                "## 标签级模型表现",
                "",
                "| 标签 | AUC | AP | F1 | balanced accuracy | 平均最佳迭代 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in label_metrics.iterrows():
            lines.append(
                f"| `{row['label']}` | `{format_float(row['auc_mean'])}` | `{format_float(row['ap_mean'])}` | "
                f"`{format_float(row['f1_mean'])}` | `{format_float(row['balanced_accuracy_mean'])}` | "
                f"`{format_float(row['best_iteration_mean'], digits=1)}` |"
            )

        lines.extend(
            [
                "",
                "## 各标签 AUC 置乱下降 Top 字段",
                "",
                "| 标签 | 字段 | AUC drop | AP drop | F1 drop | gain importance |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for label_name in LABEL_COLUMNS:
            subset = field_importance[field_importance["label"] == label_name].sort_values(
                ["auc_drop_mean", "gain_importance_mean"], ascending=[False, False]
            )
            for _, row in subset.head(5).iterrows():
                lines.append(
                    f"| `{row['label']}` | `{row['field']}` | `{format_float(row['auc_drop_mean'])}` | "
                    f"`{format_float(row['ap_drop_mean'])}` | `{format_float(row['f1_drop_mean'])}` | "
                    f"`{format_float(row['gain_importance_mean'])}` |"
                )

    lines.extend(
        [
            "",
            "## 注意",
            "",
            "- 这是结构化字段的初步机器学习关联分析，只能说明字段对预测有帮助，不能证明因果。",
            "- `operationValue` 和 `reportTitle` 可能包含检查流程代理信息，若重要性很高，论文中需要单独说明并结合消融/置乱结果解释。",
            "- 本实验不读取图像，不调用 `train.py`，不会影响 `exp_8` 或 `exp_mm_ablation_hypergraph` 的训练输出。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = RunConfig(
        folds=args.folds,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        permutation_repeats=args.permutation_repeats,
        min_category_count=args.min_category_count,
        max_samples=args.max_samples,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print(f"实验名称：{EXPERIMENT_NAME}")
    print(f"源报告总表：{args.report_csv}")
    print(f"TASK2 datalist：{args.datalist_csv}")
    print(f"输出目录：{output_dir}")
    print("纳入字段：", ", ".join(FIELD_ORDER))

    helper = load_helper_module()
    merged_df = helper.load_merged_rows(
        args.report_csv.expanduser().resolve(),
        args.datalist_csv.expanduser().resolve(),
    )
    x, y_df, groups, field_audit = prepare_data(
        merged_df,
        min_category_count=cfg.min_category_count,
        max_samples=cfg.max_samples,
        seed=cfg.seed,
    )
    field_audit.to_csv(output_dir / "field_audit.csv", index=False, encoding="utf-8-sig")

    if args.dry_run:
        write_readme(
            output_path=output_dir / "README.md",
            args=args,
            cfg=cfg,
            row_count=len(x),
            field_audit=field_audit,
            label_metrics=None,
            field_importance=None,
            dry_run=True,
        )
        print("\n已完成 dry-run。主要输出：")
        print(f"- 字段审计：{output_dir / 'field_audit.csv'}")
        print(f"- 结果说明：{output_dir / 'README.md'}")
        return

    lightgbm_api = require_lightgbm()
    metric_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []

    label_iter = progress(LABEL_COLUMNS, total=len(LABEL_COLUMNS), desc="训练 LightGBM 标签模型")
    for label_name in label_iter:
        label_metric_rows, label_importance_rows, label_permutation_rows = train_one_label(
            label_name=label_name,
            x=x,
            y=y_df[label_name],
            groups=groups,
            cfg=cfg,
            lightgbm_api=lightgbm_api,
        )
        metric_rows.extend(label_metric_rows)
        importance_rows.extend(label_importance_rows)
        permutation_rows.extend(label_permutation_rows)

    metric_df = pd.DataFrame(metric_rows)
    importance_df = pd.DataFrame(importance_rows)
    permutation_df = pd.DataFrame(permutation_rows)
    if metric_df.empty:
        raise RuntimeError("没有得到有效 LightGBM 训练折，请检查标签分布和折数设置。")

    label_metrics, field_importance = summarize_results(metric_df, importance_df, permutation_df)
    metric_df.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    label_metrics.to_csv(output_dir / "label_metrics.csv", index=False, encoding="utf-8-sig")
    importance_df.to_csv(output_dir / "lightgbm_importance_detail.csv", index=False, encoding="utf-8-sig")
    permutation_df.to_csv(output_dir / "permutation_importance_detail.csv", index=False, encoding="utf-8-sig")
    field_importance.to_csv(output_dir / "field_importance_by_label.csv", index=False, encoding="utf-8-sig")

    write_readme(
        output_path=output_dir / "README.md",
        args=args,
        cfg=cfg,
        row_count=len(x),
        field_audit=field_audit,
        label_metrics=label_metrics,
        field_importance=field_importance,
        dry_run=False,
    )

    print("\n完成。主要输出：")
    print(f"- 字段审计：{output_dir / 'field_audit.csv'}")
    print(f"- 标签指标：{output_dir / 'label_metrics.csv'}")
    print(f"- 字段重要性：{output_dir / 'field_importance_by_label.csv'}")
    print(f"- 结果说明：{output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
