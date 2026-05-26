from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks import get_task_spec, list_task_specs
from tasks.common import SelectionResult, load_rows, resolve_compatible_path, write_csv
from tasks.task1 import build_selection_result as build_task1_selection_result
from tasks.task2 import build_selection_result as build_task2_selection_result


TASK_BUILDERS: dict[str, Callable[[list[dict[str, str]], dict[str, str | None]], SelectionResult]] = {
    "task1": build_task1_selection_result,
    "task2": build_task2_selection_result,
}


def parse_args(default_tasks: str = "task1") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按任务生成任务样本表")
    parser.add_argument("--config", type=Path, default=None, help="路径配置文件，默认读取 configs/<task>/path.yaml")
    parser.add_argument("--task-config", type=Path, default=None, help="单任务模式下覆盖 data.yaml 路径")
    parser.add_argument("--tasks", type=str, default=default_tasks, help="要生成的任务，逗号分隔")
    parser.add_argument("--report-csv", type=Path, default=None, help="仅单任务模式可用：覆盖该任务的报告 CSV")
    parser.add_argument("--output-dir", type=Path, default=None, help="覆盖任务数据根目录，默认 datasets/task_data")
    return parser.parse_args()


def resolve_config_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def load_path_config(config_path: Path) -> dict[str, Path]:
    if not config_path.is_file():
        raise FileNotFoundError(f"未找到基础路径配置文件：{config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("基础路径配置文件格式错误")
    paths = payload.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("基础路径配置文件缺少 paths 分组")

    config_dir = config_path.resolve().parent
    resolved: dict[str, Path] = {}
    for key in ("project_root", "dataset_base_root", "dataset_root", "output_dir"):
        raw = paths.get(key)
        if raw is not None and str(raw).strip():
            resolved[key] = resolve_config_path(str(raw), config_dir)

    if "dataset_base_root" not in resolved:
        raise ValueError("paths.dataset_base_root 不能为空")
    if "output_dir" not in resolved:
        raise ValueError("paths.output_dir 不能为空")
    return resolved


def resolve_default_path_config(task_name: str) -> Path:
    spec = get_task_spec(task_name)
    return (ROOT / "configs" / spec.name / "path.yaml").resolve()


def resolve_default_task_config(task_name: str) -> Path:
    spec = get_task_spec(task_name)
    return (ROOT / "configs" / spec.name / "data.yaml").resolve()


def load_task_config(task_name: str, task_config_override: Path | None = None) -> dict[str, Any]:
    config_path = (
        task_config_override.expanduser().resolve()
        if task_config_override is not None
        else resolve_default_task_config(task_name)
    )
    if not config_path.is_file():
        return {}
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"任务配置文件格式错误：{config_path}")
    return payload


def resolve_report_csv_path(
    *,
    task_name: str,
    task_payload: dict[str, Any],
    path_cfg: dict[str, Path],
    report_csv_override: Path | None,
) -> Path:
    if report_csv_override is not None:
        return report_csv_override.expanduser().resolve()

    spec = get_task_spec(task_name)
    raw = None
    data_payload = task_payload.get("data", {})
    if isinstance(data_payload, dict):
        raw = data_payload.get("report_csv")
    if raw is None:
        raw = spec.default_report_csv
    if not raw:
        raise ValueError(f"任务 {task_name} 未配置 report_csv")
    return resolve_compatible_path(resolve_config_path(str(raw), path_cfg["dataset_base_root"]))


def resolve_tasks_argument(raw_tasks: str) -> list[str]:
    requested = [item.strip() for item in str(raw_tasks).split(",") if item.strip()]
    if not requested or requested == ["all"]:
        return [item.name for item in list_task_specs()]
    task_names = []
    for task_name in requested:
        get_task_spec(task_name)
        task_names.append(task_name)
    return task_names


def build_task_summary_markdown(
    *,
    task_name: str,
    report_csv_path: Path,
    datalist_path: Path,
    result: SelectionResult,
) -> str:
    spec = get_task_spec(task_name)
    positive_lines = [f"- `{key}`：`{value}`" for key, value in sorted(result.positive_counter.items())] or ["- 无"]
    exclude_lines = [f"- `{key}`：`{value}`" for key, value in sorted(result.exclude_counter.items())] or ["- 无"]
    lines = [
        f"# {spec.display_name} 数据摘要",
        "",
        f"- 任务名：`{spec.name}`",
        f"- 报告 CSV：`{report_csv_path}`",
        f"- 输出 datalist：`{datalist_path}`",
        f"- 候选记录数：`{result.total_candidates}`",
        f"- 纳入记录数：`{result.selected_count}`",
        f"- 纳入图像总数：`{result.selected_image_sum}`",
        "",
        "## 标签/类别统计",
        "",
        *positive_lines,
        "",
        "## 主要剔除原因",
        "",
        *exclude_lines,
        "",
    ]
    return "\n".join(lines)


def main(default_tasks: str = "task1") -> None:
    args = parse_args(default_tasks=default_tasks)
    task_names = resolve_tasks_argument(args.tasks)
    default_path_config = args.config.expanduser().resolve() if args.config is not None else resolve_default_path_config(task_names[0])
    path_cfg = load_path_config(default_path_config)

    if args.report_csv is not None and len(task_names) != 1:
        raise ValueError("--report-csv 仅支持单任务模式")
    if args.task_config is not None and len(task_names) != 1:
        raise ValueError("--task-config 仅支持单任务模式")

    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (path_cfg["dataset_base_root"] / "task_data").resolve()
    )

    print("=" * 72)
    print("任务数据生成")
    print(f"任务={','.join(task_names)}")
    print(f"输出根目录={output_root}")

    for task_name in task_names:
        spec = get_task_spec(task_name)
        task_payload = load_task_config(task_name, task_config_override=args.task_config)
        report_csv_path = resolve_report_csv_path(
            task_name=task_name,
            task_payload=task_payload,
            path_cfg=path_cfg,
            report_csv_override=args.report_csv,
        )

        rows, columns = load_rows(report_csv_path)
        result = TASK_BUILDERS[task_name](rows, columns)

        task_dir = output_root / spec.data_subdir
        datalist_path = task_dir / spec.datalist_filename
        summary_path = task_dir / "README.md"
        write_csv(datalist_path, result.rows, result.fieldnames)
        summary_path.write_text(
            build_task_summary_markdown(
                task_name=task_name,
                report_csv_path=report_csv_path,
                datalist_path=datalist_path,
                result=result,
            ),
            encoding="utf-8",
        )

        print(f"\n[{task_name}]")
        print(f"候选记录数：{result.total_candidates}")
        print(f"纳入记录数：{result.selected_count}")
        print(f"纳入图像数：{result.selected_image_sum}")
        print(f"datalist：{datalist_path}")
        print(f"摘要文档：{summary_path}")


if __name__ == "__main__":
    main(default_tasks="task1")
