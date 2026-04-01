from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from shutil import get_terminal_size
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'path.yaml'

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    tqdm = None


class SimpleProgressBar:
    def __init__(self, total: int, desc: str) -> None:
        self.total = max(total, 1)
        self.current = 0
        self.desc = desc

    def update(self, step: int = 1) -> None:
        self.current = min(self.total, self.current + step)
        width = min(40, max(10, get_terminal_size((80, 20)).columns - 42))
        ratio = self.current / self.total
        done = int(width * ratio)
        bar = '=' * done + '-' * (width - done)
        print(f"\r{self.desc}: [{bar}] {self.current}/{self.total} ({ratio * 100:5.1f}%)", end='', flush=True)
        if self.current >= self.total:
            print()

    def close(self) -> None:
        return


def build_progress(total: int, desc: str):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit='条')
    return SimpleProgressBar(total=total, desc=desc)


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f'路径配置文件不存在：{config_path}')

    lines = config_path.read_text(encoding='utf-8').splitlines()
    payload: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in lines:
        line_wo_comment = raw_line.split('#', 1)[0]
        if not line_wo_comment.strip():
            continue

        indent = len(line_wo_comment) - len(line_wo_comment.lstrip(' '))
        line = line_wo_comment.strip()
        if line.endswith(':'):
            current_section = line[:-1]
            payload[current_section] = {}
            continue

        key, sep, value = line.partition(':')
        if not sep:
            raise ValueError(f'无法解析配置行：{raw_line}')

        cleaned_value = value.strip().strip('"').strip("'")
        if indent == 0:
            payload[key.strip()] = cleaned_value
            current_section = None
            continue

        if current_section is None:
            raise ValueError(f'发现未归属分组的缩进行：{raw_line}')
        section_payload = payload.setdefault(current_section, {})
        if not isinstance(section_payload, dict):
            raise ValueError(f'配置分组格式错误：{current_section}')
        section_payload[key.strip()] = cleaned_value
    return payload


@dataclass
class PathConfig:
    report_csv_path: Path
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='按规则生成胃镜/肠镜任务标签与统计报告')
    parser.add_argument('--config', type=Path, default=CONFIG_PATH, help='路径配置文件，默认 configs/path.yaml')
    parser.add_argument('--report-csv', type=Path, default=None, help='可选：覆盖配置中的 valid_dicts_report_csv')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT, help='输出目录（默认项目根目录）')
    return parser.parse_args()


def build_path_config(config_path: Path, report_csv_override: Path | None, output_dir_override: Path) -> PathConfig:
    payload = load_yaml_config(config_path.expanduser())
    paths_payload = payload.get('paths')
    if not isinstance(paths_payload, dict):
        raise ValueError('path.yaml 必须包含 paths 分组')

    config_dir = config_path.expanduser().resolve().parent

    def resolve_path(raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path
        return (config_dir.parent / path).resolve()

    report_csv_config = paths_payload.get('valid_dicts_report_csv')
    report_csv_path = (
        report_csv_override.expanduser().resolve()
        if report_csv_override is not None
        else resolve_path(str(report_csv_config))
    )
    return PathConfig(report_csv_path=report_csv_path, output_dir=output_dir_override.expanduser().resolve())


REQUIRED_COLUMNS = {'reportTitle', 'watchResult'}
OPTIONAL_ALIASES: dict[str, list[str]] = {
    'exam_dir': ['exam_dir', 'examDir', 'exam_directory', 'check_dir'],
    'reportTitle': ['reportTitle', 'report_title', 'title'],
    'watchResult': ['watchResult', 'watch_result', 'diagnosis', 'result'],
    'img_num': ['img_num', 'imgNum', 'image_num', 'image_count'],
}


def detect_column(fieldnames: list[str], aliases: list[str]) -> str | None:
    for alias in aliases:
        if alias in fieldnames:
            return alias
    return None


def resolve_columns(fieldnames: list[str]) -> dict[str, str | None]:
    resolved: dict[str, str | None] = {}
    for canonical_name, aliases in OPTIONAL_ALIASES.items():
        resolved[canonical_name] = detect_column(fieldnames, aliases)
    for required in REQUIRED_COLUMNS:
        if resolved.get(required) is None:
            raise KeyError(f'输入 CSV 缺少必需字段：{required}（可接受别名：{OPTIONAL_ALIASES[required]}）')
    return resolved


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize('NFKC', text or '')
    normalized = normalized.replace('\u3000', ' ')
    normalized = re.sub(r'\s+', '', normalized)
    normalized = normalized.replace('（', '(').replace('）', ')')
    normalized = normalized.replace('【', '[').replace('】', ']')
    normalized = normalized.replace('，', ',').replace('；', ';').replace('：', ':')
    normalized = normalized.lower()
    return normalized


GASTROSCOPY_HINTS = [
    '胃镜', '食管', '胃窦', '胃体', '胃角', '贲门', '幽门', '十二指肠', '无痛胃镜', '超声胃镜',
]
COLONOSCOPY_HINTS = [
    '肠镜', '结肠', '直肠', '回盲', '乙状结肠', '升结肠', '降结肠', '横结肠', '盲肠',
]

GASTRIC_LABEL_RULES: dict[str, list[str]] = {
    'label_esophageal_smt': [
        '食管smt', '食管黏膜下隆起', '食管隆起性病变', '食管smt(来源于黏膜肌层)',
        '食管smt(来源于固有肌层)', '食管smt(来源于黏膜下层)', '食管黏膜下肿物',
    ],
    'label_esophageal_mucosal_or_tumor': [
        '食管黏膜病变', '食管肿物', '食管黏膜病变(待病理)', '食管黏膜病变(性质待定)',
        '食管肿物(待病理)', '食管占位', '食管新生物',
    ],
    'label_gastritis': [
        '慢性胃炎', '慢性非活动性胃炎', '慢性活动性胃炎', '萎缩性胃炎', '糜烂性胃炎', '浅表性胃炎',
        '胆汁反流性胃炎', '胃炎', 'c1', 'c2', 'c3', 'o1', 'o2', 'o3',
    ],
}

UNCERTAIN_TOKENS = ['待病理', '性质待定', '考虑', '可能', '？', '?', '待定']

NORMAL_PATTERNS = [
    '无异常发现', '未见异常', '未见明显异常', '检查无异常发现', '结肠镜检查未见明显异常', '结肠镜检查无异常发现',
]
POLYP_PATTERNS = ['息肉', '结肠息肉', '直肠息肉', '结直肠息肉']
MULTI_POLYP_PATTERNS = ['多发息肉', '结肠多发息肉', '结直肠多发息肉', '直肠多发息肉', '多发性结直肠息肉']
LESION_TOKENS = ['息肉', '肿物', '炎', '憩室', '溃疡', '糜烂', '癌', '出血', '狭窄', '病变']


def contains_any(text: str, patterns: list[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def match_rule_map(text: str, rule_map: dict[str, list[str]]) -> dict[str, int]:
    return {label: int(contains_any(text, keywords)) for label, keywords in rule_map.items()}


def to_int(value: str | None) -> int:
    if value is None:
        return 0
    cleaned = re.sub(r'[^0-9]', '', str(value))
    return int(cleaned) if cleaned else 0


def load_rows(report_csv_path: Path) -> tuple[list[dict[str, str]], dict[str, str | None], list[str]]:
    if not report_csv_path.is_file():
        raise FileNotFoundError(f'未找到报告总表：{report_csv_path}')

    with report_csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    resolved_columns = resolve_columns(fieldnames)
    return rows, resolved_columns, fieldnames


def get_value(row: dict[str, str], col_name: str | None) -> str:
    if col_name is None:
        return ''
    return str(row.get(col_name, '')).strip()


def is_gastroscopy_record(report_title_norm: str, watch_result_norm: str) -> bool:
    has_gastric = contains_any(report_title_norm, GASTROSCOPY_HINTS) or contains_any(watch_result_norm, GASTROSCOPY_HINTS)
    has_colon = contains_any(report_title_norm, COLONOSCOPY_HINTS) or contains_any(watch_result_norm, COLONOSCOPY_HINTS)
    return has_gastric and not has_colon


def is_colonoscopy_record(report_title_norm: str, watch_result_norm: str) -> bool:
    has_colon = contains_any(report_title_norm, COLONOSCOPY_HINTS) or contains_any(watch_result_norm, COLONOSCOPY_HINTS)
    has_gastric = contains_any(report_title_norm, GASTROSCOPY_HINTS) or contains_any(watch_result_norm, GASTROSCOPY_HINTS)
    return has_colon and not has_gastric


def build_outputs(rows: list[dict[str, str]], columns: dict[str, str | None], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    gastric_rows: list[dict[str, Any]] = []
    binary_rows: list[dict[str, Any]] = []
    triclass_rows: list[dict[str, Any]] = []

    gastric_label_counter = Counter()
    binary_counter = Counter()
    triclass_counter = Counter()
    gastric_exclude_counter = Counter()
    binary_exclude_counter = Counter()
    triclass_exclude_counter = Counter()

    gastric_total = 0
    colon_total = 0
    gastric_valid_count = 0
    binary_valid_count = 0
    triclass_valid_count = 0
    gastric_img_sum = 0
    binary_img_sum = 0
    triclass_img_sum = 0

    progress = build_progress(total=len(rows), desc='处理报告并生成任务标签')
    try:
        for row in rows:
            exam_dir = get_value(row, columns.get('exam_dir'))
            report_title = get_value(row, columns.get('reportTitle'))
            watch_result = get_value(row, columns.get('watchResult'))
            img_num = to_int(get_value(row, columns.get('img_num')))

            report_title_norm = normalize_text(report_title)
            watch_result_norm = normalize_text(watch_result)

            is_gastroscopy = int(is_gastroscopy_record(report_title_norm, watch_result_norm))
            is_colonoscopy = int(is_colonoscopy_record(report_title_norm, watch_result_norm))

            if is_gastroscopy:
                gastric_total += 1
            if is_colonoscopy:
                colon_total += 1

            label_map = match_rule_map(watch_result_norm, GASTRIC_LABEL_RULES)
            gastric_label_sum = sum(label_map.values())
            has_uncertain = contains_any(watch_result_norm, UNCERTAIN_TOKENS)

            gastric_is_valid_sample = 1
            gastric_exclude_reason = ''
            if not is_gastroscopy:
                gastric_is_valid_sample = 0
                gastric_exclude_reason = 'not_gastroscopy'
            elif not watch_result_norm:
                gastric_is_valid_sample = 0
                gastric_exclude_reason = 'empty_watchResult'
            elif gastric_label_sum == 0:
                gastric_is_valid_sample = 0
                gastric_exclude_reason = 'no_target_label'
            elif has_uncertain and gastric_label_sum == 0:
                gastric_is_valid_sample = 0
                gastric_exclude_reason = 'unclear_watchResult'

            if gastric_is_valid_sample:
                gastric_valid_count += 1
                gastric_img_sum += img_num
                for k, v in label_map.items():
                    if v == 1:
                        gastric_label_counter[k] += 1
            else:
                gastric_exclude_counter[gastric_exclude_reason] += 1

            gastric_rows.append(
                {
                    'exam_dir': exam_dir,
                    'reportTitle': report_title,
                    'watchResult': watch_result,
                    'img_num': img_num,
                    'is_gastroscopy': is_gastroscopy,
                    'label_esophageal_smt': label_map['label_esophageal_smt'],
                    'label_esophageal_mucosal_or_tumor': label_map['label_esophageal_mucosal_or_tumor'],
                    'label_gastritis': label_map['label_gastritis'],
                    'gastric_label_sum': gastric_label_sum,
                    'gastric_is_valid_sample': gastric_is_valid_sample,
                    'gastric_exclude_reason': gastric_exclude_reason,
                }
            )

            has_normal = contains_any(watch_result_norm, NORMAL_PATTERNS)
            has_polyp = contains_any(watch_result_norm, POLYP_PATTERNS)
            has_multi_polyp = contains_any(watch_result_norm, MULTI_POLYP_PATTERNS)
            has_other_lesion = contains_any(watch_result_norm, LESION_TOKENS)

            binary_is_valid_sample = 1
            binary_exclude_reason = ''
            binary_label = ''
            binary_label_name = ''

            if not is_colonoscopy:
                binary_is_valid_sample = 0
                binary_exclude_reason = 'not_colonoscopy'
            elif not watch_result_norm:
                binary_is_valid_sample = 0
                binary_exclude_reason = 'empty_watchResult'
            elif has_polyp:
                binary_label = 1
                binary_label_name = '息肉'
            elif has_normal and not has_other_lesion:
                binary_label = 0
                binary_label_name = '正常'
            elif has_normal and has_other_lesion:
                binary_is_valid_sample = 0
                binary_exclude_reason = 'conflicting_findings'
            else:
                binary_is_valid_sample = 0
                binary_exclude_reason = 'no_target_label'

            if binary_is_valid_sample:
                binary_valid_count += 1
                binary_img_sum += img_num
                binary_counter[binary_label_name] += 1
            else:
                binary_exclude_counter[binary_exclude_reason] += 1

            binary_rows.append(
                {
                    'exam_dir': exam_dir,
                    'reportTitle': report_title,
                    'watchResult': watch_result,
                    'img_num': img_num,
                    'is_colonoscopy': is_colonoscopy,
                    'binary_label': binary_label,
                    'binary_label_name': binary_label_name,
                    'binary_is_valid_sample': binary_is_valid_sample,
                    'binary_exclude_reason': binary_exclude_reason,
                }
            )

            triclass_is_valid_sample = 1
            triclass_exclude_reason = ''
            triclass_label = ''
            triclass_label_name = ''
            if not is_colonoscopy:
                triclass_is_valid_sample = 0
                triclass_exclude_reason = 'not_colonoscopy'
            elif not watch_result_norm:
                triclass_is_valid_sample = 0
                triclass_exclude_reason = 'empty_watchResult'
            elif has_multi_polyp:
                triclass_label = 2
                triclass_label_name = '多发息肉'
            elif has_polyp:
                triclass_label = 1
                triclass_label_name = '单发息肉'
            elif has_normal and not has_other_lesion:
                triclass_label = 0
                triclass_label_name = '正常'
            elif has_normal and has_other_lesion:
                triclass_is_valid_sample = 0
                triclass_exclude_reason = 'conflicting_findings'
            else:
                triclass_is_valid_sample = 0
                triclass_exclude_reason = 'no_target_label'

            if triclass_is_valid_sample:
                triclass_valid_count += 1
                triclass_img_sum += img_num
                triclass_counter[triclass_label_name] += 1
            else:
                triclass_exclude_counter[triclass_exclude_reason] += 1

            triclass_rows.append(
                {
                    'exam_dir': exam_dir,
                    'reportTitle': report_title,
                    'watchResult': watch_result,
                    'img_num': img_num,
                    'triclass_label': triclass_label,
                    'triclass_label_name': triclass_label_name,
                    'triclass_is_valid_sample': triclass_is_valid_sample,
                    'triclass_exclude_reason': triclass_exclude_reason,
                }
            )
            progress.update(1)
    finally:
        progress.close()

    gastric_csv = output_dir / 'gastric_multilabel_task.csv'
    binary_csv = output_dir / 'colonoscopy_binary_task.csv'
    triclass_csv = output_dir / 'colonoscopy_triclass_task.csv'
    summary_md = output_dir / 'task_definition_summary.md'

    write_csv(gastric_csv, gastric_rows)
    write_csv(binary_csv, binary_rows)
    write_csv(triclass_csv, triclass_rows)

    summary_text = render_summary(
        gastric_total=gastric_total,
        gastric_valid_count=gastric_valid_count,
        gastric_img_sum=gastric_img_sum,
        gastric_label_counter=gastric_label_counter,
        gastric_exclude_counter=gastric_exclude_counter,
        colon_total=colon_total,
        binary_valid_count=binary_valid_count,
        binary_img_sum=binary_img_sum,
        binary_counter=binary_counter,
        binary_exclude_counter=binary_exclude_counter,
        triclass_valid_count=triclass_valid_count,
        triclass_img_sum=triclass_img_sum,
        triclass_counter=triclass_counter,
        triclass_exclude_counter=triclass_exclude_counter,
    )
    summary_md.write_text(summary_text, encoding='utf-8')

    print('\n=== 任务划分统计 ===')
    print(f'胃镜总记录数：{gastric_total}')
    print(f'胃镜纳入多标签任务的记录数：{gastric_valid_count}')
    print(f'胃镜标签-食管SMT样本数：{gastric_label_counter.get("label_esophageal_smt", 0)}')
    print(f'胃镜标签-食管黏膜病变/食管肿物样本数：{gastric_label_counter.get("label_esophageal_mucosal_or_tumor", 0)}')
    print(f'胃镜标签-胃炎类样本数：{gastric_label_counter.get("label_gastritis", 0)}')
    print(f'肠镜总记录数：{colon_total}')
    print(f'肠镜二分类纳入记录数：{binary_valid_count}')
    print(f'肠镜二分类正常样本数：{binary_counter.get("正常", 0)}')
    print(f'肠镜二分类息肉样本数：{binary_counter.get("息肉", 0)}')
    print(f'肠镜三分类纳入记录数：{triclass_valid_count}')
    print(f'肠镜三分类正常样本数：{triclass_counter.get("正常", 0)}')
    print(f'肠镜三分类单发息肉样本数：{triclass_counter.get("单发息肉", 0)}')
    print(f'肠镜三分类多发息肉样本数：{triclass_counter.get("多发息肉", 0)}')
    print('输出文件路径：')
    for path in [gastric_csv, binary_csv, triclass_csv, summary_md]:
        print(f'- {path}')

    return {
        'gastric_csv': gastric_csv,
        'binary_csv': binary_csv,
        'triclass_csv': triclass_csv,
        'summary_md': summary_md,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_counter(counter: Counter) -> str:
    if not counter:
        return '- 无\n'
    return ''.join([f'- {k}: {v}\n' for k, v in counter.most_common()])


def render_summary(
    *,
    gastric_total: int,
    gastric_valid_count: int,
    gastric_img_sum: int,
    gastric_label_counter: Counter,
    gastric_exclude_counter: Counter,
    colon_total: int,
    binary_valid_count: int,
    binary_img_sum: int,
    binary_counter: Counter,
    binary_exclude_counter: Counter,
    triclass_valid_count: int,
    triclass_img_sum: int,
    triclass_counter: Counter,
    triclass_exclude_counter: Counter,
) -> str:
    return f"""# 任务划分与标签生成统计报告

## 胃镜任务定义说明
- 任务类型：3 标签多标签分类。
- 标签包括：食管 SMT、食管黏膜病变/食管肿物、胃炎类。
- 可多标签共存，不因多标签直接剔除。

## 胃镜 3 标签规则说明
- `label_esophageal_smt`：基于“食管SMT/食管黏膜下隆起/来源层描述”等关键词匹配。
- `label_esophageal_mucosal_or_tumor`：基于“食管黏膜病变/食管肿物/待病理/性质待定”等关键词匹配。
- `label_gastritis`：基于“慢性胃炎/萎缩性胃炎/糜烂性胃炎/C1~O3”等关键词匹配。

## 肠镜二分类任务定义说明
- 类别 0：正常（明确无异常，且不伴随病变词）。
- 类别 1：息肉（出现息肉相关关键词）。

## 肠镜三分类候选方案说明
- 类别 0：正常。
- 类别 1：单发息肉（有息肉但不含“多发”语义）。
- 类别 2：多发息肉（命中多发息肉关键词）。

## 最终纳入样本数（检查目录数）
- 胃镜多标签纳入：{gastric_valid_count} / 总胃镜记录 {gastric_total}
- 肠镜二分类纳入：{binary_valid_count} / 总肠镜记录 {colon_total}
- 肠镜三分类纳入：{triclass_valid_count} / 总肠镜记录 {colon_total}

## 最终图像总数（img_num）
- 胃镜多标签：{gastric_img_sum}
- 肠镜二分类：{binary_img_sum}
- 肠镜三分类：{triclass_img_sum}

## 标签/类别分布统计
### 胃镜 3 标签
{render_counter(gastric_label_counter)}
### 肠镜二分类
{render_counter(binary_counter)}
### 肠镜三分类候选
{render_counter(triclass_counter)}

## 剔除样本统计与主要原因
### 胃镜任务剔除原因
{render_counter(gastric_exclude_counter)}
### 肠镜二分类剔除原因
{render_counter(binary_exclude_counter)}
### 肠镜三分类剔除原因
{render_counter(triclass_exclude_counter)}

## 混杂表达说明
- 已对中文文本做标准化处理（NFKC、空白合并、全半角与括号统一）。
- 对“正常”类启用病变词冲突检查，若同条记录同时出现病变语义则剔除。
- 对“待病理/性质待定/考虑/可能”等不确定表达做兼容；若无目标标签则剔除。
"""


def main() -> None:
    args = parse_args()
    config = build_path_config(args.config, args.report_csv, args.output_dir)
    rows, columns, fieldnames = load_rows(config.report_csv_path)

    print('输入 CSV 字段映射：')
    for key, value in columns.items():
        print(f'- {key}: {value}')
    print(f'输入字段总数：{len(fieldnames)}')
    print(f'输入记录总数：{len(rows)}')

    build_outputs(rows=rows, columns=columns, output_dir=config.output_dir)


if __name__ == '__main__':
    main()
