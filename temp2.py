from __future__ import annotations

import argparse
import json
from pathlib import Path

from statiscs import build_path_config, collect_pdf_stats
from check_pdf import CONFIG_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="定位统计过程中解析失败的 PDF")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="路径配置文件，默认使用 configs/path.yaml",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="可选：覆盖 path.yaml 中的 dataset_root",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="可选：仅处理前 N 个患者，便于快速抽查",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path_config = build_path_config(args.config, args.dataset_root)

    if not path_config.dataset_root.exists():
        print(f"数据集根目录不存在：{path_config.dataset_root}")
        return
    if not path_config.dataset_root.is_dir():
        print(f"数据集根路径不是目录：{path_config.dataset_root}")
        return

    _, _, _, errors = collect_pdf_stats(
        path_config.dataset_root,
        max_patients=args.max_patients,
    )

    print(f"解析失败 PDF 数：{len(errors)}")
    if not errors:
        print("没有发现解析失败的 PDF。")
        return

    print(json.dumps(errors, ensure_ascii=False, indent=2))

    print("\n失败 PDF 路径列表：")
    for index, item in enumerate(errors, start=1):
        print(f"{index}. {item['pdf_path']}")
        print(f"   患者：{item['patient_id']}")
        print(f"   检查：{item['exam_id']}")
        print(f"   原因：{item['reason']}")


if __name__ == "__main__":
    main()
