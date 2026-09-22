#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


# ============================================================
# 路径
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "datasets/mr_rate_1k"
OUT = ROOT / "outputs/mr_rate_1k"
SOURCE = DATA / "source"

MANIFEST = DATA / "samples_1000.csv"
FOLDS_JSON = DATA / "folds_5fold.json"
SUMMARY_JSON = DATA / "cohort_summary.json"
PLAN_JSON = DATA / "download_plan.json"

REPO_ID = "Forithmus/MR-RATE"


# ============================================================
# 固定标签
#
# bit 顺序：
# 0 Unspecific
# 1 Neurodegenerative
# 2 Cerebrovascular
# 3 Neoplastic
# ============================================================

LABELS = [
    "PP_Unspecific_bucket",
    "PP_Neurodegenerative",
    "PP_Cerebrovascular",
    "PP_Neoplastic",
]

SHORT_LABELS = [
    "Unspecific",
    "Neurodegenerative",
    "Cerebrovascular",
    "Neoplastic",
]


# ============================================================
# 固定 MR-RATE-1K 标签组合
#
# 最终：
# 0 positive = 350
# 1 positive = 292
# 2 positive = 191
# 3 positive = 142
# 4 positive = 25
#
# 标签总阳性：
# 379 / 320 / 270 / 231
# ============================================================

COMBOS = [
    "0000",
    "0001",
    "0010",
    "0011",
    "0100",
    "0101",
    "0110",
    "0111",
    "1000",
    "1001",
    "1010",
    "1011",
    "1100",
    "1101",
    "1110",
    "1111",
]

QUOTAS = {
    "0000": 350,
    "0001": 69,
    "0010": 51,
    "0011": 21,
    "0100": 83,
    "0101": 14,
    "0110": 22,
    "0111": 11,
    "1000": 89,
    "1001": 36,
    "1010": 31,
    "1011": 33,
    "1100": 67,
    "1101": 22,
    "1110": 76,
    "1111": 25,
}


# ============================================================
# 五折组合配额
#
# 每个 fold = 200 patients / 200 studies
#
# Fold标签阳性：
# F1 76 / 64 / 54 / 46
# F2 75 / 64 / 54 / 46
# F3 76 / 64 / 54 / 47
# F4 76 / 64 / 54 / 46
# F5 76 / 64 / 54 / 46
# ============================================================

FOLD_MATRIX = {
    "0000": [70, 70, 70, 70, 70],
    "0001": [14, 13, 14, 14, 14],
    "0010": [10, 11, 10, 10, 10],
    "0011": [4, 5, 4, 4, 4],
    "0100": [16, 17, 16, 17, 17],
    "0101": [3, 3, 3, 3, 2],
    "0110": [5, 4, 5, 4, 4],
    "0111": [2, 2, 2, 2, 3],
    "1000": [18, 18, 18, 17, 18],
    "1001": [7, 7, 7, 8, 7],
    "1010": [6, 6, 6, 7, 6],
    "1011": [7, 6, 7, 6, 7],
    "1100": [14, 13, 13, 13, 14],
    "1101": [4, 5, 5, 4, 4],
    "1110": [15, 15, 15, 16, 15],
    "1111": [5, 5, 5, 5, 5],
}


# ============================================================
# 固定官方标签版本
#
# 使用我们前面统计过的官方 merged-majority benchmark。
# 固定 Git commit，防止以后官方仓库更新后队列变化。
# ============================================================

GITHUB_COMMIT = "43c05c61bf98d00c4e825a27a0fca3c742754d78"

BASE = (
    "https://raw.githubusercontent.com/forithmus/MR-RATE/"
    + GITHUB_COMMIT
    + "/contrastive-pretraining/scripts/eval_labels/"
    + "splits_merged_majority/"
)

LABEL_URL = BASE + "mrrate_merged_labels.csv"
SPLIT_URL = BASE + "splits.csv"


def fnv1a(text: str) -> int:
    """稳定的32-bit FNV-1a，保证每次重新生成同一批病例。"""
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def download_small_source(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"[source] 下载 {path.name}", flush=True)
        urllib.request.urlretrieve(url, path)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def prepare_cohort():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)

    labels_path = SOURCE / "mrrate_merged_labels.csv"
    splits_path = SOURCE / "splits_merged_majority.csv"

    download_small_source(LABEL_URL, labels_path)
    download_small_source(SPLIT_URL, splits_path)

    label_rows = read_csv(labels_path)
    split_rows = read_csv(splits_path)

    split_map = {
        row["study_uid"]: row
        for row in split_rows
    }

    grouped = defaultdict(list)

    for row in label_rows:
        study_uid = row["study_uid"]

        if study_uid not in split_map:
            continue

        s = split_map[study_uid]

        bits = "".join(
            "1" if row[label] == "1" else "0"
            for label in LABELS
        )

        item = {
            "study_uid": study_uid,
            "patient_uid": s["patient_uid"],
            "benchmark_batch_id": s["batch_id"],
            "official_split": s["split"],
            "combo": bits,
        }

        item["stable_hash"] = fnv1a(
            "MRRATE-AMEF-1K-balanced-v1|"
            + item["patient_uid"]
            + "|"
            + item["study_uid"]
        )

        grouped[bits].append(item)

    for combo in COMBOS:
        grouped[combo].sort(
            key=lambda x: (x["stable_hash"], x["study_uid"])
        )

    # 优先处理相对稀有的组合，避免患者去重导致稀有组合不够
    combo_order = sorted(
        COMBOS,
        key=lambda c: (
            len(grouped[c]) / QUOTAS[c],
            len(grouped[c]),
            c,
        ),
    )

    used_patients = set()
    selected = {}

    for combo in combo_order:
        selected[combo] = []

        for item in grouped[combo]:

            if item["patient_uid"] in used_patients:
                continue

            selected[combo].append(item)
            used_patients.add(item["patient_uid"])

            if len(selected[combo]) >= QUOTAS[combo]:
                break

        if len(selected[combo]) != QUOTAS[combo]:
            raise RuntimeError(
                f"{combo} 数量不足："
                f"{len(selected[combo])}/{QUOTAS[combo]}"
            )

    manifest = []

    # 按固定矩阵分到5个fold
    for combo in COMBOS:
        rows = selected[combo]

        cursor = 0

        for fold_idx in range(5):
            n = FOLD_MATRIX[combo][fold_idx]

            for _ in range(n):
                item = dict(rows[cursor])
                cursor += 1

                item["fold"] = fold_idx + 1
                manifest.append(item)

        assert cursor == QUOTAS[combo]

    assert len(manifest) == 1000
    assert len({x["patient_uid"] for x in manifest}) == 1000
    assert len({x["study_uid"] for x in manifest}) == 1000

    # 汇总
    positive_distribution = [0, 0, 0, 0, 0]
    label_positive_counts = [0, 0, 0, 0]

    fold_summary = {}

    for fold in range(1, 6):
        fold_rows = [x for x in manifest if x["fold"] == fold]

        fold_k = [0, 0, 0, 0, 0]
        fold_labels = [0, 0, 0, 0]

        for item in fold_rows:
            k = item["combo"].count("1")
            fold_k[k] += 1

            for i, bit in enumerate(item["combo"]):
                if bit == "1":
                    fold_labels[i] += 1

        fold_summary[str(fold)] = {
            "n": len(fold_rows),
            "positive_count_distribution": fold_k,
            "label_positive_counts": fold_labels,
        }

    for item in manifest:
        k = item["combo"].count("1")
        positive_distribution[k] += 1

        for i, bit in enumerate(item["combo"]):
            if bit == "1":
                label_positive_counts[i] += 1

    # 强制验证我们商定的数据结构
    assert positive_distribution == [350, 292, 191, 142, 25]
    assert label_positive_counts == [379, 320, 270, 231]

    expected_fold_labels = [
        [76, 64, 54, 46],
        [75, 64, 54, 46],
        [76, 64, 54, 47],
        [76, 64, 54, 46],
        [76, 64, 54, 46],
    ]

    for i in range(5):
        assert fold_summary[str(i + 1)]["n"] == 200
        assert (
            fold_summary[str(i + 1)]["label_positive_counts"]
            == expected_fold_labels[i]
        )

    # 保存manifest
    manifest.sort(
        key=lambda x: (
            x["fold"],
            x["combo"],
            x["stable_hash"],
            x["study_uid"],
        )
    )

    fields = [
        "case_index",
        "patient_uid",
        "study_uid",
        "fold",
        "combo",
        "positive_count",
        *LABELS,
        "official_split",
        "benchmark_batch_id",
    ]

    with MANIFEST.open(
        "w", encoding="utf-8", newline=""
    ) as f:

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for idx, item in enumerate(manifest):
            out = {
                "case_index": idx,
                "patient_uid": item["patient_uid"],
                "study_uid": item["study_uid"],
                "fold": item["fold"],
                "combo": item["combo"],
                "positive_count": item["combo"].count("1"),
                "official_split": item["official_split"],
                "benchmark_batch_id": item["benchmark_batch_id"],
            }

            for i, label in enumerate(LABELS):
                out[label] = int(item["combo"][i])

            writer.writerow(out)

    folds_payload = {
        str(fold): [
            x["study_uid"]
            for x in manifest
            if x["fold"] == fold
        ]
        for fold in range(1, 6)
    }

    FOLDS_JSON.write_text(
        json.dumps(
            folds_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "dataset": "MR-RATE-1K",
        "patients": 1000,
        "studies": 1000,
        "labels": LABELS,
        "positive_count_distribution": {
            "0": 350,
            "1": 292,
            "2": 191,
            "3": 142,
            "4": 25,
        },
        "label_positive_counts": dict(
            zip(LABELS, label_positive_counts)
        ),
        "folds": fold_summary,
        "source_commit": GITHUB_COMMIT,
        "selection": (
            "balanced enriched 4-label cohort; "
            "one study per patient"
        ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n[cohort] MR-RATE-1K 已固定")
    print("[cohort] 1000 patients / 1000 studies")
    print(
        "[cohort] 0/1/2/3/4 positive = "
        "350 / 292 / 191 / 142 / 25"
    )
    print(
        "[cohort] label positives = "
        "379 / 320 / 270 / 231"
    )

    return manifest


def build_download_plan(manifest):
    print("\n[HF] 读取MR-RATE文件列表……", flush=True)

    api = HfApi()

    # token=True：使用 hf auth login 保存的token
    try:
        info = api.dataset_info(
            REPO_ID,
            token=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "\n无法访问 Forithmus/MR-RATE。\n"
            "请确认：\n"
            "1. 已在网页同意MR-RATE使用条款；\n"
            "2. 已运行 hf auth login；\n"
            "3. 当前token具有read权限。\n"
        ) from exc

    revision = info.sha

    repo_files = api.list_repo_files(
        REPO_ID,
        repo_type="dataset",
        revision=revision,
        token=True,
    )

    target_uids = {
        x["study_uid"]
        for x in manifest
    }

    path_by_uid = {}

    for path in repo_files:

        if not (
            path.startswith("mri/")
            and path.endswith(".zip")
        ):
            continue

        uid = Path(path).stem

        if uid in target_uids:
            path_by_uid[uid] = path

    missing = sorted(target_uids - set(path_by_uid))

    if missing:
        raise RuntimeError(
            f"有 {len(missing)} 个study没有找到MRI ZIP："
            + ", ".join(missing[:10])
        )

    mri_paths = sorted(path_by_uid.values())

    # 精确查询1000个ZIP的远程文件大小
    remote_size = {}

    for start in tqdm(
        range(0, len(mri_paths), 200),
        desc="查询ZIP精确大小",
    ):
        chunk = mri_paths[start:start + 200]

        infos = api.get_paths_info(
            REPO_ID,
            paths=chunk,
            repo_type="dataset",
            revision=revision,
            token=True,
        )

        for item in infos:
            remote_size[item.path] = int(item.size)

    if len(remote_size) != 1000:
        raise RuntimeError(
            f"只取得 {len(remote_size)}/1000 个ZIP大小"
        )

    total_bytes = sum(remote_size.values())

    batches = sorted({
        Path(path).parent.name
        for path in mri_paths
    })

    plan = {
        "repo_id": REPO_ID,
        "revision": revision,
        "mri_zip_count": len(mri_paths),
        "selected_batches": batches,
        "mri_zip_bytes": total_bytes,
        "mri_zip_GB_decimal": total_bytes / 1e9,
        "mri_zip_GiB": total_bytes / (1024 ** 3),
        "files": [
            {
                "path": path,
                "bytes": remote_size[path],
            }
            for path in mri_paths
        ],
    }

    PLAN_JSON.write_text(
        json.dumps(
            plan,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n============================================")
    print("MR-RATE-1K 下载计划")
    print("============================================")
    print(f"Study ZIP数 : {len(mri_paths)}")
    print(f"患者数      : 1000")
    print(f"HF revision : {revision}")
    print(
        f"MRI精确大小 : "
        f"{total_bytes / 1e9:.2f} GB "
        f"({total_bytes / (1024**3):.2f} GiB)"
    )
    print(
        "注意：以上只计算1000个MRI ZIP；"
        "metadata/report CSV额外占用很小。"
    )
    print("============================================")

    return plan


def download_dataset(plan, workers: int):
    revision = plan["revision"]

    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    paths = [
        item["path"]
        for item in plan["files"]
    ]

    batches = plan["selected_batches"]

    # --------------------------------------------------------
    # 下载小型表格
    # --------------------------------------------------------

    small_files = [
        "pathology_labels/mrrate_labels.csv",
        "splits.csv",
    ]

    for batch in batches:
        small_files.extend([
            f"metadata/{batch}_metadata.csv",
            f"reports/{batch}_reports.csv",
        ])

    print(
        f"\n[download] 下载 "
        f"{len(small_files)} 个metadata/report文件",
        flush=True,
    )

    for path in tqdm(
        small_files,
        desc="metadata/reports",
    ):
        hf_hub_download(
            REPO_ID,
            filename=path,
            repo_type="dataset",
            revision=revision,
            local_dir=DATA,
            token=True,
        )

    # --------------------------------------------------------
    # 下载1000个MRI study ZIP
    # --------------------------------------------------------

    print(
        f"\n[download] 开始下载1000个MRI ZIP，"
        f"workers={workers}",
        flush=True,
    )

    def worker(path):
        result = hf_hub_download(
            REPO_ID,
            filename=path,
            repo_type="dataset",
            revision=revision,
            local_dir=DATA,
            token=True,
        )
        return path, result

    failed = []

    with ThreadPoolExecutor(
        max_workers=workers
    ) as pool:

        futures = {
            pool.submit(worker, path): path
            for path in paths
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="MRI studies",
        ):
            path = futures[future]

            try:
                future.result()
            except Exception as exc:
                failed.append({
                    "path": path,
                    "error": repr(exc),
                })

    if failed:
        failed_path = OUT / "download_failed.json"

        failed_path.write_text(
            json.dumps(
                failed,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        raise RuntimeError(
            f"{len(failed)} 个文件下载失败。"
            f"重新运行相同命令即可续传。"
        )

    # --------------------------------------------------------
    # 最终检查
    # --------------------------------------------------------

    missing = []
    local_bytes = 0

    for path in paths:
        local = DATA / path

        if not local.exists():
            missing.append(path)
            continue

        local_bytes += local.stat().st_size

    if missing:
        raise RuntimeError(
            f"下载后仍缺少 {len(missing)} 个ZIP"
        )

    report = {
        "status": "complete",
        "studies": 1000,
        "patients": 1000,
        "mri_zip_count": 1000,
        "mri_zip_bytes": local_bytes,
        "mri_zip_GB_decimal": local_bytes / 1e9,
        "mri_zip_GiB": local_bytes / (1024 ** 3),
        "data_root": str(DATA),
        "manifest": str(MANIFEST),
        "folds": str(FOLDS_JSON),
    }

    (OUT / "download_complete.json").write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n============================================")
    print("MR-RATE-1K 下载完成")
    print("============================================")
    print(f"Patients : 1000")
    print(f"Studies  : 1000")
    print(
        "0/1/2/3/4 positive : "
        "350 / 292 / 191 / 142 / 25"
    )
    print(
        f"MRI ZIP实际大小 : "
        f"{local_bytes / 1e9:.2f} GB "
        f"({local_bytes / (1024**3):.2f} GiB)"
    )
    print(f"DATA : {DATA}")
    print(f"OUT  : {OUT}")
    print("============================================")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prepare-only",
        action="store_true",
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
    )

    parser.add_argument(
        "--download",
        action="store_true",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    manifest = prepare_cohort()

    if args.prepare_only:
        return

    plan = build_download_plan(manifest)

    if args.plan_only:
        return

    if args.download:
        download_dataset(
            plan,
            max(1, args.workers),
        )
        return

    print(
        "\n未指定下载。"
        "\n先运行 --plan-only 查看精确大小，"
        "\n然后运行 --download。"
    )


if __name__ == "__main__":
    main()
