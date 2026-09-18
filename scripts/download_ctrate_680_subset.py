#!/usr/bin/env python3
"""下载固定680例CT-RATE；保留原始报告，逐例校验大小、SHA256及NIfTI头。"""

import argparse
import csv
import fcntl
import hashlib
import json
import logging
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

# 使用可续传HTTP下载，避免额外的Xet分块缓存；令牌由HF凭据存储读取。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

import nibabel as nib
import requests
from huggingface_hub import HfApi, configure_http_backend, hf_hub_download


V2RAYA_PROXY = "http://127.0.0.1:21171"


def configure_network(mode):
    """仅设置下载进程的网络，显式选择直连或本机V2rayA，不回退到其他代理。"""
    for key in list(os.environ):
        if key.lower().endswith("_proxy"):
            os.environ.pop(key, None)
    proxy = V2RAYA_PROXY if mode == "v2raya" else None
    bypass = "localhost,127.0.0.1,::1" if proxy else "*"
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = bypass
    if proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ[key] = proxy

    def session_factory():
        session = requests.Session()
        session.trust_env = False
        if proxy:
            session.proxies = {"http": proxy, "https": proxy}
        return session

    configure_http_backend(backend_factory=session_factory)
    return proxy

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "outputs/dataset_candidates/ct_rate_680_file_sizes.json"
DATA = ROOT / "datasets/ct_rate_680"
OUT = ROOT / "outputs/ct_rate_680/download"
LABEL_FIELDS = ["Emphysema", "Atelectasis", "Pulmonary fibrotic sequela"]
REPORT_FIELDS = ["ClinicalInformation_EN", "Technique_EN", "Findings_EN", "Impressions_EN"]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_csv(path, rows, fields):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def fetch(manifest, name):
    return Path(hf_hub_download(
        manifest["repository"], name, repo_type="dataset", revision=manifest["revision"],
        local_dir=DATA, token=True, etag_timeout=30,
    ))


def prepare_tables(manifest):
    """用官方VolumeName连接报告和元数据，标签保持已选人工标注名单不变。"""
    report_map, metadata_map, non_chest = {}, {}, set()
    for split, flag_split in (("train", "train"), ("validation", "valid")):
        for folder, suffix, mapping in (
            ("radiology_text_reports", "reports", report_map),
            ("metadata", "metadata", metadata_map),
        ):
            path = fetch(manifest, f"dataset/{folder}/{split}_{suffix}.csv")
            with path.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    key = row["VolumeName"]
                    if key in mapping and mapping[key] != row:
                        raise ValueError("官方表出现同影像编号但内容冲突")
                    mapping[key] = row
        flags = fetch(manifest, f"dataset/metadata/no_chest_{flag_split}.txt")
        non_chest.update(Path(line.strip()).name for line in flags.read_text().splitlines() if line.strip())
    fetch(manifest, "dataset/data_correction_note.md")
    samples, metadata, flagged = [], [], []
    for item in manifest["selected_files"]:
        name = Path(item["file_path"]).name
        report, meta = report_map[name], metadata_map[name]
        if not report["Findings_EN"].strip():
            raise ValueError(f"缺少影像所见文本：{name}")
        if name in non_chest:
            flagged.append(name)
        row = {key: item[key] for key in ("patient_id", "exam_id", "file_path")}
        row["VolumeName"] = name
        row.update(dict(zip(LABEL_FIELDS, item["labels"])))
        row.update({key: report[key] for key in REPORT_FIELDS})
        samples.append(row)
        metadata.append(meta)
    if flagged:
        write_json(OUT / "non_chest_conflicts.json", flagged)
        raise ValueError(f"已选名单中有{len(flagged)}例进入官方非胸部清单，停止并等待核查")
    fields = ["patient_id", "exam_id", "file_path", "VolumeName"] + LABEL_FIELDS + REPORT_FIELDS
    write_csv(DATA / "samples_680.csv", samples, fields)
    write_csv(DATA / "metadata_680.csv", metadata, list(metadata[0]))
    write_json(DATA / "selection_manifest.json", manifest)
    summary = {
        "patients": len(samples), "reports": len(samples), "metadata_rows": len(metadata),
        "positive_counts": [sum(row[label] for row in samples) for label in LABEL_FIELDS],
        "non_chest_conflicts": 0, "report_text_modified": False,
        "label_source": "作者公开人工标注子集；不是全量自动提取标签",
        "report_source": "官方英文临床报告，保留各段原文；尚未进行实验用掩码或输入筛选",
    }
    write_json(OUT / "table_verification.json", summary)
    print("已配对680例报告、元数据与三标签；官方非胸部清单交集为0。", flush=True)


def load_hashes(manifest):
    path = OUT / "expected_sha256.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached["revision"] == manifest["revision"] and len(cached["files"]) == 680:
            return cached["files"]
    api, files = HfApi(), {}
    selected = manifest["selected_files"]
    for start in range(0, len(selected), 100):
        rows = selected[start:start + 100]
        info = api.get_paths_info(
            manifest["repository"], [r["file_path"] for r in rows],
            repo_type="dataset", revision=manifest["revision"], token=True,
        )
        entries = {entry.path: entry for entry in info}
        for row in rows:
            entry = entries[row["file_path"]]
            if entry.size != row["bytes"] or entry.lfs is None:
                raise ValueError("官方文件大小或校验元数据与固定名单不一致")
            files[row["file_path"]] = entry.lfs.sha256
        print(f"已读取官方校验值：{len(files)}/680", flush=True)
    write_json(path, {"revision": manifest["revision"], "files": files})
    return files


def download_one(manifest, row, expected_hash):
    for attempt in range(1, 5):
        try:
            path = fetch(manifest, row["file_path"])
            if path.stat().st_size != row["bytes"]:
                raise ValueError("文件字节数不匹配")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected_hash:
                raise ValueError("SHA256不匹配；保留文件以便核查")
            volume = nib.load(str(path))
            if len(volume.shape) != 3 or min(volume.shape) <= 0:
                raise ValueError("NIfTI维度异常")
            return {
                "patient_id": row["patient_id"], "file_path": row["file_path"],
                "bytes": row["bytes"], "sha256": digest.hexdigest(),
                "shape": list(volume.shape),
                "spacing": [float(x) for x in volume.header.get_zooms()],
                "axis_codes": list(nib.aff2axcodes(volume.affine)),
                "verified_at": timestamp(),
            }
        except Exception as exc:
            # 不输出异常URL，避免签名下载链接或认证信息出现在日志中。
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if attempt == 4 or isinstance(exc, ValueError) or code in (401, 403):
                return {"patient_id": row["patient_id"], "file_path": row["file_path"],
                        "error_type": type(exc).__name__, "http_status": code}
            print(f"{row['patient_id']}重试{attempt}/3，原因：{type(exc).__name__}", flush=True)
            time.sleep(attempt * 3)


def run(args):
    manifest = json.loads(INPUT.read_text())
    rows = manifest["selected_files"]
    assert len(rows) == len({row["patient_id"] for row in rows}) == 680
    assert manifest["label_names"] == LABEL_FIELDS
    assert [sum(row["labels"][j] for row in rows) for j in range(3)] == [148, 169, 188]
    assert [sum(sum(row["labels"]) == count for row in rows) for count in range(4)] == [305, 259, 102, 14]
    existing = sum(min((DATA / r["file_path"]).stat().st_size, r["bytes"])
                   for r in rows if (DATA / r["file_path"]).exists())
    required = manifest["bytes"] - existing + 20 * 2**30
    if shutil.disk_usage(DATA).free < required:
        raise ValueError("磁盘空间不足：需要剩余影像容量加20GiB工作余量")
    state = {"status": "preparing", "started_at": timestamp(), "pid": os.getpid(),
             "total": 680, "total_bytes": manifest["bytes"], "verified": 0, "failed": 0,
             "verified_bytes": 0, "data_root": str(DATA), "workers": args.workers,
             "network_mode": args.network_mode, "proxy_enabled": args.network_mode == "v2raya",
             "proxy_url": V2RAYA_PROXY if args.network_mode == "v2raya" else None}
    write_json(OUT / "status.json", state)
    prepare_tables(manifest)
    hashes = load_hashes(manifest)
    if args.prepare_only:
        state.update(status="prepared", updated_at=timestamp())
        write_json(OUT / "status.json", state)
        return
    completed, failures = [], []
    started = time.monotonic()
    state["status"] = "downloading"
    with ThreadPoolExecutor(args.workers) as pool:
        pending = {pool.submit(download_one, manifest, row, hashes[row["file_path"]]) for row in rows}
        while pending:
            done, pending = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                (failures if "error_type" in result else completed).append(result)
            verified_bytes = sum(row["bytes"] for row in completed)
            state.update(verified=len(completed), failed=len(failures), verified_bytes=verified_bytes,
                         elapsed_seconds=round(time.monotonic() - started, 1), updated_at=timestamp())
            write_json(OUT / "status.json", state)
            write_json(OUT / "verified_files.json", sorted(completed, key=lambda row: row["patient_id"]))
            write_json(OUT / "failed_files.json", failures)
            print(f"已校验 {len(completed)}/680（{verified_bytes / 1e9:.2f}/{manifest['bytes'] / 1e9:.2f} GB），失败 {len(failures)}", flush=True)
    state.update(status="complete" if len(completed) == 680 and not failures else "incomplete",
                 finished_at=timestamp())
    write_json(OUT / "status.json", state)
    print("680例下载与校验全部完成。" if state["status"] == "complete" else "下载结束，有未完成文件，请查看失败清单。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--network-mode", choices=("direct", "v2raya"), default="direct",
                        help="直连，或通过本机V2rayA的HTTP端口21171下载；不自动回退")
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 8:
        parser.error("并发数需在1到8之间")
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
    selected_proxy = configure_network(arguments.network_mode)
    print(f"网络模式：V2rayA，代理地址{selected_proxy}；不自动切换其他代理。" if selected_proxy
          else "网络模式：仅直连，代理环境变量与HTTP会话代理均已禁用。", flush=True)
    with (OUT / "download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("已有同一下载任务运行，退出以避免重复。")
        try:
            run(arguments)
        except Exception as exc:
            write_json(OUT / "fatal_error.json", {"at": timestamp(), "type": type(exc).__name__})
            print(f"下载任务停止，错误类型：{type(exc).__name__}；未记录认证信息。", flush=True)
            raise SystemExit(1)
