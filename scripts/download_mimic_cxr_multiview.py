#!/usr/bin/env python3
"""
Download the MIMIC-CXR multi-image study subset for Project4/AMEF-MIL.

Default inclusion criteria:
- one study has >=2 JPG images;
- one matched free-text radiology report;
- one study-level CheXpert label row;
- preserve raw CheXpert states: 1 positive, 0 negative, -1 uncertain, blank not stated.

Project paths:
  data:   /xmlg/Lim/Project4/datasets/mimic_cxr
  status: /xmlg/Lim/Project4/outputs/mimic_cxr/download

Only MIMIC-CXR-JPG images are downloaded. Original DICOM files are not downloaded.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import getpass
import gzip
import hashlib
import json
import os
import shutil
import threading
import time
import zipfile
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/mimic_cxr"
OUT = ROOT / "outputs/mimic_cxr/download"

JPG_BASE = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"
CXR_BASE = "https://physionet.org/files/mimic-cxr/2.1.0/"
V2RAYA_PROXY = "http://127.0.0.1:21171"

TABLE_FILES = {
    "split": "mimic-cxr-2.0.0-split.csv.gz",
    "metadata": "mimic-cxr-2.0.0-metadata.csv.gz",
    "chexpert": "mimic-cxr-2.0.0-chexpert.csv.gz",
    "negbio": "mimic-cxr-2.0.0-negbio.csv.gz",
}
REPORT_ARCHIVE = "mimic-cxr-reports.zip"

_thread_local = threading.local()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def credentials(args):
    username = args.username or os.environ.get("PHYSIONET_USERNAME")
    if not username:
        raise SystemExit("请用 --username 提供 PhysioNet 用户名。")
    password = os.environ.get("PHYSIONET_PASSWORD")
    if password is None:
        password = getpass.getpass("PhysioNet password（不会写入日志）: ")
    return username, password


def session_for(username, password, network_mode):
    key = (username, network_mode)
    cached = getattr(_thread_local, "cached", None)
    if cached and cached[0] == key:
        return cached[1]
    s = requests.Session()
    s.auth = (username, password)
    s.trust_env = False
    s.headers.update({"User-Agent": "Project4-MIMIC-CXR/1.0"})
    if network_mode == "v2raya":
        s.proxies = {"http": V2RAYA_PROXY, "https": V2RAYA_PROXY}
    _thread_local.cached = (key, s)
    return s


def stream_download(session, url, target: Path, retries=4):
    """Resumable download via .part. Existing final files are never deleted."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    part = target.with_name(target.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            start = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={start}-"} if start else {}
            r = session.get(url, headers=headers, stream=True, timeout=(20, 180))
            if r.status_code in (401, 403):
                r.close()
                raise PermissionError("PhysioNet 401/403")
            if start and r.status_code == 206:
                mode = "ab"
            elif start and r.status_code == 200:
                mode = "wb"
            else:
                r.raise_for_status()
                mode = "wb"
            with r, part.open(mode) as f:
                for block in r.iter_content(4 * 1024 * 1024):
                    if block:
                        f.write(block)
                f.flush()
                os.fsync(f.fileno())
            if part.stat().st_size <= 0:
                raise IOError("empty download")
            part.replace(target)
            return target
        except PermissionError:
            raise
        except Exception:
            if attempt == retries:
                raise
            time.sleep(attempt * 3)
    return target


def authorize(session):
    r = session.get(JPG_BASE + TABLE_FILES["split"], stream=True, timeout=(20, 60))
    try:
        if r.status_code in (401, 403):
            raise SystemExit(
                "PhysioNet 拒绝访问。请确认已 credentialed、完成 CITI training，"
                "并签署 MIMIC-CXR 与 MIMIC-CXR-JPG DUA。"
            )
        r.raise_for_status()
    finally:
        r.close()


def support_files(args, username, password):
    s = session_for(username, password, args.network_mode)
    authorize(s)
    for name in TABLE_FILES.values():
        target = DATA / name
        if not target.exists():
            print(f"下载元数据 {name}", flush=True)
            stream_download(s, JPG_BASE + name, target)
    archive = DATA / REPORT_ARCHIVE
    if not archive.exists():
        print("下载 radiology reports 压缩包（不下载 DICOM）", flush=True)
        stream_download(s, CXR_BASE + REPORT_ARCHIVE, archive)
    with zipfile.ZipFile(archive) as zf:
        if not zf.namelist():
            raise ValueError("报告压缩包为空")
    return archive


def read_labels(path: Path):
    rows = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fields = [x for x in r.fieldnames if x not in ("subject_id", "study_id")]
        for row in r:
            key = (str(row["subject_id"]), str(row["study_id"]))
            rows[key] = {field: row.get(field, "") for field in fields}
    return rows, fields


def read_split(path: Path):
    studies = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            subject_id = str(row["subject_id"])
            study_id = str(row["study_id"])
            key = (subject_id, study_id)
            item = studies.setdefault(key, {
                "subject_id": subject_id,
                "study_id": study_id,
                "split": row["split"],
                "dicom_ids": [],
            })
            if item["split"] != row["split"]:
                raise ValueError(f"同一 study 出现多个 split: {key}")
            item["dicom_ids"].append(str(row["dicom_id"]))
    return studies


def report_index(archive: Path):
    mapping = {}
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            base = Path(name).name
            if base.startswith("s") and base.endswith(".txt") and base[1:-4].isdigit():
                study_id = base[1:-4]
                mapping[study_id] = name
    return mapping


def extract_reports(archive: Path, report_map, study_ids):
    result = {}
    with zipfile.ZipFile(archive) as zf:
        for i, study_id in enumerate(sorted(study_ids), 1):
            member = Path(report_map[study_id])
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"非法压缩包路径: {member}")
            target = DATA / member
            result[study_id] = target.relative_to(DATA).as_posix()
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                raw = zf.read(report_map[study_id])
                if not raw.strip():
                    continue
                tmp = target.with_name(target.name + ".tmp")
                tmp.write_bytes(raw)
                tmp.replace(target)
            if i % 5000 == 0:
                print(f"报告提取 {i}/{len(study_ids)}", flush=True)
    return result


def selected_metadata(path: Path, dicom_ids):
    out = {}
    wanted = set(dicom_ids)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            did = str(row["dicom_id"])
            if did in wanted:
                out[did] = {
                    "ViewPosition": row.get("ViewPosition", ""),
                    "Rows": row.get("Rows", ""),
                    "Columns": row.get("Columns", ""),
                    "StudyDate": row.get("StudyDate", ""),
                    "StudyTime": row.get("StudyTime", ""),
                }
    return out


def image_path(subject_id, study_id, dicom_id):
    return (
        Path("files") / f"p{subject_id[:2]}" / f"p{subject_id}" /
        f"s{study_id}" / f"{dicom_id}.jpg"
    ).as_posix()


def prepare_subset(args, archive):
    labels, label_fields = read_labels(DATA / TABLE_FILES[args.label_source])
    studies = read_split(DATA / TABLE_FILES["split"])
    reports = report_index(archive)

    chosen = []
    excluded = defaultdict(int)
    for key, study in studies.items():
        if len(study["dicom_ids"]) < args.min_images:
            excluded["too_few_images"] += 1
            continue
        if key not in labels:
            excluded["missing_labels"] += 1
            continue
        if study["study_id"] not in reports:
            excluded["missing_report"] += 1
            continue
        chosen.append(study)

    chosen.sort(key=lambda x: (int(x["subject_id"]), int(x["study_id"])))
    if args.max_studies:
        chosen = chosen[:args.max_studies]

    report_paths = extract_reports(archive, reports, {x["study_id"] for x in chosen})

    final = []
    for s in chosen:
        p = DATA / report_paths[s["study_id"]]
        if p.exists() and p.read_text(encoding="utf-8", errors="replace").strip():
            final.append(s)
        else:
            excluded["empty_report"] += 1

    all_dicoms = [d for s in final for d in s["dicom_ids"]]
    meta = selected_metadata(DATA / TABLE_FILES["metadata"], all_dicoms)

    sample_rows, image_rows = [], []
    positive = {x: 0 for x in label_fields}
    uncertain = {x: 0 for x in label_fields}
    missing = {x: 0 for x in label_fields}
    split_studies, split_images = defaultdict(int), defaultdict(int)

    for s in final:
        key = (s["subject_id"], s["study_id"])
        lrow = labels[key]
        split_studies[s["split"]] += 1
        split_images[s["split"]] += len(s["dicom_ids"])
        for label in label_fields:
            v = (lrow.get(label) or "").strip()
            if v in ("1", "1.0"):
                positive[label] += 1
            elif v in ("-1", "-1.0"):
                uncertain[label] += 1
            elif v == "":
                missing[label] += 1

        paths, views = [], []
        for did in sorted(s["dicom_ids"]):
            rel = image_path(s["subject_id"], s["study_id"], did)
            m = meta.get(did, {})
            paths.append(rel)
            views.append(m.get("ViewPosition", ""))
            image_rows.append({
                "subject_id": s["subject_id"],
                "study_id": s["study_id"],
                "dicom_id": did,
                "split": s["split"],
                "view_position": m.get("ViewPosition", ""),
                "rows": m.get("Rows", ""),
                "columns": m.get("Columns", ""),
                "study_date": m.get("StudyDate", ""),
                "study_time": m.get("StudyTime", ""),
                "image_path": rel,
            })

        row = {
            "subject_id": s["subject_id"],
            "study_id": s["study_id"],
            "split": s["split"],
            "image_count": len(paths),
            "image_paths_json": json.dumps(paths),
            "views_json": json.dumps(views),
            "report_path": report_paths[s["study_id"]],
        }
        row.update(lrow)
        sample_rows.append(row)

    write_csv(DATA / "samples_multiview.csv", sample_rows, [
        "subject_id", "study_id", "split", "image_count",
        "image_paths_json", "views_json", "report_path", *label_fields,
    ])
    write_csv(DATA / "images_multiview.csv", image_rows, [
        "subject_id", "study_id", "dicom_id", "split", "view_position",
        "rows", "columns", "study_date", "study_time", "image_path",
    ])

    manifest = {
        "created_at": now(),
        "source": {
            "mimic_cxr_jpg": "2.1.0",
            "mimic_cxr": "2.1.0",
            "label_source": args.label_source,
        },
        "selection": {
            "minimum_images_per_study": args.min_images,
            "require_report": True,
            "require_labels": True,
            "reports_modified": False,
            "labels_modified": False,
            "max_studies_debug_limit": args.max_studies or None,
        },
        "counts": {
            "studies": len(sample_rows),
            "patients": len({r["subject_id"] for r in sample_rows}),
            "images": len(image_rows),
            "reports": len(sample_rows),
            "split_studies": dict(split_studies),
            "split_images": dict(split_images),
            "excluded": dict(excluded),
        },
        "label_fields": label_fields,
        "positive_counts": positive,
        "uncertain_counts": uncertain,
        "not_stated_counts": missing,
        "label_semantics": {
            "1": "positive", "0": "negative", "-1": "uncertain", "blank": "not stated"
        },
        "note": (
            "胸片多视图不等同于胃镜/CT长序列。images_multiview.csv中的排序仅为确定性文件排序，"
            "后续训练不应把它解释为真实采集时序。"
        ),
    }
    write_json(DATA / "selection_manifest.json", manifest)
    write_json(OUT / "selection_summary.json", manifest)
    print(
        f"筛选完成: {len(sample_rows)} studies, "
        f"{manifest['counts']['patients']} patients, {len(image_rows)} images",
        flush=True,
    )
    return image_rows, manifest


def validate_jpg(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return False, None
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            return True, [im.width, im.height]
    except Exception:
        return False, None


def download_image(args, username, password, row):
    target = DATA / row["image_path"]
    ok, size = validate_jpg(target)
    if ok:
        return {"status": "existing", "path": row["image_path"], "bytes": target.stat().st_size, "pixel_size": size}
    if target.exists():
        return {"status": "failed", "path": row["image_path"], "error": "existing_final_jpg_failed_validation"}
    try:
        s = session_for(username, password, args.network_mode)
        stream_download(s, JPG_BASE + row["image_path"], target)
        ok, size = validate_jpg(target)
        if not ok:
            return {"status": "failed", "path": row["image_path"], "error": "downloaded_jpg_failed_validation"}
        h = hashlib.sha256()
        with target.open("rb") as f:
            for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                h.update(block)
        return {
            "status": "downloaded", "path": row["image_path"],
            "bytes": target.stat().st_size, "sha256": h.hexdigest(), "pixel_size": size,
        }
    except PermissionError:
        return {"status": "failed", "path": row["image_path"], "error": "physionet_access_denied"}
    except Exception as exc:
        return {"status": "failed", "path": row["image_path"], "error": type(exc).__name__}


def run_download(args, username, password, image_rows, manifest):
    reserve = args.reserve_free_gib * 1024**3
    if shutil.disk_usage(DATA).free < reserve:
        raise ValueError("项目盘空闲空间低于保留余量")

    verified_path = OUT / "verified_files.jsonl"
    failed_path = OUT / "failed_files.json"
    verified_path.write_text("", encoding="utf-8")
    failures = []
    state = {
        "status": "downloading", "started_at": now(), "pid": os.getpid(),
        "studies": manifest["counts"]["studies"], "total_images": len(image_rows),
        "completed_images": 0, "existing_images": 0, "downloaded_images": 0,
        "failed_images": 0, "completed_bytes": 0, "workers": args.workers,
        "data_root": str(DATA), "network_mode": args.network_mode,
    }
    write_json(OUT / "status.json", state)

    started = time.monotonic()
    with ThreadPoolExecutor(args.workers) as pool:
        iterator = iter(image_rows)
        pending = set()
        max_pending = max(32, args.workers * 8)

        def fill():
            while len(pending) < max_pending:
                try:
                    row = next(iterator)
                except StopIteration:
                    return
                pending.add(pool.submit(download_image, args, username, password, row))

        fill()
        last_print = 0
        while pending:
            done, pending = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            verified = []
            for fut in done:
                result = fut.result()
                if result["status"] == "failed":
                    failures.append(result)
                else:
                    verified.append(result)
                    state["completed_images"] += 1
                    state["completed_bytes"] += int(result.get("bytes", 0))
                    state["existing_images"] += int(result["status"] == "existing")
                    state["downloaded_images"] += int(result["status"] == "downloaded")

            if verified:
                with verified_path.open("a", encoding="utf-8") as f:
                    for row in verified:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

            state["failed_images"] = len(failures)
            state["elapsed_seconds"] = round(time.monotonic() - started, 1)
            state["updated_at"] = now()
            state["free_gib"] = round(shutil.disk_usage(DATA).free / 1024**3, 2)
            write_json(OUT / "status.json", state)
            write_json(failed_path, failures)

            if shutil.disk_usage(DATA).free < reserve:
                state["status"] = "stopped_low_disk"
                write_json(OUT / "status.json", state)
                raise RuntimeError("剩余空间低于保留余量，停止下载；已完成文件保留，可续跑。")

            processed = state["completed_images"] + state["failed_images"]
            if processed - last_print >= 100 or not pending:
                print(
                    f"{processed}/{len(image_rows)}; 成功 {state['completed_images']}; "
                    f"失败 {state['failed_images']}; {state['completed_bytes']/1e9:.2f} GB; "
                    f"剩余 {state['free_gib']:.1f} GiB",
                    flush=True,
                )
                last_print = processed
            fill()

    state["status"] = "complete" if not failures else "incomplete"
    state["finished_at"] = now()
    write_json(OUT / "status.json", state)
    write_json(failed_path, failures)
    print("下载完成。" if not failures else f"下载结束，失败 {len(failures)} 个文件。", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--username", default=os.environ.get("PHYSIONET_USERNAME"))
    p.add_argument("--label-source", choices=("chexpert", "negbio"), default="chexpert")
    p.add_argument("--min-images", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--max-studies", type=int, default=0, help="仅用于小规模测试；0=不限")
    p.add_argument("--network-mode", choices=("direct", "v2raya"), default="direct")
    p.add_argument("--reserve-free-gib", type=int, default=20)
    args = p.parse_args()
    if args.min_images < 2:
        p.error("--min-images 必须 >= 2")
    if not 1 <= args.workers <= 8:
        p.error("--workers 必须在 1..8")
    return args


def main():
    args = parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    username, password = credentials(args)

    print(f"ROOT={ROOT}", flush=True)
    print(f"DATA={DATA}", flush=True)
    print(f"OUT={OUT}", flush=True)

    with (OUT / "download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("已有 MIMIC-CXR 下载进程在运行") from exc

        archive = support_files(args, username, password)
        image_rows, manifest = prepare_subset(args, archive)
        if args.prepare_only:
            write_json(OUT / "status.json", {
                "status": "prepared", "updated_at": now(),
                "studies": manifest["counts"]["studies"],
                "patients": manifest["counts"]["patients"],
                "images": manifest["counts"]["images"],
            })
            print("prepare-only 完成，尚未下载 JPG。", flush=True)
            return
        run_download(args, username, password, image_rows, manifest)


if __name__ == "__main__":
    main()
