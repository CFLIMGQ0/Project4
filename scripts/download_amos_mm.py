#!/usr/bin/env python3
"""通过 V2rayA 下载 AMOS-MM 原始图像和报告；按 ZIP 条目流式解包并校验。"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import io
import json
import logging
import os
import shutil
import struct
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

import nibabel as nib
import requests


ROOT = Path(__file__).resolve().parents[2]
GIB = 1024 ** 3
SOURCES = {
    "imagesTr": {"id": "1y0fKm_SSa3uriBLrh1pez34jhclAOglA", "bytes": 59642919535},
    "imagesVa": {"id": "1w8HFuWqauIgVQMZVt7XefD6QibmWJHBL", "bytes": 19138381316},
    "report_generation_train_val.json": {
        "id": "14ZIhEM1IDvrj6JkIjjCju-YAjLaJ22As", "bytes": 4070405,
    },
    "vqa_train_val.json": {"id": "1VPvuMvbXo2puVj-rcN-vZy4oksKJhA_W", "bytes": 16597409},
}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def session(proxy):
    result = requests.Session()
    result.trust_env = False
    result.proxies = {"http": proxy, "https": proxy}
    result.headers.update({"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"})
    return result


def source_url(source):
    return ("https://drive.usercontent.google.com/download?id=" + source["id"]
            + "&export=download&confirm=t")


def get_range(client, source, start, end):
    response = client.get(source_url(source), headers={"Range": f"bytes={start}-{end}"},
                          stream=True, timeout=(15, 60))
    expected = f"bytes {start}-{end}/{source['bytes']}"
    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
        status, value = response.status_code, response.headers.get("Content-Range")
        response.close()
        raise IOError(f"远端分段响应异常：{status}, {value}; 预期 {expected}")
    if response.headers.get("Content-Encoding", "identity") != "identity":
        response.close()
        raise IOError("分段响应出现额外内容编码")
    return response


class RemoteZip(io.RawIOBase):
    """只读取中央目录的小块数据，避免缓存整个 ZIP。"""

    def __init__(self, source, client):
        self.source, self.client, self.pos, self.cache = source, client, 0, {}

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (0 if whence == 0 else self.pos if whence == 1 else self.source["bytes"]) + offset
        if self.pos < 0:
            raise ValueError("负偏移")
        return self.pos

    def read(self, n=-1):
        n = min(self.source["bytes"] - self.pos, n if n >= 0 else self.source["bytes"])
        if n <= 0:
            return b""
        if n > 4 * 1024 ** 2:
            raise ValueError("中央目录读取超过安全大小")
        block = 256 * 1024
        start = self.pos // block * block
        end = min(self.source["bytes"] - 1, ((self.pos + n + block - 1) // block) * block - 1)
        key = (start, end)
        if key not in self.cache:
            with get_range(self.client, self.source, start, end) as response:
                data = response.raw.read(end - start + 2)
            if len(data) != end - start + 1:
                raise IOError("中央目录分段长度不符")
            self.cache[key] = data
        data = self.cache[key][self.pos - start:self.pos - start + n]
        self.pos += len(data)
        return data


def read_index(split, client):
    source = SOURCES[split]
    with zipfile.ZipFile(RemoteZip(source, client)) as archive:
        infos = sorted(archive.infolist(), key=lambda item: item.header_offset)
        rows = []
        for i, info in enumerate(infos):
            path = Path(info.filename)
            if path.parent.as_posix() != split or not path.name.startswith("amos_") or not path.name.endswith(".nii.gz"):
                continue
            if info.flag_bits & 1 or info.compress_type not in (0, 8):
                raise ValueError(f"不支持的 ZIP 条目：{info.filename}")
            rows.append({
                "split": split, "filename": info.filename, "scan_id": path.name.removesuffix(".nii.gz"),
                "bytes": info.file_size, "compressed_bytes": info.compress_size,
                "crc32": info.CRC, "method": info.compress_type,
                "header_offset": info.header_offset,
                "range_end": (infos[i + 1].header_offset if i + 1 < len(infos) else archive.start_dir) - 1,
            })
    return rows


def fetch_json(name, root, client):
    target, source = root / name, SOURCES[name]
    if target.exists():
        if target.stat().st_size != source["bytes"]:
            raise ValueError(f"已有元数据大小异常，请人工核查：{target}")
        raw = target.read_bytes()
    else:
        with get_range(client, source, 0, source["bytes"] - 1) as response:
            raw = response.raw.read(source["bytes"] + 1)
        if len(raw) != source["bytes"]:
            raise IOError("元数据长度异常")
        json.loads(raw)
        tmp = target.with_name(target.name + ".part")
        tmp.write_bytes(raw)
        tmp.replace(target)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def inspect_volume(path):
    # 部分下载使用 .part 后缀，直接读取压缩 NIfTI 头而非依赖文件扩展名。
    with gzip.open(path, "rb") as stream:
        header = nib.Nifti1Header.from_fileobj(stream)
    shape = list(header.get_data_shape())
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError(f"NIfTI 维度异常：{shape}")
    return {"shape": shape, "spacing": [float(v) for v in header.get_zooms()]}


def check_file(path, row):
    crc, size, sha = 0, 0, hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2 * 1024 ** 2), b""):
            crc = zlib.crc32(block, crc)
            size += len(block)
            sha.update(block)
    if size != row["bytes"] or crc != row["crc32"]:
        raise ValueError(f"已有文件与官方 ZIP 的大小/CRC32 不符，保留待核查：{path}")
    return {"sha256": sha.hexdigest(), **inspect_volume(path)}


def unpack_response(response, row, target):
    """流式还原单个 .nii.gz；校验通过后再改为正式文件名。"""
    raw = response.raw
    header = raw.read(30)
    if len(header) != 30:
        raise IOError("ZIP 本地文件头不完整")
    signature, _, flags, method, _, _, _, _, _, name_len, extra_len = struct.unpack("<4s5H3I2H", header)
    if signature != b"PK\x03\x04" or flags & 1 or method != row["method"]:
        raise ValueError("ZIP 本地头与中央目录不符")
    name = raw.read(name_len).decode("utf-8" if flags & 0x800 else "cp437")
    extra = raw.read(extra_len)
    if name != row["filename"] or len(extra) != extra_len:
        raise ValueError("ZIP 条目名称或扩展字段不符")
    inflater = zlib.decompressobj(-zlib.MAX_WBITS) if method == 8 else None
    remain, size, crc, sha = row["compressed_bytes"], 0, 0, hashlib.sha256()
    tmp = target.with_name(target.name + ".part")
    with tmp.open("wb") as stream:
        while remain:
            block = raw.read(min(remain, 1024 ** 2))
            if not block:
                raise IOError("图像分段提前结束")
            remain -= len(block)
            data = inflater.decompress(block) if inflater else block
            size += len(data)
            if size > row["bytes"]:
                raise ValueError("解包输出超过中央目录声明大小")
            stream.write(data)
            crc = zlib.crc32(data, crc)
            sha.update(data)
        if inflater and (not inflater.eof or inflater.unused_data):
            raise ValueError("DEFLATE 数据流异常")
        stream.flush()
    if size != row["bytes"] or crc != row["crc32"]:
        raise ValueError("图像字节数或官方 ZIP CRC32 校验失败")
    detail = inspect_volume(tmp)
    tmp.replace(target)
    return {"sha256": sha.hexdigest(), **detail}


def prepare(args):
    root, out = args.data_root, args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    args.train_root.mkdir(parents=True, exist_ok=True)
    alias = root / "imagesTr"
    if os.path.lexists(alias):
        if not alias.is_symlink() or alias.resolve() != args.train_root.resolve():
            raise ValueError(f"训练目录入口已存在且目标不符，保留原目录：{alias}")
    else:
        alias.symlink_to(args.train_root.resolve(), target_is_directory=True)
    (root / "imagesVa").mkdir(exist_ok=True)
    with session(args.proxy) as client:
        index_path = out / "archive_index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            if index["sources"] != SOURCES:
                raise ValueError("缓存来源版本不符")
        else:
            rows = []
            for split in ("imagesTr", "imagesVa"):
                rows.extend(read_index(split, client))
                logging.info("已读取 %s 官方目录", split)
            if len(rows) != 1690 or len({r['scan_id'] for r in rows}) != 1687:
                raise ValueError("当前官方图像目录数量或编号发生变化，请重新核查")
            index = {"created_at": timestamp(), "sources": SOURCES, "files": rows}
            write_json(index_path, index)
        hashes = {}
        report, hashes["report_generation_train_val.json"] = fetch_json("report_generation_train_val.json", root, client)
        _, hashes["vqa_train_val.json"] = fetch_json("vqa_train_val.json", root, client)
    report_rows = {}
    for split in ("training", "validation", "testing"):
        for record in report.get(split, []):
            name = Path(record["image"]).name.removesuffix(".nii.gz")
            if name in report_rows:
                raise ValueError("官方报告存在重复图像编号")
            report_rows[name] = {"official_split": split, "image": record["image"]}
    archive_images = {r["filename"]: r for r in index["files"]}
    images = {}
    for scan_id, record in report_rows.items():
        official_path = Path(record["image"]).as_posix()
        if official_path in archive_images:
            images[scan_id] = archive_images[official_path]
    skipped = [r for r in index["files"] if r["filename"] not in {v["filename"] for v in images.values()}]
    for row in skipped:
        chosen = images.get(row["scan_id"])
        if chosen is None or any(row[key] != chosen[key] for key in ("bytes", "crc32")):
            raise ValueError("压缩包中存在尚未核实的额外图像，保留目录信息并等待核查")
    candidate_path = ROOT / "outputs/amos_mm/leavs_label_audit/完整三标签候选.csv"
    with candidate_path.open(encoding="utf-8-sig", newline="") as stream:
        candidates = list(csv.DictReader(stream))
    missing_reports = sorted(set(report_rows) - set(images))
    missing_candidates = sorted({r["scan_id"] for r in candidates} - (set(images) & set(report_rows)))
    mapping = {
        "created_at": timestamp(), "official_page": "https://era-ai-biomed.github.io/amos/dataset.html",
        "data_root": str(root), "training_storage": str(args.train_root.resolve()),
        "validation_storage": str((root / "imagesVa").resolve()), "proxy": args.proxy,
        "source_license": report.get("licence"), "source_local_sha256": hashes,
        "official_archive_ct_entries": len(index["files"]), "download_ct_files": len(images),
        "report_cases": len(report_rows), "skipped_duplicate_entries": skipped,
        "candidate_cases": len(candidates), "images_without_report": sorted(set(images) - set(report_rows)),
        "reports_without_image": missing_reports, "candidates_without_image_or_report": missing_candidates,
        "report_text_modified": False, "minimum_free_gib": args.reserve_gib,
        "note": "候选标签来自前期 LEAVS 审计；图像与报告按检查编号配对，不代表已核实患者唯一性。",
    }
    write_json(out / "data_mapping.json", mapping)
    if missing_reports or missing_candidates:
        raise ValueError("图像与报告/候选清单存在未匹配编号，已记录，请人工核查")
    with (out / "candidate_image_paths.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidates[0]) + ["image_path", "official_split"])
        writer.writeheader()
        for row in candidates:
            writer.writerow({**row, "image_path": str(root / images[row["scan_id"]]["filename"]),
                             "official_split": report_rows[row["scan_id"]]["official_split"]})
    selected_names = {r["filename"] for r in images.values()}
    groups = [[r for r in index["files"] if r["split"] == split and r["filename"] in selected_names]
              for split in ("imagesTr", "imagesVa")]
    return [row for pair in zip_longest(*groups) for row in pair if row is not None]


def run(args, rows):
    out, root = args.output_dir, args.data_root
    manifest = out / "verified_files.jsonl"
    verified = {}
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                verified[record["filename"]] = record
    # 续跑时保留已完成条目；路径或文件变化时重新核验，不覆盖异常正式文件。
    done = {}
    for row in rows:
        path, record = root / row["filename"], verified.get(row["filename"])
        if record and path.exists() and path.stat().st_size == row["bytes"] and (
            record.get("mtime_ns") == path.stat().st_mtime_ns
            and record.get("path") == str(path.resolve()) and record.get("crc32") == row["crc32"]
        ):
            done[row["filename"]] = record
    pending = [r for r in rows if r["filename"] not in done]
    if args.limit:
        pending = pending[:args.limit]
    errors, lock, stop = {}, threading.Lock(), threading.Event()
    start, initial_bytes = time.monotonic(), sum(r["bytes"] for r in done.values())
    total_bytes = sum(r["bytes"] for r in rows)

    def status(state):
        size = sum(r["bytes"] for r in done.values())
        elapsed = time.monotonic() - start
        write_json(out / "status.json", {
            "updated_at": timestamp(), "state": state, "pid": os.getpid(),
            "completed_files": len(done), "total_files": len(rows),
            "verified_bytes": size, "total_bytes": total_bytes,
            "progress_percent": round(100 * size / total_bytes, 3),
            "this_run_mib_per_second": round((size - initial_bytes) / max(elapsed, 1) / 1024 ** 2, 3),
            "elapsed_seconds": round(elapsed), "failed_files": errors,
            "free_gib": {s: round(shutil.disk_usage(root / s).free / GIB, 2) for s in ("imagesTr", "imagesVa")},
        })

    def worker(row):
        target = root / row["filename"]
        if stop.is_set():
            return None
        if target.exists():
            detail = check_file(target, row)
        else:
            with session(args.proxy) as client:
                for attempt in range(1, args.retries + 1):
                    if stop.is_set():
                        return None
                    # 为所有并发任务预留最大条目写入量，防止写满分区。
                    reserve = args.reserve_gib * GIB + args.workers * 300 * 1024 ** 2
                    if shutil.disk_usage(target.parent).free < reserve:
                        stop.set()
                        raise OSError(f"{target.parent} 空闲低于安全余量，已暂停")
                    try:
                        with get_range(client, SOURCES[row["split"]], row["header_offset"], row["range_end"]) as response:
                            detail = unpack_response(response, row, target)
                        break
                    except Exception as exc:
                        if attempt == args.retries:
                            raise
                        logging.warning("%s 第 %d 次下载失败：%s", row["filename"], attempt, exc)
                        stop.wait(min(5 * attempt, 30))
        record = {**row, **detail, "path": str(target.resolve()),
                  "mtime_ns": target.stat().st_mtime_ns, "verified_at": timestamp()}
        with lock:
            with manifest.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            done[row["filename"]] = record
            status("downloading")
            logging.info("已校验 %d/%d（%.2f GiB）：%s", len(done), len(rows),
                         sum(r["bytes"] for r in done.values()) / GIB, row["filename"])
        return record

    status("downloading")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, row): row for row in pending}
        for future in as_completed(futures):
            row = futures[future]
            try:
                future.result()
            except Exception as exc:
                with lock:
                    errors[row["filename"]] = str(exc)
                    status("downloading")
                logging.error("%s 未完成：%s", row["filename"], exc)
    final = "complete" if len(done) == len(rows) else "paused" if stop.is_set() else "incomplete"
    status(final)
    logging.info("本轮结束：%s，已校验 %d/%d", final, len(done), len(rows))
    return 0 if final == "complete" or (args.limit and not errors) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets/amos_mm")
    parser.add_argument("--train-root", type=Path, default=Path("/home/Lim/datasets/amos_mm/imagesTr"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/amos_mm/download")
    parser.add_argument("--proxy", default="http://127.0.0.1:21171")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reserve-gib", type=float, default=20)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="只下载若干个待完成条目，用于冒烟测试")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.reserve_gib < 1 or args.limit < 0:
        parser.error("并发数、安全余量必须为正；limit 不能为负")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "download.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = prepare(args)
        if args.prepare_only:
            logging.info("已完成目录与元数据核对：%d 个 CT 文件", len(rows))
            return 0
        return run(args, rows)


if __name__ == "__main__":
    raise SystemExit(main())
