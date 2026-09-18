#!/usr/bin/env python3
"""审计AMOS-MM 5964损坏证据；只记录结论，绝不补零或替换CT。"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/amos_mm/repair_amos_5964"
TARGET = ROOT / "datasets/amos_mm/imagesTr/amos_5964.nii.gz"
REJECTED = OUT / "amos_5964.zenodo_mismatched.nii.gz"
EXPECTED_OFFICIAL_SHA256 = "844aa95e45beab856fd1fc459ac956416c7d2ab4a55f23af520600f126a0110a"
EXPECTED_REJECTED_SHA256 = "2c7f073b79467de46d25aac6165b4c69ad6f28c56cec8d7db80a7910fb98e65a"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gzip_error(path):
    try:
        gzip.decompress(path.read_bytes())
    except (EOFError, OSError) as error:
        return f"{type(error).__name__}: {error}"
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    official_entry = json.loads((OUT / "source_entry.json").read_text())
    rejected_entry = json.loads((OUT / "source_entry.zenodo_mismatched.json").read_text())
    assert sha256(TARGET) == EXPECTED_OFFICIAL_SHA256
    assert TARGET.stat().st_size == official_entry["bytes"] == 4194304
    official_error = gzip_error(TARGET)
    assert official_error is not None, "官方文件现在可以完整解压，请重新评估排除策略"
    assert sha256(REJECTED) == EXPECTED_REJECTED_SHA256
    assert gzip_error(REJECTED) is None
    rejected_image = nib.load(REJECTED)
    assert list(rejected_image.shape) == rejected_entry["shape"] == [768, 768, 74]
    assert official_entry["shape"] == [512, 512, 202]
    conclusion = {
        "scan_id": "amos_5964",
        "decision": "exclude_from_pure_image_and_multimodal_development_only",
        "official_file": {
            "path": str(TARGET),
            "sha256": EXPECTED_OFFICIAL_SHA256,
            "bytes": TARGET.stat().st_size,
            "header_shape": official_entry["shape"],
            "gzip_integrity": False,
            "error": official_error,
        },
        "rejected_replacement": {
            "path": str(REJECTED),
            "sha256": EXPECTED_REJECTED_SHA256,
            "shape": list(rejected_image.shape),
            "reason": "Geometry and uncompressed content differ; this is not the AMOS-MM CT.",
        },
        "safety": "No file was modified. Zero-padding missing voxels or substituting a different scan is prohibited.",
        "time": time.time(),
    }
    temporary = OUT / "repair_conclusion.tmp.json"
    temporary.write_text(json.dumps(conclusion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUT / "repair_conclusion.json")
    print("审计完成：官方文件不可恢复，已确认采用影像开发集排除策略。", flush=True)


if __name__ == "__main__":
    main()
