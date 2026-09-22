#!/usr/bin/env python3
"""MR-RATE-1K：按CT-RATE协议准备4标签、findings掩码、固定五折与64实例ConvNeXt缓存。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/mr_rate_1k"
OUT = ROOT / "outputs/mr_rate_1k/experiment"

LABELS = [
    "PP_Unspecific_bucket",
    "PP_Neurodegenerative",
    "PP_Cerebrovascular",
    "PP_Neoplastic",
]

# 对所有病例统一使用，不根据病例标签决定是否mask。
TARGET_PATTERN = re.compile(
    r"\b(?:"
    r"gliosis|gliotic|"
    r"cerebral\s+edema|brain\s+edema|edema|oedema|"
    r"encephalomalacia|encephalomalacic|"
    r"cerebral\s+atrophy|brain\s+atrophy|atroph(?:y|ic)|"
    r"ventriculomegaly|"
    r"cerebellar\s+degeneration|cerebellar\s+atrophy|"
    r"cerebral\s+infarct(?:ion)?|"
    r"lacunar\s+infarct(?:ion)?|"
    r"watershed\s+infarct(?:ion)?|"
    r"infarct(?:ion)?|"
    r"cerebral\s+h(?:a)?emorrhage|"
    r"intracranial\s+h(?:a)?emorrhage|"
    r"h(?:a)?emorrhage|"
    r"micro[- ]?h(?:a)?emorrhage|microbleed(?:s)?|"
    r"cavernous\s+hemangioma|cavernoma|"
    r"subdural\s+h(?:a)?ematoma|"
    r"intracranial\s+aneurysm|aneurysm|"
    r"brain\s+metasta(?:sis|ses)|metasta(?:sis|ses|tic)|"
    r"intracranial\s+meningioma|meningioma|"
    r"glioma|pituitary\s+adenoma|schwannoma"
    r")\b",
    re.I,
)


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        return list(csv.DictReader(stream))


def safe_float(value, default=1e12):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_manifest():
    rows = read_csv(DATA / "samples_1000.csv")

    assert len(rows) == 1000
    assert len({r["patient_uid"] for r in rows}) == 1000
    assert len({r["study_uid"] for r in rows}) == 1000

    rows.sort(key=lambda r: int(r["case_index"]))

    assert [
        int(r["case_index"])
        for r in rows
    ] == list(range(1000))

    return rows


def load_metadata(target_studies):
    result = {
        uid: []
        for uid in target_studies
    }

    for path in sorted(
        (DATA / "metadata").glob("*_metadata.csv")
    ):
        for row in read_csv(path):
            uid = row.get("study_uid", "")

            if uid in result:
                result[uid].append(row)

    missing = [
        uid
        for uid, rows in result.items()
        if not rows
    ]

    if missing:
        raise RuntimeError(
            f"{len(missing)}个study缺少metadata：{missing[:5]}"
        )

    return result


def load_reports(target_studies):
    result = {}

    for path in sorted(
        (DATA / "reports").glob("*_reports.csv")
    ):
        for row in read_csv(path):
            uid = row.get("study_uid", "")

            if uid in target_studies:
                result[uid] = row

    missing = sorted(
        target_studies - set(result)
    )

    if missing:
        raise RuntimeError(
            f"{len(missing)}个study缺少report：{missing[:5]}"
        )

    return result


def build_zip_map(target_studies):
    result = {}

    for path in sorted(
        (DATA / "mri").glob("batch*/*.zip")
    ):
        if path.stem in target_studies:
            result[path.stem] = path

    missing = sorted(
        target_studies - set(result)
    )

    if missing:
        raise RuntimeError(
            f"{len(missing)}个study缺少MRI ZIP：{missing[:5]}"
        )

    return result


def prepare_tables():
    manifest = load_manifest()

    target_studies = {
        r["study_uid"]
        for r in manifest
    }

    metadata = load_metadata(
        target_studies
    )

    reports = load_reports(
        target_studies
    )

    zips = build_zip_map(
        target_studies
    )

    rows = []
    mask_counter = {}

    for item in tqdm(
        manifest,
        desc="整理MR-RATE-1K",
    ):
        case_index = int(
            item["case_index"]
        )

        study_uid = item["study_uid"]
        patient_uid = item["patient_uid"]

        findings = re.sub(
            r"\s+",
            " ",
            reports[study_uid].get(
                "findings",
                "",
            ) or "",
        ).strip()

        if not findings:
            raise ValueError(
                f"{study_uid} findings为空；"
                "为保持CT-RATE协议，不回退到impression"
            )

        hits = [
            m.group(0).lower()
            for m in TARGET_PATTERN.finditer(
                findings
            )
        ]

        for hit in hits:
            mask_counter[hit] = (
                mask_counter.get(hit, 0) + 1
            )

        masked = TARGET_PATTERN.sub(
            " MASKTARGET ",
            findings,
        )

        masked = re.sub(
            r"\s+",
            " ",
            masked,
        ).strip()

        assert (
            TARGET_PATTERN.search(masked)
            is None
        )

        series = []

        for meta in metadata[study_uid]:
            series_id = (
                meta.get("series_id")
                or ""
            ).strip()

            if not series_id:
                continue

            series.append({
                "series_id":
                    series_id,

                "series_number":
                    safe_float(
                        meta.get(
                            "SeriesNumber"
                        )
                    ),

                "classified_modality":
                    (
                        meta.get(
                            "classified_modality"
                        )
                        or ""
                    ).strip(),

                "acquisition_plane":
                    (
                        meta.get(
                            "acquisition_plane"
                        )
                        or ""
                    ).strip().upper(),

                "array_shape":
                    meta.get(
                        "array_shape",
                        "",
                    ),
            })

        series.sort(
            key=lambda x: (
                x["series_number"],
                x["series_id"],
            )
        )

        if not series:
            raise ValueError(
                f"{study_uid}没有有效series"
            )

        labels = [
            int(item[label])
            for label in LABELS
        ]

        rows.append({
            "case_index":
                case_index,

            "patient_uid":
                patient_uid,

            "study_uid":
                study_uid,

            "fold":
                int(item["fold"]),

            "labels":
                labels,

            "findings_masked":
                masked,

            "mask_hits":
                hits,

            "zip_path":
                str(
                    zips[study_uid]
                    .relative_to(DATA)
                ),

            "series":
                series,
        })

    rows.sort(
        key=lambda x:
            x["case_index"]
    )

    assert [
        r["case_index"]
        for r in rows
    ] == list(range(1000))

    y = np.asarray(
        [r["labels"] for r in rows],
        dtype=np.int64,
    )

    assert y.sum(0).tolist() == [
        379,
        320,
        270,
        231,
    ]

    assert [
        int((y.sum(1) == n).sum())
        for n in range(5)
    ] == [
        350,
        292,
        191,
        142,
        25,
    ]

    folds = [
        np.asarray(
            [
                r["case_index"]
                for r in rows
                if r["fold"] == fold
            ],
            dtype=np.int64,
        )
        for fold in range(1, 6)
    ]

    assert [
        len(group)
        for group in folds
    ] == [200] * 5

    assert sorted(
        np.concatenate(folds).tolist()
    ) == list(range(1000))

    fold_payload = {
        "labels":
            LABELS,

        "patient_ids":
            list(range(1000)),

        "patient_uid_mapping": {
            str(r["case_index"]):
                r["patient_uid"]
            for r in rows
        },

        "study_uid_mapping": {
            str(r["case_index"]):
                r["study_uid"]
            for r in rows
        },

        "positive_counts":
            y.sum(0).tolist(),

        "positive_count_distribution": [
            int(
                (y.sum(1) == n).sum()
            )
            for n in range(5)
        ],

        "folds": [
            group.tolist()
            for group in folds
        ],

        "fold_positive_counts": [
            y[group].sum(0).tolist()
            for group in folds
        ],

        "fold_co_positive_counts": [
            int(
                (
                    y[group].sum(1) >= 2
                ).sum()
            )
            for group in folds
        ],

        "split":
            "test=k，validation=(k+1)%5，"
            "其余三折训练；600/200/200",
    }

    parameters = {
        "dataset":
            "MR-RATE-1K",

        "cases":
            1000,

        "labels":
            LABELS,

        "max_instances":
            64,

        "image_size":
            224,

        "series_order":
            "study内按SeriesNumber排序",

        "instance_sampling":
            "全部MRI series切片串成一个检查级有序序列，"
            "再从完整序列均匀采样最多64实例",

        "position_reference":
            "AMEF使用完整展平序列global index + "
            "original instance count；"
            "instance dropout后不重新编号",

        "mri_intensity":
            "每个series独立P1-P99裁剪缩放[0,1]；"
            "灰度复制为3通道",

        "orientation":
            "NIfTI转换至RAS canonical；"
            "按metadata acquisition_plane取切片",

        "backbone":
            "冻结ImageNet ConvNeXt-Tiny；"
            "224x224；ImageNet normalization；768维",

        "text_field":
            "findings",

        "excluded_text_fields": [
            "impression",
            "clinical_information",
            "technique",
            "report",
        ],

        "target_pattern":
            TARGET_PATTERN.pattern,

        "mask_token":
            "MASKTARGET",

        "raw_reports_modified":
            False,

        "manifest_sha256":
            sha256(
                DATA / "samples_1000.csv"
            ),

        "preparation_source_sha256":
            sha256(
                Path(__file__)
            ),
    }

    parameters[
        "preparation_sha256"
    ] = hashlib.sha256(
        json.dumps(
            parameters,
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()

    prior = (
        OUT
        / "preparation_protocol.json"
    )

    if prior.exists():
        old = json.loads(
            prior.read_text()
        )

        if (
            old.get(
                "preparation_sha256"
            )
            !=
            parameters[
                "preparation_sha256"
            ]
        ):
            raise ValueError(
                "已有MR-RATE缓存采用其他准备协议；"
                "禁止自动覆盖混用"
            )

    save_json(
        OUT / "samples.json",
        rows,
    )

    save_json(
        OUT / "patient_folds.json",
        fold_payload,
    )

    save_json(
        prior,
        parameters,
    )

    save_json(
        OUT / "text_audit.json",
        {
            "cases":
                1000,

            "input_field":
                "findings",

            "excluded_fields": [
                "impression",
                "clinical_information",
                "technique",
                "report",
            ],

            "mask_hit_cases":
                int(
                    sum(
                        bool(r["mask_hits"])
                        for r in rows
                    )
                ),

            "mask_terms":
                mask_counter,

            "residual_target_pattern_hits":
                0,

            "label_provenance":
                "MR-RATE merged-majority "
                "report-derived labels；"
                "统一掩码直接诊断词",
        },
    )

    print(
        "MR-RATE-1K表格、标签、"
        "findings掩码和固定五折已准备。",
        flush=True,
    )


def plane_axis(
    plane,
    zooms,
):
    if plane == "AXIAL":
        return 2

    if plane == "CORONAL":
        return 1

    if plane == "SAGITTAL":
        return 0

    # OBLIQUE或缺失：
    # 最大spacing轴作为through-plane确定性近似
    return int(
        np.argmax(
            np.asarray(
                zooms[:3],
                dtype=np.float64,
            )
        )
    )


def slice_2d(
    data,
    axis,
    index,
):
    if axis == 2:
        image = data[:, :, index].T

    elif axis == 1:
        image = data[:, index, :].T

    else:
        image = data[index, :, :].T

    image = np.flip(
        image,
        axis=(0, 1),
    )

    return np.ascontiguousarray(
        image,
        dtype=np.float32,
    )


def robust_percentiles(data):
    flat = data.reshape(-1)

    step = max(
        1,
        flat.size // 500_000,
    )

    sample = flat[::step]

    sample = sample[
        np.isfinite(sample)
    ]

    nonzero = sample[
        np.abs(sample) > 1e-6
    ]

    if len(nonzero) >= 100:
        sample = nonzero

    if len(sample) == 0:
        raise ValueError(
            "MRI volume没有有限体素"
        )

    lo, hi = np.percentile(
        sample,
        [1, 99],
    )

    lo = float(lo)
    hi = float(hi)

    if (
        not np.isfinite(lo)
        or not np.isfinite(hi)
        or hi <= lo
    ):
        lo = float(
            np.min(sample)
        )

        hi = float(
            np.max(sample)
        )

    if hi <= lo:
        hi = lo + 1.0

    return lo, hi


def feature_worker(
    index,
    devices,
    limit=None,
):
    import nibabel as nib
    import torch
    import torch.nn.functional as F
    from torchvision import models

    torch.set_num_threads(2)

    torch.hub.set_dir(
        str(
            ROOT
            / "pre_weights"
        )
    )

    base_model = (
        models.convnext_tiny(
            weights=
                models
                .ConvNeXt_Tiny_Weights
                .IMAGENET1K_V1
        )
    )

    encoder = torch.nn.Sequential(
        base_model.features,
        base_model.avgpool,
        base_model.classifier[0],
        torch.nn.Flatten(1),
    ).eval().cuda()

    del base_model

    mean = torch.tensor(
        [.485, .456, .406],
        device="cuda",
    )[None, :, None, None]

    std = torch.tensor(
        [.229, .224, .225],
        device="cuda",
    )[None, :, None, None]

    protocol = json.loads(
        (
            OUT
            / "preparation_protocol.json"
        ).read_text()
    )

    digest = protocol[
        "preparation_sha256"
    ]

    rows = json.loads(
        (
            OUT
            / "samples.json"
        ).read_text()
    )

    jobs = [
        row
        for row in rows
        if (
            row["case_index"]
            % len(devices)
            == index
        )
    ]

    if limit is not None:
        jobs = jobs[:limit]

    feature_dir = (
        OUT
        / "features"
    )

    feature_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for done, row in enumerate(
        tqdm(
            jobs,
            desc=(
                f"GPU{devices[index]} "
                "MR-RATE特征"
            ),
        ),
        1,
    ):
        cache_path = (
            feature_dir
            / f"{row['case_index']:04d}.npz"
        )

        if cache_path.exists():
            with np.load(
                cache_path,
                allow_pickle=False,
            ) as cache:
                assert (
                    str(
                        cache[
                            "preparation_sha256"
                        ]
                    )
                    == digest
                )

                assert (
                    str(
                        cache["study_uid"]
                    )
                    ==
                    row["study_uid"]
                )

                assert (
                    cache["features"].shape[1]
                    == 768
                )

                assert np.isfinite(
                    cache["features"]
                ).all()

            continue

        started = time.monotonic()

        zip_path = (
            DATA
            / row["zip_path"]
        )

        if not zip_path.exists():
            raise FileNotFoundError(
                zip_path
            )

        with tempfile.TemporaryDirectory(
            prefix=(
                f"mrrate_"
                f"{row['study_uid']}_"
            )
        ) as tmp_dir:

            tmp_dir = Path(
                tmp_dir
            )

            with zipfile.ZipFile(
                zip_path
            ) as archive:

                members = [
                    member
                    for member
                    in archive.namelist()
                    if (
                        "/img/" in member
                        and member.endswith(
                            ".nii.gz"
                        )
                    )
                ]

                if not members:
                    raise ValueError(
                        f"{row['study_uid']} "
                        "ZIP内没有img/*.nii.gz"
                    )

                member_by_series = {}

                prefix = (
                    row["study_uid"]
                    + "_"
                )

                for member in members:
                    name = Path(
                        member
                    ).name

                    if (
                        name.startswith(
                            prefix
                        )
                        and name.endswith(
                            ".nii.gz"
                        )
                    ):
                        series_id = (
                            name[
                                len(prefix):-7
                            ]
                        )

                        member_by_series[
                            series_id
                        ] = member

                ordered = []

                for meta in row["series"]:
                    series_id = (
                        meta["series_id"]
                    )

                    member = (
                        member_by_series
                        .get(series_id)
                    )

                    if member is None:
                        continue

                    archive.extract(
                        member,
                        tmp_dir,
                    )

                    ordered.append(
                        (
                            meta,
                            tmp_dir / member,
                        )
                    )

            if not ordered:
                raise ValueError(
                    f"{row['study_uid']} "
                    "metadata与ZIP影像无法匹配"
                )

            descriptors = []
            original_count = 0

            for series_index, (
                meta,
                local_path,
            ) in enumerate(ordered):

                image = (
                    nib
                    .as_closest_canonical(
                        nib.load(
                            str(local_path)
                        )
                    )
                )

                if len(
                    image.shape
                ) != 3:
                    continue

                zooms = (
                    image.header
                    .get_zooms()[:3]
                )

                axis = plane_axis(
                    meta[
                        "acquisition_plane"
                    ],
                    zooms,
                )

                count = int(
                    image.shape[axis]
                )

                if count < 1:
                    continue

                descriptors.append({
                    "series_index":
                        series_index,

                    "meta":
                        meta,

                    "path":
                        local_path,

                    "axis":
                        axis,

                    "count":
                        count,

                    "offset":
                        original_count,
                })

                original_count += count

            if original_count < 1:
                raise ValueError(
                    f"{row['study_uid']} "
                    "没有可用MRI切片"
                )

            slice_indices = np.unique(
                np.linspace(
                    0,
                    original_count - 1,
                    min(
                        64,
                        original_count,
                    ),
                )
                .round()
                .astype(np.int64)
            )

            sampled_images = [
                None
            ] * len(slice_indices)

            selected_series = (
                np.empty(
                    len(slice_indices),
                    dtype=np.int64,
                )
            )

            selected_local = (
                np.empty(
                    len(slice_indices),
                    dtype=np.int64,
                )
            )

            selected_series_number = (
                np.empty(
                    len(slice_indices),
                    dtype=np.float32,
                )
            )

            normalization = []

            for descriptor in descriptors:
                start = descriptor[
                    "offset"
                ]

                end = (
                    start
                    + descriptor["count"]
                )

                positions = np.flatnonzero(
                    (
                        slice_indices >= start
                    )
                    &
                    (
                        slice_indices < end
                    )
                )

                if len(positions) == 0:
                    continue

                local_indices = (
                    slice_indices[positions]
                    - start
                )

                image = (
                    nib
                    .as_closest_canonical(
                        nib.load(
                            str(
                                descriptor[
                                    "path"
                                ]
                            )
                        )
                    )
                )

                data = np.asarray(
                    image.dataobj,
                    dtype=np.float32,
                )

                lo, hi = (
                    robust_percentiles(
                        data
                    )
                )

                normalization.append({
                    "series_id":
                        descriptor[
                            "meta"
                        ][
                            "series_id"
                        ],

                    "p1":
                        lo,

                    "p99":
                        hi,

                    "axis":
                        descriptor[
                            "axis"
                        ],

                    "slice_count":
                        descriptor[
                            "count"
                        ],
                })

                for pos, local_index in zip(
                    positions.tolist(),
                    local_indices.tolist(),
                ):
                    image_2d = slice_2d(
                        data,
                        descriptor["axis"],
                        int(local_index),
                    )

                    image_2d = np.nan_to_num(
                        image_2d,
                        nan=lo,
                        posinf=hi,
                        neginf=lo,
                    )

                    image_2d = np.clip(
                        (
                            image_2d - lo
                        )
                        /
                        (
                            hi - lo
                        ),
                        0.0,
                        1.0,
                    ).astype(
                        np.float32
                    )

                    image_2d = (
                        F.interpolate(
                            torch.from_numpy(image_2d)[None, None],
                            size=(224, 224),
                            mode="bilinear",
                            align_corners=False,
                            antialias=True,
                        )
                        .squeeze(0)
                        .squeeze(0)
                        .numpy()
                    )

                    sampled_images[
                        pos
                    ] = image_2d

                    selected_series[
                        pos
                    ] = descriptor[
                        "series_index"
                    ]

                    selected_local[
                        pos
                    ] = int(
                        local_index
                    )

                    selected_series_number[
                        pos
                    ] = float(
                        descriptor[
                            "meta"
                        ][
                            "series_number"
                        ]
                    )

                del data

            if any(
                image is None
                for image
                in sampled_images
            ):
                raise RuntimeError(
                    f"{row['study_uid']} "
                    "存在未提取的实例"
                )

            feature_chunks = []

            with torch.inference_mode():
                for start in range(
                    0,
                    len(sampled_images),
                    16,
                ):
                    batch_np = np.stack(
                        sampled_images[
                            start:start + 16
                        ]
                    )

                    values = (
                        torch
                        .from_numpy(batch_np)
                        .cuda(
                            non_blocking=True
                        )
                        [:, None]
                    )

                    channels = (
                        values
                        .repeat(
                            1,
                            3,
                            1,
                            1,
                        )
                    )

                    channels = (
                        F.interpolate(
                            channels,
                            size=(224, 224),
                            mode="bilinear",
                            align_corners=False,
                            antialias=True,
                        )
                    )

                    with torch.autocast(
                        "cuda"
                    ):
                        features = (
                            encoder(
                                (
                                    channels
                                    - mean
                                )
                                /
                                std
                            )
                        )

                    feature_chunks.append(
                        features
                        .float()
                        .cpu()
                        .numpy()
                    )

            features = np.concatenate(
                feature_chunks
            )

            assert (
                features.shape
                ==
                (
                    len(slice_indices),
                    768,
                )
            )

            assert np.isfinite(
                features
            ).all()

        temporary = (
            cache_path
            .with_suffix(
                ".tmp.npz"
            )
        )

        np.savez_compressed(
            temporary,

            features=
                features,

            slice_indices=
                slice_indices,

            original_count=
                original_count,

            selected_series=
                selected_series,

            selected_local_slice=
                selected_local,

            selected_series_number=
                selected_series_number,

            study_uid=
                np.asarray(
                    row["study_uid"]
                ),

            patient_uid=
                np.asarray(
                    row["patient_uid"]
                ),

            preparation_sha256=
                np.asarray(
                    digest
                ),

            normalization_json=
                np.asarray(
                    json.dumps(
                        normalization,
                        ensure_ascii=False,
                    )
                ),
        )

        temporary.replace(
            cache_path
        )

        save_json(
            OUT
            / (
                f"feature_worker_"
                f"{index}.json"
            ),
            {
                "status":
                    "running",

                "done":
                    done,

                "total":
                    len(jobs),

                "last_study":
                    row["study_uid"],

                "last_seconds":
                    (
                        time.monotonic()
                        - started
                    ),
            },
        )

    save_json(
        OUT
        / (
            f"feature_worker_"
            f"{index}.json"
        ),
        {
            "status":
                "complete",

            "done":
                len(jobs),

            "total":
                len(jobs),
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--tables-only",
        action="store_true",
    )

    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=[
            0,
            1,
            2,
            3,
        ],
    )

    parser.add_argument(
        "--worker-index",
        type=int,
    )

    parser.add_argument(
        "--limit",
        type=int,
    )

    args = parser.parse_args()

    if (
        args.worker_index
        is not None
    ):
        feature_worker(
            args.worker_index,
            args.devices,
            args.limit,
        )
        return

    prepare_tables()

    if args.tables_only:
        return

    handles = []
    workers = []

    for index, device in enumerate(
        args.devices
    ):
        env = os.environ.copy()

        env.update(
            CUDA_VISIBLE_DEVICES=
                str(device),

            OMP_NUM_THREADS=
                "2",
        )

        command = [
            sys.executable,
            "-u",
            str(
                Path(__file__)
                .resolve()
            ),
            "--worker-index",
            str(index),
            "--devices",
            *map(
                str,
                args.devices,
            ),
        ]

        if args.limit is not None:
            command += [
                "--limit",
                str(args.limit),
            ]

        handle = (
            OUT
            / (
                f"feature_worker_"
                f"{index}.log"
            )
        ).open("a")

        handles.append(
            handle
        )

        workers.append(
            subprocess.Popen(
                command,
                env=env,
                stdout=handle,
                stderr=
                    subprocess.STDOUT,
            )
        )

    state = {
        "status":
            "running",

        "expected_cases":
            1000,

        "worker_pids":
            [
                p.pid
                for p in workers
            ],
    }

    while any(
        p.poll() is None
        for p in workers
    ):
        count = len(
            list(
                (
                    OUT
                    / "features"
                ).glob(
                    "*.npz"
                )
            )
        )

        state.update(
            completed_cases=count,
            updated_unix=
                time.time(),
        )

        save_json(
            OUT
            / "feature_state.json",
            state,
        )

        print(
            "已生成MR-RATE特征缓存："
            f"{count}/1000",
            flush=True,
        )

        time.sleep(10)

    codes = [
        p.returncode
        for p in workers
    ]

    count = len(
        list(
            (
                OUT
                / "features"
            ).glob(
                "*.npz"
            )
        )
    )

    expected = (
        1000
        if args.limit is None
        else min(
            1000,
            args.limit
            * len(
                args.devices
            ),
        )
    )

    state.update(
        status=(
            "complete"
            if (
                count >= expected
                and not any(codes)
            )
            else "incomplete"
        ),

        completed_cases=
            count,

        worker_exit_codes=
            codes,

        finished_unix=
            time.time(),
    )

    save_json(
        OUT
        / "feature_state.json",
        state,
    )

    for handle in handles:
        handle.close()

    if any(codes):
        raise SystemExit(1)

    print(
        "MR-RATE特征准备结束："
        f"{count}/1000",
        flush=True,
    )


if __name__ == "__main__":
    main()
