from __future__ import annotations

import csv
import hashlib
import math
import os
import random
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from tasks import DEFAULT_GASTRO_TASK_NAME, get_task_spec
from tasks.common import derive_patient_id_from_exam_dir
from tasks.task2 import generate_pseudo_labels

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

GASTRO_LABEL_NAMES = list(get_task_spec(DEFAULT_GASTRO_TASK_NAME).label_names)
IMAGE_CACHE_MODES = {"none", "memory", "disk", "memory_and_disk"}
STRUCTURED_FIELD_NAMES = ("reportTitle", "age", "sex", "hp", "operationValue")
STRUCTURED_CATEGORICAL_FIELDS = ("reportTitle", "sex", "hp", "operationValue")
STRUCTURED_NUMERIC_FIELDS = ("age",)
STRUCTURED_FIELD_TO_INDEX = {name: index for index, name in enumerate(STRUCTURED_FIELD_NAMES)}
STRUCTURED_CATEGORICAL_TO_INDEX = {name: index for index, name in enumerate(STRUCTURED_CATEGORICAL_FIELDS)}
STRUCTURED_NUMERIC_TO_INDEX = {name: index for index, name in enumerate(STRUCTURED_NUMERIC_FIELDS)}
STRUCTURED_MISSING_TOKEN = "__MISSING__"
STRUCTURED_OTHER_TOKEN = "__OTHER_LOW_FREQ__"
STRUCTURED_REPORT_ALIASES = {
    "reportTitle": ("reportTitle", "report_title"),
    "age": ("age",),
    "sex": ("sex",),
    "hp": ("hp", "hp_status"),
    "operationValue": ("operationValue", "openationValue", "operation_value"),
}
TEXT_FIELD_NAMES = (
    "reportTitle",
    "age",
    "sex",
    "hp",
    "operationValue",
    "specimen",
    "score",
    "suggest",
    "watch",
)
TEXT_REPORT_ALIASES = {
    "reportTitle": ("reportTitle", "report_title"),
    "age": ("age",),
    "sex": ("sex",),
    "hp": ("hp", "hp_status"),
    "operationValue": ("operationValue", "openationValue", "operation_value"),
    "specimen": ("specimen",),
    "score": ("score",),
    "suggest": ("suggest", "suggestion"),
    "watch": ("watch", "watch_text"),
}
TEXT_TOKEN_VOCAB_SIZE = 8192
TEXT_TOKEN_MAX_LENGTH = 128
TEXT_SAFE_FIELDS = ("reportTitle", "hp", "operationValue", "specimen", "score")
TEXT_WATCH_FIELDS = ("watch",)
TEXT_GUIDED_FIELDS = ("reportTitle", "age", "sex", "hp", "operationValue", "specimen", "score")


def _iter_progress(iterable, total: int | None = None, desc: str = "处理中"):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def to_int(value: Any) -> int:
    if value is None:
        return 0
    cleaned = re.sub(r"[^0-9]", "", str(value))
    return int(cleaned) if cleaned else 0


def _normalize_structured_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\u3000", " ").strip())


def _structured_row_value(row: dict[str, Any], field_name: str) -> str:
    for alias in STRUCTURED_REPORT_ALIASES[field_name]:
        if alias in row:
            value = _normalize_structured_value(row.get(alias, ""))
            if value:
                return value
    return ""


def _report_row_value(row: dict[str, Any], field_name: str) -> str:
    for alias in TEXT_REPORT_ALIASES[field_name]:
        if alias in row:
            value = _normalize_structured_value(row.get(alias, ""))
            if value:
                return value
    return ""


def _text_raw_from_row(row: dict[str, Any]) -> dict[str, str]:
    return {field_name: _report_row_value(row, field_name) for field_name in TEXT_FIELD_NAMES}


def _hash_text_token(token: str, *, vocab_size: int = TEXT_TOKEN_VOCAB_SIZE) -> int:
    digest = hashlib.sha1(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (int(vocab_size) - 1) + 1


def _tokenize_text_value(value: Any) -> list[str]:
    text = _normalize_structured_value(value).lower()
    if not text:
        return []
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text)


def _encode_text_fields(
    text_raw: dict[str, Any],
    fields: tuple[str, ...],
    *,
    max_length: int = TEXT_TOKEN_MAX_LENGTH,
    vocab_size: int = TEXT_TOKEN_VOCAB_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids: list[int] = []
    for field_name in fields:
        value = text_raw.get(field_name, "")
        for token in _tokenize_text_value(value):
            token_ids.append(_hash_text_token(f"{field_name}:{token}", vocab_size=vocab_size))
            if len(token_ids) >= max_length:
                break
        if len(token_ids) >= max_length:
            break

    ids = torch.zeros(max_length, dtype=torch.long)
    mask = torch.zeros(max_length, dtype=torch.bool)
    if token_ids:
        used = token_ids[:max_length]
        ids[: len(used)] = torch.tensor(used, dtype=torch.long)
        mask[: len(used)] = True
    return ids, mask


def _parse_age_value(value: Any) -> float | None:
    cleaned = _normalize_structured_value(value)
    if not cleaned:
        return None
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def resolve_exam_dir(exam_dir: str | Path, dataset_root: str | Path | None = None) -> tuple[Path, bool]:
    exam_path = Path(exam_dir).expanduser()
    if exam_path.is_dir():
        return exam_path, False

    if dataset_root is None or not str(dataset_root).strip():
        return exam_path, False

    root = Path(dataset_root).expanduser()
    if not root.is_absolute():
        root = root.resolve()

    if not exam_path.is_absolute():
        candidate = (root / exam_path).resolve()
        if candidate.is_dir():
            return candidate, True

    if len(exam_path.parts) >= 2:
        candidate = root / exam_path.parts[-2] / exam_path.parts[-1]
        if candidate.is_dir():
            return candidate, True

    return exam_path, False


def collect_image_paths(exam_dir: str | Path) -> list[str]:
    base = Path(exam_dir)
    if not base.exists() or not base.is_dir():
        return []
    image_paths: list[str] = []
    for path in base.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(str(path))
    image_paths.sort()
    return image_paths


def load_task_rows(task_csv_path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(task_csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"未找到任务 CSV: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def build_task_records(
    task_csv_path: str | Path,
    task_name: str,
    min_instances: int = 1,
    dataset_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    if task_name == "gastro_multilabel":
        task_spec = get_task_spec(DEFAULT_GASTRO_TASK_NAME)
    else:
        task_spec = get_task_spec(task_name)

    rows = load_task_rows(task_csv_path)
    records: list[dict[str, Any]] = []

    for row in _iter_progress(rows, total=len(rows), desc=f"构建{task_name}样本"):
        exam_dir = str(row.get("exam_dir", "")).strip()
        patient_id = str(row.get("patient_id", "")).strip() or derive_patient_id_from_exam_dir(exam_dir)
        report_title = str(row.get("reportTitle", "")).strip()
        watch_result = str(row.get("watchResult", "")).strip()
        watch = str(row.get("watch", "")).strip()
        specimen = str(row.get("specimen", "")).strip()
        hp = str(row.get("hp", "")).strip()
        score = _report_row_value(row, "score")
        suggest = _report_row_value(row, "suggest")
        img_num = to_int(row.get("img_num", 0))
        structured_raw = {
            "reportTitle": report_title,
            "age": _structured_row_value(row, "age"),
            "sex": _structured_row_value(row, "sex"),
            "hp": hp or _structured_row_value(row, "hp"),
            "operationValue": _structured_row_value(row, "operationValue"),
        }
        text_raw = _text_raw_from_row(row)
        text_raw.update(
            {
                "reportTitle": report_title,
                "hp": hp or text_raw.get("hp", ""),
                "specimen": specimen,
                "score": score,
                "suggest": suggest,
                "watch": watch,
            }
        )

        resolved_exam_dir, _ = resolve_exam_dir(exam_dir, dataset_root=dataset_root)

        image_paths = collect_image_paths(resolved_exam_dir)
        if len(image_paths) < min_instances:
            continue

        if task_spec.is_multilabel:
            labels = [to_int(row.get(label_name, 0)) for label_name in task_spec.label_names]
            if sum(labels) <= 0:
                continue
            record = {
                "patient_id": patient_id,
                "exam_dir": str(resolved_exam_dir),
                "report_title": report_title,
                "watch_result": watch_result,
                "watch": watch,
                "specimen": specimen,
                "hp": hp,
                "score": score,
                "suggest": suggest,
                "age": structured_raw["age"],
                "sex": structured_raw["sex"],
                "operation_value": structured_raw["operationValue"],
                "structured_raw": dict(structured_raw),
                "text_raw": dict(text_raw),
                "img_num": img_num,
                "image_paths": image_paths,
                "labels": labels,
            }
            if task_spec.name == "task2":
                pseudo_payload = generate_pseudo_labels(
                    watch=watch,
                    specimen=specimen,
                    num_images=len(image_paths),
                )
                record["pseudo_region_labels"] = pseudo_payload["region_labels"]
                record["pseudo_relevance"] = pseudo_payload["relevance_scores"]
            records.append(record)
        elif task_spec.is_binary:
            label = to_int(row.get("binary_label", -1))
            if label not in {0, 1}:
                continue
            records.append(
                {
                    "patient_id": patient_id,
                    "exam_dir": str(resolved_exam_dir),
                    "report_title": report_title,
                    "watch_result": watch_result,
                    "watch": watch,
                    "specimen": specimen,
                    "hp": hp,
                    "score": score,
                    "suggest": suggest,
                    "age": structured_raw["age"],
                    "sex": structured_raw["sex"],
                    "operation_value": structured_raw["operationValue"],
                    "structured_raw": dict(structured_raw),
                    "text_raw": dict(text_raw),
                    "img_num": img_num,
                    "image_paths": image_paths,
                    "label": label,
                }
            )
        else:
            raise ValueError(f"未知 task_name: {task_name}")

    return records


def split_records(
    records: list[dict[str, Any]],
    seed: int,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    group_by_patient: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    if len(records) == 0:
        return {"train": [], "val": [], "test": []}

    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("ratios 必须和为 1")

    rng = random.Random(seed)

    if group_by_patient:
        patient_to_records: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            patient_id = str(record.get("patient_id", "")).strip() or derive_patient_id_from_exam_dir(
                str(record.get("exam_dir", ""))
            )
            patient_to_records.setdefault(patient_id, []).append(record)

        patient_ids = list(patient_to_records.keys())
        rng.shuffle(patient_ids)

        total_groups = len(patient_ids)
        num_train = int(total_groups * ratios[0])
        num_val = int(total_groups * ratios[1])
        num_test = total_groups - num_train - num_val

        train_ids = patient_ids[:num_train]
        val_ids = patient_ids[num_train : num_train + num_val]
        test_ids = patient_ids[num_train + num_val : num_train + num_val + num_test]

        return {
            "train": [record for patient_id in train_ids for record in patient_to_records[patient_id]],
            "val": [record for patient_id in val_ids for record in patient_to_records[patient_id]],
            "test": [record for patient_id in test_ids for record in patient_to_records[patient_id]],
        }

    indices = list(range(len(records)))
    rng.shuffle(indices)
    shuffled = [records[index] for index in indices]

    total = len(shuffled)
    num_train = int(total * ratios[0])
    num_val = int(total * ratios[1])
    num_test = total - num_train - num_val

    return {
        "train": shuffled[:num_train],
        "val": shuffled[num_train : num_train + num_val],
        "test": shuffled[num_train + num_val : num_train + num_val + num_test],
    }


def enrich_records_with_report_fields(
    records: list[dict[str, Any]],
    report_csv_path: str | Path | None,
) -> dict[str, Any]:
    """按 exam_dir 回连源报告表，补齐 datalist 未保留的结构化字段。"""

    if report_csv_path is None or not str(report_csv_path).strip():
        return {"enabled": False, "reason": "empty_report_csv_path", "matched": 0, "total": len(records)}

    path = Path(report_csv_path).expanduser()
    if not path.is_file():
        return {"enabled": False, "reason": f"report_csv_not_found: {path}", "matched": 0, "total": len(records)}

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        report_rows = list(reader)

    row_map: dict[str, dict[str, str]] = {}
    for row in report_rows:
        exam_dir = _normalize_structured_value(row.get("exam_dir", ""))
        if not exam_dir:
            continue
        row_map[exam_dir] = row
        try:
            resolved = str(Path(exam_dir).expanduser().resolve())
            row_map.setdefault(resolved, row)
        except OSError:
            pass

    matched = 0
    for record in records:
        raw_payload = dict(record.get("structured_raw", {}))
        for field_name in STRUCTURED_FIELD_NAMES:
            raw_payload.setdefault(field_name, "")
        text_payload = dict(record.get("text_raw", {}))
        for field_name in TEXT_FIELD_NAMES:
            text_payload.setdefault(field_name, "")

        row = row_map.get(str(record.get("exam_dir", "")).strip())
        if row is None:
            try:
                row = row_map.get(str(Path(str(record.get("exam_dir", ""))).expanduser().resolve()))
            except OSError:
                row = None

        if row is not None:
            matched += 1
            for field_name in STRUCTURED_FIELD_NAMES:
                current_value = _normalize_structured_value(raw_payload.get(field_name, ""))
                if current_value:
                    raw_payload[field_name] = current_value
                    continue
                raw_payload[field_name] = _structured_row_value(row, field_name)
            for field_name in TEXT_FIELD_NAMES:
                current_value = _normalize_structured_value(text_payload.get(field_name, ""))
                if current_value:
                    text_payload[field_name] = current_value
                    continue
                text_payload[field_name] = _report_row_value(row, field_name)

        record["structured_raw"] = raw_payload
        record["text_raw"] = text_payload
        record["age"] = raw_payload.get("age", "")
        record["sex"] = raw_payload.get("sex", "")
        record["operation_value"] = raw_payload.get("operationValue", "")
        record["score"] = text_payload.get("score", "")
        record["suggest"] = text_payload.get("suggest", "")
        if text_payload.get("specimen", "") and not record.get("specimen"):
            record["specimen"] = text_payload["specimen"]
        if text_payload.get("watch", "") and not record.get("watch"):
            record["watch"] = text_payload["watch"]
        if raw_payload.get("hp", "") and not record.get("hp"):
            record["hp"] = raw_payload["hp"]

    return {
        "enabled": True,
        "report_csv_path": str(path.resolve()),
        "source_rows": len(report_rows),
        "matched": matched,
        "total": len(records),
        "match_rate": float(matched / len(records)) if records else 0.0,
    }


def fit_structured_feature_metadata(
    train_records: list[dict[str, Any]],
    *,
    min_category_count: int = 20,
) -> dict[str, Any]:
    min_category_count = max(1, int(min_category_count))
    category_maps: dict[str, dict[str, int]] = {}
    category_counts: dict[str, dict[str, int]] = {}
    audit: list[dict[str, Any]] = []

    for field_name in STRUCTURED_CATEGORICAL_FIELDS:
        counts: dict[str, int] = {}
        non_missing = 0
        for record in train_records:
            value = _normalize_structured_value(record.get("structured_raw", {}).get(field_name, ""))
            if value:
                non_missing += 1
                counts[value] = counts.get(value, 0) + 1

        kept_values = sorted(value for value, count in counts.items() if count >= min_category_count)
        vocab = [STRUCTURED_MISSING_TOKEN, STRUCTURED_OTHER_TOKEN, *kept_values]
        category_maps[field_name] = {value: index for index, value in enumerate(vocab)}
        category_counts[field_name] = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
        audit.append(
            {
                "field": field_name,
                "type": "categorical",
                "train_total": len(train_records),
                "train_non_missing": non_missing,
                "train_missing": len(train_records) - non_missing,
                "train_missing_rate": float((len(train_records) - non_missing) / len(train_records)) if train_records else 0.0,
                "raw_unique_non_missing": len(counts),
                "encoded_unique": len(vocab),
                "min_category_count": min_category_count,
                "kept_categories": kept_values,
                "low_frequency_category_count": len(counts) - len(kept_values),
            }
        )

    age_values = [
        parsed
        for record in train_records
        for parsed in [_parse_age_value(record.get("structured_raw", {}).get("age", ""))]
        if parsed is not None
    ]
    age_mean = float(np.mean(age_values)) if age_values else 0.0
    age_std = float(np.std(age_values)) if age_values else 1.0
    if age_std < 1e-6:
        age_std = 1.0
    audit.append(
        {
            "field": "age",
            "type": "numeric",
            "train_total": len(train_records),
            "train_non_missing": len(age_values),
            "train_missing": len(train_records) - len(age_values),
            "train_missing_rate": float((len(train_records) - len(age_values)) / len(train_records)) if train_records else 0.0,
            "raw_unique_non_missing": len(set(age_values)),
            "encoded_unique": 1,
            "mean": age_mean,
            "std": age_std,
        }
    )

    return {
        "field_names": list(STRUCTURED_FIELD_NAMES),
        "categorical_fields": list(STRUCTURED_CATEGORICAL_FIELDS),
        "numeric_fields": list(STRUCTURED_NUMERIC_FIELDS),
        "field_to_index": dict(STRUCTURED_FIELD_TO_INDEX),
        "categorical_to_index": dict(STRUCTURED_CATEGORICAL_TO_INDEX),
        "numeric_to_index": dict(STRUCTURED_NUMERIC_TO_INDEX),
        "missing_token": STRUCTURED_MISSING_TOKEN,
        "other_token": STRUCTURED_OTHER_TOKEN,
        "min_category_count": min_category_count,
        "category_maps": category_maps,
        "category_counts": category_counts,
        "category_sizes": {field_name: len(vocab) for field_name, vocab in category_maps.items()},
        "numeric_stats": {"age": {"mean": age_mean, "std": age_std}},
        "audit": audit,
    }


def apply_structured_feature_metadata(
    split_data: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any],
) -> None:
    category_maps = metadata.get("category_maps", {})
    numeric_stats = metadata.get("numeric_stats", {})
    age_stats = numeric_stats.get("age", {}) if isinstance(numeric_stats, dict) else {}
    age_mean = float(age_stats.get("mean", 0.0))
    age_std = float(age_stats.get("std", 1.0)) or 1.0

    for records in split_data.values():
        for record in records:
            raw_payload = dict(record.get("structured_raw", {}))
            categorical_ids: list[int] = []
            field_mask = [0.0 for _ in STRUCTURED_FIELD_NAMES]

            for field_name in STRUCTURED_CATEGORICAL_FIELDS:
                value = _normalize_structured_value(raw_payload.get(field_name, ""))
                if value:
                    field_mask[STRUCTURED_FIELD_TO_INDEX[field_name]] = 1.0
                vocab = category_maps.get(field_name, {})
                if not value:
                    encoded = int(vocab.get(STRUCTURED_MISSING_TOKEN, 0))
                else:
                    encoded = int(vocab.get(value, vocab.get(STRUCTURED_OTHER_TOKEN, 1)))
                categorical_ids.append(encoded)

            age_value = _parse_age_value(raw_payload.get("age", ""))
            if age_value is None:
                age_encoded = 0.0
            else:
                field_mask[STRUCTURED_FIELD_TO_INDEX["age"]] = 1.0
                age_encoded = float((age_value - age_mean) / age_std)

            record["structured_categorical"] = categorical_ids
            record["structured_numeric"] = [age_encoded]
            record["structured_mask"] = field_mask


def prepare_structured_features(
    split_data: dict[str, list[dict[str, Any]]],
    *,
    fit_records: list[dict[str, Any]],
    min_category_count: int = 20,
) -> dict[str, Any]:
    metadata = fit_structured_feature_metadata(fit_records, min_category_count=min_category_count)
    apply_structured_feature_metadata(split_data, metadata)
    return metadata


def build_image_transform(image_size: int, is_train: bool) -> transforms.Compose:
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15), antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _normalize_image_key(image_path: str | Path) -> str:
    return str(Path(image_path).expanduser().resolve())


def load_roi_index(
    roi_index_path: str | Path | None,
    *,
    max_crops_per_source: int,
    min_score: float,
) -> dict[str, list[str]]:
    if roi_index_path is None or not str(roi_index_path).strip():
        return {}
    index_path = Path(roi_index_path).expanduser()
    if not index_path.is_file():
        raise FileNotFoundError(f"未找到 ROI 索引文件: {index_path}")

    raw_map: dict[str, list[tuple[float, int, str]]] = {}
    with index_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            source_path = str(row.get("source_image_path", "")).strip()
            crop_path = str(row.get("crop_path", "")).strip()
            if not source_path or not crop_path:
                continue
            try:
                score = float(row.get("score", 0.0))
            except Exception:
                score = 0.0
            if score < min_score:
                continue
            try:
                rank = int(float(row.get("rank", 9999)))
            except Exception:
                rank = 9999
            crop = Path(crop_path).expanduser()
            if not crop.is_file():
                continue
            try:
                if crop.stat().st_size <= 0:
                    continue
            except OSError:
                continue
            raw_map.setdefault(_normalize_image_key(source_path), []).append((score, rank, str(crop.resolve())))

    roi_map: dict[str, list[str]] = {}
    keep_num = max(1, int(max_crops_per_source))
    for source_path, items in raw_map.items():
        items.sort(key=lambda item: (-item[0], item[1], item[2]))
        roi_map[source_path] = [crop_path for _, _, crop_path in items[:keep_num]]
    return roi_map


class _LRUImageArrayCache:
    def __init__(self, max_items: int) -> None:
        self.max_items = max(0, int(max_items))
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: str) -> np.ndarray | None:
        if self.max_items <= 0:
            return None
        value = self._data.get(key)
        if value is None:
            return None
        self._data.move_to_end(key)
        return value

    def put(self, key: str, value: np.ndarray) -> None:
        if self.max_items <= 0:
            return
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)


class MILBagDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        task_name: str,
        max_instances: int,
        min_instances: int,
        bag_sampling_strategy: str,
        is_train: bool,
        image_size: int,
        random_instance_dropout: float = 0.0,
        image_cache_mode: str = "none",
        image_cache_dir: str | Path | None = None,
        legacy_image_cache_dirs: list[str | Path] | None = None,
        memory_cache_size: int = 0,
        roi_index_path: str | Path | None = None,
        roi_enabled: bool = False,
        roi_max_crops_per_bag: int = 0,
        roi_max_crops_per_source: int = 1,
        roi_min_score: float = 0.0,
        split_name: str = "",
        structured_shuffle_fields: list[str] | tuple[str, ...] | None = None,
        structured_shuffle_apply_to: str = "none",
        structured_shuffle_seed: int = 0,
    ) -> None:
        super().__init__()
        self.records = records
        self.task_name = task_name
        self.max_instances = max_instances
        self.min_instances = min_instances
        self.strategy = bag_sampling_strategy
        self.is_train = is_train
        self.random_instance_dropout = random_instance_dropout if is_train else 0.0
        self.transform = build_image_transform(image_size=image_size, is_train=is_train)
        self.cache_image_size = int(image_size * 1.5)
        self.roi_enabled = bool(roi_enabled)
        self.roi_max_crops_per_bag = max(0, int(roi_max_crops_per_bag))
        self.roi_max_crops_per_source = max(1, int(roi_max_crops_per_source))
        self.roi_min_score = float(roi_min_score)
        self.split_name = str(split_name).strip().lower()
        self.structured_shuffle_fields = [
            field_name
            for field_name in (structured_shuffle_fields or [])
            if field_name in STRUCTURED_FIELD_TO_INDEX
        ]
        self.structured_shuffle_apply_to = str(structured_shuffle_apply_to).strip().lower() or "none"
        self.structured_shuffle_seed = int(structured_shuffle_seed)
        self._structured_shuffle_values = self._build_structured_shuffle_values()
        self.roi_map = (
            load_roi_index(
                roi_index_path,
                max_crops_per_source=self.roi_max_crops_per_source,
                min_score=self.roi_min_score,
            )
            if self.roi_enabled and self.roi_max_crops_per_bag > 0
            else {}
        )
        self.image_cache_mode = str(image_cache_mode).strip().lower() or "none"
        if self.image_cache_mode not in IMAGE_CACHE_MODES:
            raise ValueError(f"未知 image_cache_mode: {image_cache_mode}")
        self.use_memory_cache = self.image_cache_mode in {"memory", "memory_and_disk"}
        self.use_disk_cache = self.image_cache_mode in {"disk", "memory_and_disk"}
        self.image_cache_dir = None
        if self.use_disk_cache:
            if image_cache_dir is None or not str(image_cache_dir).strip():
                raise ValueError("启用磁盘图像缓存时必须提供 image_cache_dir")
            self.image_cache_dir = Path(image_cache_dir).expanduser().resolve()
            self.image_cache_dir.mkdir(parents=True, exist_ok=True)
        self.legacy_image_cache_dirs: list[Path] = []
        if self.use_disk_cache and legacy_image_cache_dirs:
            seen_legacy: set[Path] = set()
            for raw_dir in legacy_image_cache_dirs:
                if raw_dir is None or not str(raw_dir).strip():
                    continue
                resolved_dir = Path(raw_dir).expanduser().resolve()
                if self.image_cache_dir is not None and resolved_dir == self.image_cache_dir:
                    continue
                if resolved_dir in seen_legacy:
                    continue
                seen_legacy.add(resolved_dir)
                self.legacy_image_cache_dirs.append(resolved_dir)
        self.memory_cache = _LRUImageArrayCache(memory_cache_size if self.use_memory_cache else 0)

    def _should_shuffle_structured_fields(self) -> bool:
        if not self.structured_shuffle_fields:
            return False
        apply_to = self.structured_shuffle_apply_to
        if apply_to in {"", "none", "false", "off"}:
            return False
        if apply_to == "all":
            return True
        if apply_to in {"test", "test_only"}:
            return self.split_name == "test"
        if apply_to in {"eval", "val_test", "validation_test"}:
            return self.split_name in {"val", "test"}
        if apply_to in {"train", "train_only"}:
            return self.split_name == "train"
        return self.split_name == apply_to

    def _build_structured_shuffle_values(self) -> dict[str, list[tuple[float, float]]] | dict[str, list[tuple[int, float]]]:
        if not self._should_shuffle_structured_fields() or not self.records:
            return {}

        rng = random.Random(self.structured_shuffle_seed + sum(ord(char) for char in self.split_name))
        shuffled_values: dict[str, list[Any]] = {}
        for field_name in self.structured_shuffle_fields:
            values: list[Any] = []
            if field_name in STRUCTURED_CATEGORICAL_TO_INDEX:
                cat_index = STRUCTURED_CATEGORICAL_TO_INDEX[field_name]
                mask_index = STRUCTURED_FIELD_TO_INDEX[field_name]
                for record in self.records:
                    categorical = list(record.get("structured_categorical", []))
                    mask = list(record.get("structured_mask", []))
                    value = int(categorical[cat_index]) if cat_index < len(categorical) else 0
                    mask_value = float(mask[mask_index]) if mask_index < len(mask) else 0.0
                    values.append((value, mask_value))
            elif field_name in STRUCTURED_NUMERIC_TO_INDEX:
                numeric_index = STRUCTURED_NUMERIC_TO_INDEX[field_name]
                mask_index = STRUCTURED_FIELD_TO_INDEX[field_name]
                for record in self.records:
                    numeric = list(record.get("structured_numeric", []))
                    mask = list(record.get("structured_mask", []))
                    value = float(numeric[numeric_index]) if numeric_index < len(numeric) else 0.0
                    mask_value = float(mask[mask_index]) if mask_index < len(mask) else 0.0
                    values.append((value, mask_value))
            rng.shuffle(values)
            shuffled_values[field_name] = values
        return shuffled_values

    def _structured_tensors_for_record(self, record: dict[str, Any], index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if "structured_categorical" not in record or "structured_numeric" not in record or "structured_mask" not in record:
            return None
        categorical = [int(value) for value in record.get("structured_categorical", [])]
        numeric = [float(value) for value in record.get("structured_numeric", [])]
        field_mask = [float(value) for value in record.get("structured_mask", [])]

        if self._structured_shuffle_values:
            for field_name, values in self._structured_shuffle_values.items():
                if not values:
                    continue
                shuffled_value, shuffled_mask = values[index % len(values)]
                if field_name in STRUCTURED_CATEGORICAL_TO_INDEX:
                    cat_index = STRUCTURED_CATEGORICAL_TO_INDEX[field_name]
                    mask_index = STRUCTURED_FIELD_TO_INDEX[field_name]
                    if cat_index < len(categorical):
                        categorical[cat_index] = int(shuffled_value)
                    if mask_index < len(field_mask):
                        field_mask[mask_index] = float(shuffled_mask)
                elif field_name in STRUCTURED_NUMERIC_TO_INDEX:
                    numeric_index = STRUCTURED_NUMERIC_TO_INDEX[field_name]
                    mask_index = STRUCTURED_FIELD_TO_INDEX[field_name]
                    if numeric_index < len(numeric):
                        numeric[numeric_index] = float(shuffled_value)
                    if mask_index < len(field_mask):
                        field_mask[mask_index] = float(shuffled_mask)

        return (
            torch.tensor(categorical, dtype=torch.long),
            torch.tensor(numeric, dtype=torch.float32),
            torch.tensor(field_mask, dtype=torch.float32),
        )

    def _text_tensors_for_record(self, record: dict[str, Any]) -> dict[str, torch.Tensor] | None:
        text_raw = record.get("text_raw")
        if not isinstance(text_raw, dict):
            return None
        text_ids, text_mask = _encode_text_fields(text_raw, TEXT_SAFE_FIELDS)
        watch_ids, watch_mask = _encode_text_fields(text_raw, TEXT_WATCH_FIELDS)
        guided_ids, guided_mask = _encode_text_fields(text_raw, TEXT_GUIDED_FIELDS)
        return {
            "text_token_ids": text_ids,
            "text_token_mask": text_mask,
            "watch_token_ids": watch_ids,
            "watch_token_mask": watch_mask,
            "guided_text_token_ids": guided_ids,
            "guided_text_token_mask": guided_mask,
        }

    def __len__(self) -> int:
        return len(self.records)

    def _memory_cache_key(self, image_path: str | Path) -> str:
        return str(Path(image_path).expanduser().resolve())

    def _disk_cache_path_for_dir(self, image_path: str | Path, cache_dir: Path | None) -> Path | None:
        if cache_dir is None:
            return None

        source_path = Path(image_path).expanduser()
        try:
            resolved_path = source_path.resolve(strict=True)
            stat_result = resolved_path.stat()
            cache_signature = f"{resolved_path}|{stat_result.st_size}|{stat_result.st_mtime_ns}|{self.cache_image_size}"
        except FileNotFoundError:
            resolved_path = source_path.resolve()
            cache_signature = f"{resolved_path}|{self.cache_image_size}"

        digest = hashlib.sha1(cache_signature.encode("utf-8")).hexdigest()
        return cache_dir / digest[:2] / f"{digest}.npy"

    def _disk_cache_path(self, image_path: str | Path) -> Path | None:
        return self._disk_cache_path_for_dir(image_path, self.image_cache_dir)

    def _legacy_disk_cache_paths(self, image_path: str | Path) -> list[Path]:
        return [
            candidate
            for candidate in (
                self._disk_cache_path_for_dir(image_path, cache_dir)
                for cache_dir in self.legacy_image_cache_dirs
            )
            if candidate is not None
        ]

    def _load_disk_cache_file(self, cache_path: Path) -> np.ndarray:
        with cache_path.open("rb") as file:
            return np.load(file, allow_pickle=False)

    def _promote_disk_cache(self, source_path: Path, target_path: Path | None) -> None:
        if target_path is None or source_path == target_path or target_path.is_file():
            return
        if os.environ.get("PROJECT4_DISABLE_DISK_CACHE_WRITE") == "1":
            return
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_name(f"{target_path.name}.{os.getpid()}.tmp")
        try:
            try:
                os.link(source_path, temp_path)
            except OSError:
                shutil.copy2(source_path, temp_path)
            os.replace(temp_path, target_path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _locate_existing_disk_cache(self, image_path: str | Path) -> tuple[Path | None, Path | None]:
        primary_cache_path = self._disk_cache_path(image_path)
        if primary_cache_path is not None and primary_cache_path.is_file():
            return primary_cache_path, primary_cache_path
        for legacy_cache_path in self._legacy_disk_cache_paths(image_path):
            if legacy_cache_path.is_file():
                return primary_cache_path, legacy_cache_path
        return primary_cache_path, None

    def _load_source_image_array(self, image_path: str | Path) -> np.ndarray:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            target = self.cache_image_size
            if target > 0:
                w, h = rgb_image.size
                short_edge = min(w, h)
                if short_edge > target:
                    scale = target / short_edge
                    rgb_image = rgb_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                w, h = rgb_image.size
                if w > target or h > target:
                    left = (w - target) // 2
                    top = (h - target) // 2
                    rgb_image = rgb_image.crop((left, top, left + target, top + target))
            return np.asarray(rgb_image, dtype=np.uint8)

    def _save_disk_cache(self, image_path: str | Path, image_array: np.ndarray) -> None:
        if os.environ.get("PROJECT4_DISABLE_DISK_CACHE_WRITE") == "1":
            return
        cache_path = self._disk_cache_path(image_path)
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.is_file():
            return

        temp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        try:
            with temp_path.open("wb") as file:
                np.save(file, image_array, allow_pickle=False)
            os.replace(temp_path, cache_path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _load_image_array(self, image_path: str | Path) -> np.ndarray:
        memory_key = self._memory_cache_key(image_path)

        if self.use_memory_cache:
            cached_array = self.memory_cache.get(memory_key)
            if cached_array is not None:
                return cached_array

        primary_cache_path, existing_cache_path = self._locate_existing_disk_cache(image_path)
        if existing_cache_path is not None:
            try:
                cached_array = self._load_disk_cache_file(existing_cache_path)
                if primary_cache_path is not None and existing_cache_path != primary_cache_path:
                    self._promote_disk_cache(existing_cache_path, primary_cache_path)
                if self.use_memory_cache:
                    self.memory_cache.put(memory_key, cached_array)
                return cached_array
            except Exception:
                existing_cache_path.unlink(missing_ok=True)

        image_array = self._load_source_image_array(image_path)
        if primary_cache_path is not None:
            self._save_disk_cache(image_path, image_array)
        if self.use_memory_cache:
            self.memory_cache.put(memory_key, image_array)
        return image_array

    def prepare_image_cache(self, desc: str = "构建图像缓存") -> dict[str, int]:
        if not self.use_disk_cache:
            return {"total": 0, "created": 0, "reused": 0, "failed": 0}

        unique_paths = list(
            dict.fromkeys(
                image_path
                for record in self.records
                for image_path in record.get("image_paths", [])
            )
        )

        created = 0
        success = 0
        failed = 0
        progress = None
        if tqdm is not None:
            progress = tqdm(unique_paths, total=len(unique_paths), desc=desc, dynamic_ncols=True)
            iterator = progress
        else:
            iterator = unique_paths

        for index, image_path in enumerate(iterator, start=1):
            primary_cache_path, existing_cache_path = self._locate_existing_disk_cache(image_path)
            if existing_cache_path is not None:
                if primary_cache_path is not None and existing_cache_path != primary_cache_path:
                    self._promote_disk_cache(existing_cache_path, primary_cache_path)
                success += 1
                if progress is not None and (index % 64 == 0 or index == len(unique_paths)):
                    progress.set_postfix({"成功": success, "失败": failed}, refresh=False)
                continue
            try:
                image_array = self._load_source_image_array(image_path)
                self._save_disk_cache(image_path, image_array)
                created += 1
                success += 1
            except Exception as exc:
                failed += 1
                del exc

            if progress is not None and (index % 64 == 0 or index == len(unique_paths)):
                progress.set_postfix({"成功": success, "失败": failed}, refresh=False)

        if progress is not None:
            progress.set_postfix({"成功": success, "失败": failed}, refresh=False)
            progress.close()

        return {"total": len(unique_paths), "created": created, "reused": success - created, "failed": failed}

    def _uniform_indices(self, num_instances: int, keep_num: int) -> list[int]:
        if keep_num >= num_instances:
            return list(range(num_instances))

        step = (num_instances - 1) / float(keep_num - 1) if keep_num > 1 else 0.0
        indices = [int(round(idx * step)) for idx in range(keep_num)]
        indices = sorted(set(indices))
        current = 0
        while len(indices) < keep_num:
            if current not in indices:
                indices.append(current)
            current += 1
        return sorted(indices[:keep_num])

    def _select_indices(
        self,
        num_instances: int,
        rng: random.Random,
        max_instances_override: int | None = None,
    ) -> list[int]:
        active_max_instances = self.max_instances if max_instances_override is None else int(max_instances_override)
        max_instances = active_max_instances if active_max_instances > 0 else num_instances
        keep_num = min(num_instances, max_instances)

        if self.strategy == "random":
            if keep_num >= num_instances:
                indices = list(range(num_instances))
            else:
                indices = sorted(rng.sample(range(num_instances), keep_num))
        elif self.strategy == "uniform":
            indices = self._uniform_indices(num_instances=num_instances, keep_num=keep_num)
        elif self.strategy == "all_if_small":
            if num_instances <= max_instances:
                indices = list(range(num_instances))
            else:
                indices = self._uniform_indices(num_instances=num_instances, keep_num=keep_num)
        else:
            raise ValueError(f"未知 bag_sampling_strategy: {self.strategy}")

        if self.random_instance_dropout > 0 and len(indices) > self.min_instances:
            kept_indices: list[int] = []
            for index in indices:
                if rng.random() > self.random_instance_dropout:
                    kept_indices.append(index)
            if len(kept_indices) < self.min_instances:
                need_num = self.min_instances - len(kept_indices)
                remain = [index for index in indices if index not in kept_indices]
                kept_indices.extend(remain[:need_num])
            indices = sorted(kept_indices)

        if len(indices) < self.min_instances:
            if not indices:
                indices = [0]
            while len(indices) < self.min_instances:
                indices.append(indices[len(indices) % len(indices)])
            indices = sorted(indices)

        return indices

    def _collect_roi_crop_paths(
        self,
        selected_source_paths: list[str],
        rng: random.Random,
    ) -> list[str]:
        if not self.roi_map or self.roi_max_crops_per_bag <= 0:
            return []

        candidates: list[str] = []
        for source_path in selected_source_paths:
            candidates.extend(self.roi_map.get(_normalize_image_key(source_path), []))
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            return []
        if self.is_train:
            rng.shuffle(candidates)
        return candidates[: self.roi_max_crops_per_bag]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_paths_all: list[str] = record["image_paths"]
        rng = random.Random((index + 1) * 104729 if self.is_train else index + 17)

        original_max_instances = self.max_instances
        if self.roi_enabled and self.roi_max_crops_per_bag > 0 and self.max_instances > 0:
            original_max_instances = max(self.min_instances, self.max_instances - self.roi_max_crops_per_bag)
        selected_indices = self._select_indices(
            num_instances=len(image_paths_all),
            rng=rng,
            max_instances_override=original_max_instances,
        )
        selected_paths = [image_paths_all[item] for item in selected_indices]
        roi_crop_paths = self._collect_roi_crop_paths(selected_paths, rng=rng)
        selected_paths = [*selected_paths, *roi_crop_paths]

        roi_crop_path_set = set(roi_crop_paths)
        loaded_paths: list[str] = []
        loaded_roi_crop_paths: list[str] = []
        images = []
        for path in selected_paths:
            try:
                image = Image.fromarray(self._load_image_array(path), mode="RGB")
            except (OSError, UnidentifiedImageError):
                if path in roi_crop_path_set:
                    continue
                raise
            images.append(self.transform(image))
            loaded_paths.append(path)
            if path in roi_crop_path_set:
                loaded_roi_crop_paths.append(path)
        selected_paths = loaded_paths
        roi_crop_paths = loaded_roi_crop_paths
        bag_images = torch.stack(images, dim=0)

        if "labels" in record:
            label = torch.tensor(record["labels"], dtype=torch.float32)
        elif "label" in record:
            label = torch.tensor(record["label"], dtype=torch.long)
        else:
            raise ValueError(f"未知 task_name: {self.task_name}")

        item = {
            "images": bag_images,
            "label": label,
            "exam_dir": record["exam_dir"],
            "image_paths": selected_paths,
            "report_title": record.get("report_title", ""),
            "img_num": int(record.get("img_num", len(image_paths_all))),
            "meta": {},
        }
        structured_tensors = self._structured_tensors_for_record(record, index)
        if structured_tensors is not None:
            structured_categorical, structured_numeric, structured_mask = structured_tensors
            item["structured_categorical"] = structured_categorical
            item["structured_numeric"] = structured_numeric
            item["structured_mask"] = structured_mask
        text_tensors = self._text_tensors_for_record(record)
        if text_tensors is not None:
            item.update(text_tensors)
        if self.roi_enabled and self.roi_max_crops_per_bag > 0:
            instance_types = [0 for _ in selected_indices]
            instance_types.extend([1 for _ in roi_crop_paths])
            item["instance_types"] = torch.tensor(instance_types, dtype=torch.long)
        if "pseudo_region_labels" in record:
            pseudo_region_labels_all = list(record["pseudo_region_labels"])
            selected_values = [int(pseudo_region_labels_all[item_index]) for item_index in selected_indices]
            selected_values.extend([-100 for _ in roi_crop_paths])
            item["pseudo_region_labels"] = torch.tensor(
                selected_values,
                dtype=torch.long,
            )
        if "pseudo_relevance" in record:
            pseudo_relevance_all = list(record["pseudo_relevance"])
            selected_values = [float(pseudo_relevance_all[item_index]) for item_index in selected_indices]
            selected_values.extend([-1.0 for _ in roi_crop_paths])
            item["pseudo_relevance"] = torch.tensor(
                selected_values,
                dtype=torch.float32,
            )
        item["meta"]["roi_num_crops"] = len(roi_crop_paths)
        return item


def mil_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_num_instances = max(item["images"].shape[0] for item in batch)
    channels, height, width = batch[0]["images"].shape[1:]

    images = torch.zeros((batch_size, max_num_instances, channels, height, width), dtype=batch[0]["images"].dtype)
    mask = torch.zeros((batch_size, max_num_instances), dtype=torch.bool)

    labels: list[torch.Tensor] = []
    exam_dirs: list[str] = []
    image_paths: list[list[str]] = []
    report_titles: list[str] = []
    img_nums: list[int] = []
    metas: list[dict[str, Any]] = []
    has_instance_types = any("instance_types" in item for item in batch)
    has_pseudo_region_labels = any("pseudo_region_labels" in item for item in batch)
    has_pseudo_relevance = any("pseudo_relevance" in item for item in batch)
    has_structured = any("structured_categorical" in item for item in batch)
    text_channels = (
        ("text_token_ids", "text_token_mask"),
        ("watch_token_ids", "watch_token_mask"),
        ("guided_text_token_ids", "guided_text_token_mask"),
    )
    text_batches: dict[str, torch.Tensor] = {}
    for ids_key, mask_key in text_channels:
        if any(ids_key in item for item in batch):
            first_text = next(item for item in batch if ids_key in item)
            text_len = int(first_text[ids_key].shape[0])
            text_batches[ids_key] = torch.zeros((batch_size, text_len), dtype=torch.long)
            text_batches[mask_key] = torch.zeros((batch_size, text_len), dtype=torch.bool)
    instance_types = torch.zeros((batch_size, max_num_instances), dtype=torch.long) if has_instance_types else None
    pseudo_region_labels = torch.full((batch_size, max_num_instances), -100, dtype=torch.long) if has_pseudo_region_labels else None
    pseudo_relevance = torch.full((batch_size, max_num_instances), -1.0, dtype=torch.float32) if has_pseudo_relevance else None
    structured_categorical = None
    structured_numeric = None
    structured_mask = None
    if has_structured:
        first_structured = next(item for item in batch if "structured_categorical" in item)
        cat_dim = int(first_structured["structured_categorical"].shape[0])
        numeric_dim = int(first_structured["structured_numeric"].shape[0])
        mask_dim = int(first_structured["structured_mask"].shape[0])
        structured_categorical = torch.zeros((batch_size, cat_dim), dtype=torch.long)
        structured_numeric = torch.zeros((batch_size, numeric_dim), dtype=torch.float32)
        structured_mask = torch.zeros((batch_size, mask_dim), dtype=torch.float32)

    for batch_index, item in enumerate(batch):
        num_instances = item["images"].shape[0]
        images[batch_index, :num_instances] = item["images"]
        mask[batch_index, :num_instances] = True
        labels.append(item["label"])
        exam_dirs.append(item["exam_dir"])
        image_paths.append(item["image_paths"])
        report_titles.append(item["report_title"])
        img_nums.append(item["img_num"])
        metas.append(item["meta"])
        if instance_types is not None and "instance_types" in item:
            instance_types[batch_index, :num_instances] = item["instance_types"]
        if pseudo_region_labels is not None and "pseudo_region_labels" in item:
            pseudo_region_labels[batch_index, :num_instances] = item["pseudo_region_labels"]
        if pseudo_relevance is not None and "pseudo_relevance" in item:
            pseudo_relevance[batch_index, :num_instances] = item["pseudo_relevance"]
        if structured_categorical is not None and "structured_categorical" in item:
            structured_categorical[batch_index] = item["structured_categorical"]
        if structured_numeric is not None and "structured_numeric" in item:
            structured_numeric[batch_index] = item["structured_numeric"]
        if structured_mask is not None and "structured_mask" in item:
            structured_mask[batch_index] = item["structured_mask"]
        for ids_key, mask_key in text_channels:
            if ids_key in text_batches and ids_key in item:
                text_batches[ids_key][batch_index] = item[ids_key]
                text_batches[mask_key][batch_index] = item[mask_key]

    if labels[0].ndim == 0:
        labels_tensor = torch.stack(labels, dim=0).long()
    else:
        labels_tensor = torch.stack(labels, dim=0).float()

    collated = {
        "images": images,
        "mask": mask,
        "labels": labels_tensor,
        "exam_dirs": exam_dirs,
        "image_paths": image_paths,
        "report_titles": report_titles,
        "img_nums": img_nums,
        "metas": metas,
    }
    if pseudo_region_labels is not None:
        collated["pseudo_region_labels"] = pseudo_region_labels
    if pseudo_relevance is not None:
        collated["pseudo_relevance"] = pseudo_relevance
    if instance_types is not None:
        collated["instance_types"] = instance_types
    if structured_categorical is not None and structured_numeric is not None and structured_mask is not None:
        collated["structured_categorical"] = structured_categorical
        collated["structured_numeric"] = structured_numeric
        collated["structured_mask"] = structured_mask
    collated.update(text_batches)
    return collated


class InstanceAwareBatchSampler(Sampler[list[int]]):
    """按实例总量限制批次，降低不同检查目录图像数差异带来的显存波动。"""

    def __init__(
        self,
        records: list[dict[str, Any]],
        max_instances_per_bag: int,
        min_instances_per_bag: int,
        batch_size: int,
        max_instances_per_batch: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.records = records
        self.max_instances_per_bag = int(max_instances_per_bag)
        self.min_instances_per_bag = max(1, int(min_instances_per_bag))
        self.batch_size = max(1, int(batch_size))
        self.max_instances_per_batch = max(1, int(max_instances_per_batch))
        self.shuffle = shuffle
        self.seed = int(seed)
        self.iter_count = 0

        self.instance_counts: list[int] = []
        for record in self.records:
            num_instances = len(record.get("image_paths", []))
            num_instances = max(1, num_instances)
            if self.max_instances_per_bag > 0:
                num_instances = min(num_instances, self.max_instances_per_bag)
            num_instances = max(num_instances, self.min_instances_per_bag)
            self.instance_counts.append(num_instances)

    def state_dict(self) -> dict[str, int]:
        return {
            "seed": self.seed,
            "iter_count": self.iter_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self.iter_count = max(0, int(state.get("iter_count", self.iter_count)))

    def __iter__(self):
        indices = list(range(len(self.records)))
        if self.shuffle:
            rng = random.Random(self.seed + self.iter_count)
            rng.shuffle(indices)
        self.iter_count += 1

        batch: list[int] = []
        batch_instances = 0

        for index in indices:
            num_instances = self.instance_counts[index]

            need_flush = False
            if len(batch) >= self.batch_size:
                need_flush = True
            elif batch and (batch_instances + num_instances > self.max_instances_per_batch):
                need_flush = True

            if need_flush:
                yield batch
                batch = []
                batch_instances = 0

            batch.append(index)
            batch_instances += num_instances

        if batch:
            yield batch

    def __len__(self) -> int:
        if not self.records:
            return 0
        return int(math.ceil(len(self.records) / float(self.batch_size)))
