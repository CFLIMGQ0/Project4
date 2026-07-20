from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .masking import (
    ANSWER_TERMS_BY_LABEL,
    CATEGORY_STANDARD_TERMS_BY_LABEL,
    MASK_EXTENSION_TERMS_BY_LABEL,
    contains_answer_term,
    mask_answer_terms,
    normalize_text,
)


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-z0-9]+")


@dataclass(frozen=True)
class TextRecord:
    patient_id: str
    exam_dir: str
    masked_text: str
    labels: np.ndarray
    mask_hits: tuple[str, ...]


def tokenize(text: Any) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_text(text))


def stable_hash_token(token: str, vocab_size: int) -> int:
    digest = hashlib.sha1(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (int(vocab_size) - 1) + 1


def _derive_patient_id(exam_dir: str) -> str:
    path = Path(exam_dir)
    return path.parent.name if path.parent.name else exam_dir


def load_masked_records(config: dict[str, Any]) -> list[TextRecord]:
    data_cfg = config["data"]
    csv_path = Path(config["paths"]["data_csv"]).expanduser()
    text_field = str(data_cfg["text_field"])
    forbidden_fields = {str(field) for field in data_cfg.get("forbidden_input_fields", [])}
    if text_field in forbidden_fields:
        raise ValueError(f"禁止把标签来源字段 {text_field} 作为文本输入")
    if text_field != "watch":
        raise ValueError("exp10 正式实验固定使用 watch，不能切换到诊断结论字段")

    label_names = list(data_cfg["label_names"])
    mask_token = str(data_cfg["mask_token"])
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"exam_dir", text_field, *label_names}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"数据文件缺少字段: {missing}")

        records: list[TextRecord] = []
        for row in tqdm(reader, desc="读取并遮蔽报告文本"):
            exam_dir = str(row.get("exam_dir", "")).strip()
            patient_id = str(row.get("patient_id", "")).strip() or _derive_patient_id(exam_dir)
            masked_text, hits = mask_answer_terms(row.get(text_field, ""), mask_token=mask_token)
            if contains_answer_term(masked_text):
                raise RuntimeError(f"遮蔽后仍残留答案词: {exam_dir}")
            labels = np.asarray(
                [int(float(row.get(label_name, 0) or 0)) for label_name in label_names],
                dtype=np.int64,
            )
            records.append(
                TextRecord(
                    patient_id=patient_id,
                    exam_dir=exam_dir,
                    masked_text=masked_text,
                    labels=labels,
                    mask_hits=tuple(hits),
                )
            )
    if not records:
        raise ValueError("没有读取到 exp10 文本样本")
    return records


def split_records_by_patient(
    records: list[TextRecord], seed: int, ratios: list[float]
) -> dict[str, list[TextRecord]]:
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("split_ratio 必须包含三个数且和为 1")
    patient_records: dict[str, list[TextRecord]] = {}
    for record in records:
        patient_records.setdefault(record.patient_id, []).append(record)
    patient_ids = list(patient_records)
    random.Random(seed).shuffle(patient_ids)
    train_end = int(len(patient_ids) * ratios[0])
    val_end = train_end + int(len(patient_ids) * ratios[1])
    split_ids = {
        "train": patient_ids[:train_end],
        "val": patient_ids[train_end:val_end],
        "test": patient_ids[val_end:],
    }
    return {
        split: [record for patient_id in ids for record in patient_records[patient_id]]
        for split, ids in split_ids.items()
    }


def build_train_vocabulary(
    train_records: list[TextRecord], max_vocab_size: int, min_frequency: int
) -> dict[str, int]:
    """词表只能从训练集遮蔽文本构建，避免验证集和测试集词汇泄漏。"""

    frequencies: Counter[str] = Counter()
    for record in tqdm(train_records, desc="构建训练集词表"):
        frequencies.update(tokenize(record.masked_text))
    ordered = sorted(
        ((token, count) for token, count in frequencies.items() if count >= min_frequency),
        key=lambda item: (-item[1], item[0]),
    )
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in ordered[: max(0, int(max_vocab_size) - len(vocabulary))]:
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_text(
    text: str,
    *,
    encoder_name: str,
    vocabulary: dict[str, int],
    hash_vocab_size: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    tokens = tokenize(text)[:max_length]
    if not tokens:
        tokens = [UNK_TOKEN]
    if encoder_name == "hashed_mean_encoder":
        token_ids = [stable_hash_token(f"watch:{token}", hash_vocab_size) for token in tokens]
    else:
        unknown_id = vocabulary[UNK_TOKEN]
        token_ids = [vocabulary.get(token, unknown_id) for token in tokens]
    ids = np.zeros(max_length, dtype=np.int64)
    mask = np.zeros(max_length, dtype=np.bool_)
    ids[: len(token_ids)] = token_ids
    mask[: len(token_ids)] = True
    return ids, mask


def labels_array(records: list[TextRecord]) -> np.ndarray:
    return np.stack([record.labels for record in records], axis=0)


def build_mask_audit(
    splits: dict[str, list[TextRecord]], label_names: list[str], mask_token: str
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "input_field": "watch",
        "forbidden_input_field": "watchResult",
        "mask_token": mask_token,
        "category_standard_terms_by_label": CATEGORY_STANDARD_TERMS_BY_LABEL,
        "mask_extension_terms_by_label": MASK_EXTENSION_TERMS_BY_LABEL,
        "answer_terms_by_label": ANSWER_TERMS_BY_LABEL,
        "splits": {},
    }
    patient_sets: dict[str, set[str]] = {}
    for split, records in splits.items():
        patient_sets[split] = {record.patient_id for record in records}
        term_counts = Counter(hit.lower() for record in records for hit in record.mask_hits)
        positives = labels_array(records).sum(axis=0)
        audit["splits"][split] = {
            "samples": len(records),
            "patients": len(patient_sets[split]),
            "empty_texts": sum(not record.masked_text for record in records),
            "mask_hit_rows": sum(bool(record.mask_hits) for record in records),
            "mask_term_counts": dict(term_counts.most_common()),
            "residual_answer_rows": sum(contains_answer_term(record.masked_text) for record in records),
            "label_positive_counts": dict(zip(label_names, positives.astype(int).tolist())),
        }
    audit["patient_overlap"] = {
        "train_val": len(patient_sets["train"] & patient_sets["val"]),
        "train_test": len(patient_sets["train"] & patient_sets["test"]),
        "val_test": len(patient_sets["val"] & patient_sets["test"]),
    }
    return audit


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


__all__ = [
    "PAD_TOKEN",
    "TextRecord",
    "build_mask_audit",
    "build_train_vocabulary",
    "encode_text",
    "labels_array",
    "load_masked_records",
    "save_json",
    "split_records_by_patient",
    "tokenize",
]
