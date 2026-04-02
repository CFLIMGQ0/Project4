from __future__ import annotations

import csv
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

GASTRO_LABEL_NAMES = [
    "label_esophageal_smt",
    "label_esophageal_mucosal_or_tumor",
    "label_gastritis",
]

COLO_BINARY_CLASS_NAMES = ["normal", "polyp"]
COLO_TRICLASS_CLASS_NAMES = ["normal", "single_polyp", "multi_polyp"]

GASTROSCOPY_HINTS = [
    "胃镜",
    "食管",
    "胃窦",
    "胃体",
    "胃角",
    "贲门",
    "幽门",
    "十二指肠",
    "无痛胃镜",
    "超声胃镜",
]
COLONOSCOPY_HINTS = [
    "肠镜",
    "结肠",
    "直肠",
    "回盲",
    "乙状结肠",
    "升结肠",
    "降结肠",
    "横结肠",
    "盲肠",
]

GASTRIC_LABEL_RULES: dict[str, list[str]] = {
    "label_esophageal_smt": [
        "食管smt",
        "食管黏膜下隆起",
        "食管隆起性病变",
        "食管smt(来源于黏膜肌层)",
        "食管smt(来源于固有肌层)",
        "食管smt(来源于黏膜下层)",
        "食管黏膜下肿物",
    ],
    "label_esophageal_mucosal_or_tumor": [
        "食管黏膜病变",
        "食管肿物",
        "食管黏膜病变(待病理)",
        "食管黏膜病变(性质待定)",
        "食管肿物(待病理)",
        "食管占位",
        "食管新生物",
    ],
    "label_gastritis": [
        "慢性胃炎",
        "慢性非活动性胃炎",
        "慢性活动性胃炎",
        "萎缩性胃炎",
        "糜烂性胃炎",
        "浅表性胃炎",
        "胆汁反流性胃炎",
        "胃炎",
        "c1",
        "c2",
        "c3",
        "o1",
        "o2",
        "o3",
    ],
}

NORMAL_PATTERNS = [
    "无异常发现",
    "未见异常",
    "未见明显异常",
    "检查无异常发现",
    "结肠镜检查未见明显异常",
    "结肠镜检查无异常发现",
]
POLYP_PATTERNS = ["息肉", "结肠息肉", "直肠息肉", "结直肠息肉"]
MULTI_POLYP_PATTERNS = ["多发息肉", "结肠多发息肉", "结直肠多发息肉", "直肠多发息肉", "多发性结直肠息肉"]
LESION_TOKENS = ["息肉", "肿物", "炎", "憩室", "溃疡", "糜烂", "癌", "出血", "狭窄", "病变"]

GASTRO_SUBTYPE_NAMES = ["white_light", "surgery", "stain", "ultrasound"]


def _iter_progress(iterable, total: int | None = None, desc: str = "处理中"):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\u3000", " ")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.replace("（", "(").replace("）", ")")
    normalized = normalized.replace("【", "[").replace("】", "]")
    normalized = normalized.replace("，", ",").replace("；", ";").replace("：", ":")
    normalized = normalized.lower()
    return normalized


def contains_any(text: str, patterns: list[str]) -> bool:
    return any(p in text for p in patterns)


def match_rule_map(text: str, rule_map: dict[str, list[str]]) -> dict[str, int]:
    return {k: int(contains_any(text, v)) for k, v in rule_map.items()}


def to_int(value: Any) -> int:
    if value is None:
        return 0
    cleaned = re.sub(r"[^0-9]", "", str(value))
    return int(cleaned) if cleaned else 0


def is_gastroscopy_record(report_title_norm: str, watch_result_norm: str) -> bool:
    has_gastric = contains_any(report_title_norm, GASTROSCOPY_HINTS) or contains_any(watch_result_norm, GASTROSCOPY_HINTS)
    has_colon = contains_any(report_title_norm, COLONOSCOPY_HINTS) or contains_any(watch_result_norm, COLONOSCOPY_HINTS)
    return has_gastric and not has_colon


def is_colonoscopy_record(report_title_norm: str, watch_result_norm: str) -> bool:
    has_colon = contains_any(report_title_norm, COLONOSCOPY_HINTS) or contains_any(watch_result_norm, COLONOSCOPY_HINTS)
    has_gastric = contains_any(report_title_norm, GASTROSCOPY_HINTS) or contains_any(watch_result_norm, GASTROSCOPY_HINTS)
    return has_colon and not has_gastric


def infer_gastro_subtype(report_title_norm: str, watch_result_norm: str) -> tuple[int, str]:
    text = report_title_norm + " " + watch_result_norm
    if contains_any(text, ["超声", "eus", "超声胃镜"]):
        return 3, GASTRO_SUBTYPE_NAMES[3]
    if contains_any(text, ["手术", "术中", "术后"]):
        return 1, GASTRO_SUBTYPE_NAMES[1]
    if contains_any(text, ["染色", "色素", "nbi", "放大"]):
        return 2, GASTRO_SUBTYPE_NAMES[2]
    return 0, GASTRO_SUBTYPE_NAMES[0]


def resolve_exam_dir(exam_dir: str | Path, dataset_root: str | Path | None = None) -> tuple[Path, bool]:
    exam_path = Path(exam_dir).expanduser()
    if exam_path.is_dir():
        return exam_path, False

    if dataset_root is None or not str(dataset_root).strip():
        return exam_path, False

    ds_root = Path(dataset_root).expanduser()
    if not ds_root.is_absolute():
        ds_root = ds_root.resolve()

    if not exam_path.is_absolute():
        candidate = (ds_root / exam_path).resolve()
        if candidate.is_dir():
            return candidate, True

    if len(exam_path.parts) >= 2:
        candidate = ds_root / exam_path.parts[-2] / exam_path.parts[-1]
        if candidate.is_dir():
            return candidate, True

    return exam_path, False


def collect_image_paths(exam_dir: str | Path) -> list[str]:
    base = Path(exam_dir)
    if not base.exists() or not base.is_dir():
        return []
    image_paths: list[str] = []
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(str(p))
    image_paths.sort()
    return image_paths


def load_report_rows(report_csv_path: str | Path) -> list[dict[str, str]]:
    report_csv = Path(report_csv_path)
    if not report_csv.is_file():
        raise FileNotFoundError(f"未找到报告 CSV: {report_csv}")

    with report_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_task_records(
    report_csv_path: str | Path,
    min_instances: int = 1,
    dataset_root: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    返回:
      gastro_records: 胃镜多标签样本
      colo_records: 肠镜二分类样本（同时带三分类扩展字段）
    """
    rows = load_report_rows(report_csv_path)
    gastro_records: list[dict[str, Any]] = []
    colo_records: list[dict[str, Any]] = []
    skipped_non_target = 0
    remapped_exam_dir_count = 0
    missing_exam_dir_count = 0
    insufficient_image_count = 0

    for row in _iter_progress(rows, total=len(rows), desc="构建任务样本"):
        exam_dir = str(row.get("exam_dir", "")).strip()
        report_title = str(row.get("reportTitle", "")).strip()
        watch_result = str(row.get("watchResult", "")).strip()
        img_num = to_int(row.get("img_num", 0))

        title_norm = normalize_text(report_title)
        result_norm = normalize_text(watch_result)

        is_gastro = is_gastroscopy_record(title_norm, result_norm)
        is_colo = is_colonoscopy_record(title_norm, result_norm)
        if not (is_gastro or is_colo):
            skipped_non_target += 1
            continue

        resolved_exam_dir, was_remapped = resolve_exam_dir(exam_dir, dataset_root=dataset_root)
        if was_remapped:
            remapped_exam_dir_count += 1

        image_paths = collect_image_paths(resolved_exam_dir)
        if len(image_paths) < min_instances:
            if not resolved_exam_dir.is_dir():
                missing_exam_dir_count += 1
            else:
                insufficient_image_count += 1
            continue

        if is_gastro:
            label_map = match_rule_map(result_norm, GASTRIC_LABEL_RULES)
            y = [label_map[k] for k in GASTRO_LABEL_NAMES]
            if sum(y) > 0:
                subtype_id, subtype_name = infer_gastro_subtype(title_norm, result_norm)
                gastro_records.append(
                    {
                        "exam_dir": str(resolved_exam_dir),
                        "report_title": report_title,
                        "watch_result": watch_result,
                        "img_num": img_num,
                        "image_paths": image_paths,
                        "labels": y,
                        "gastro_subtype_id": subtype_id,
                        "gastro_subtype_name": subtype_name,
                    }
                )

        if is_colo:
            has_normal = contains_any(result_norm, NORMAL_PATTERNS)
            has_polyp = contains_any(result_norm, POLYP_PATTERNS)
            has_multi_polyp = contains_any(result_norm, MULTI_POLYP_PATTERNS)
            has_other_lesion = contains_any(result_norm, LESION_TOKENS)

            binary_valid = True
            binary_label = -1
            if has_polyp:
                binary_label = 1
            elif has_normal and not has_other_lesion:
                binary_label = 0
            else:
                binary_valid = False

            if binary_valid:
                tri_label = 0
                count_label = -1
                if has_multi_polyp:
                    tri_label = 2
                    count_label = 1  # multi
                elif has_polyp:
                    tri_label = 1
                    count_label = 0  # single
                else:
                    tri_label = 0
                    count_label = -1

                colo_records.append(
                    {
                        "exam_dir": str(resolved_exam_dir),
                        "report_title": report_title,
                        "watch_result": watch_result,
                        "img_num": img_num,
                        "image_paths": image_paths,
                        "label": int(binary_label),
                        "tri_label": int(tri_label),
                        "count_label": int(count_label),
                    }
                )

    print(
        "[数据构建] 过滤统计: "
        f"总行数={len(rows)}, "
        f"非胃/肠记录={skipped_non_target}, "
        f"路径重定位={remapped_exam_dir_count}, "
        f"目录缺失={missing_exam_dir_count}, "
        f"图像不足={insufficient_image_count}, "
        f"胃镜有效样本={len(gastro_records)}, "
        f"肠镜有效样本={len(colo_records)}"
    )
    return gastro_records, colo_records


def split_records(
    records: list[dict[str, Any]],
    seed: int,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, list[dict[str, Any]]]:
    if len(records) == 0:
        return {"train": [], "val": [], "test": []}

    assert abs(sum(ratios) - 1.0) < 1e-8, "ratios 必须和为 1"

    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    shuffled = [records[i] for i in idx]

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_test = n - n_train - n_val

    train_records = shuffled[:n_train]
    val_records = shuffled[n_train : n_train + n_val]
    test_records = shuffled[n_train + n_val : n_train + n_val + n_test]

    return {"train": train_records, "val": val_records, "test": test_records}


def kfold_indices(num_samples: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits <= 1:
        raise ValueError("n_splits 必须 > 1")
    idx = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)

    result: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i], axis=0)
        result.append((train_idx, val_idx))
    return result


def build_image_transform(image_size: int, is_train: bool) -> transforms.Compose:
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class DemoMILBagDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        task: str,
        max_instances: int,
        min_instances: int,
        bag_sampling_strategy: str,
        is_train: bool,
        image_size: int,
        random_instance_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.records = records
        self.task = task
        self.max_instances = max_instances
        self.min_instances = min_instances
        self.strategy = bag_sampling_strategy
        self.is_train = is_train
        self.random_instance_dropout = random_instance_dropout if is_train else 0.0
        self.transform = build_image_transform(image_size=image_size, is_train=is_train)

    def __len__(self) -> int:
        return len(self.records)

    def _uniform_indices(self, n: int, k: int) -> list[int]:
        if k >= n:
            return list(range(n))
        step = (n - 1) / float(k - 1) if k > 1 else 0.0
        idx = [int(round(i * step)) for i in range(k)]
        idx = sorted(set(idx))
        # 去重后不足时补齐
        cur = 0
        while len(idx) < k:
            if cur not in idx:
                idx.append(cur)
            cur += 1
        return sorted(idx[:k])

    def _select_indices(self, n: int, rng: random.Random) -> list[int]:
        max_n = self.max_instances if self.max_instances > 0 else n
        k = min(n, max_n)

        if self.strategy == "random":
            if k >= n:
                indices = list(range(n))
            else:
                indices = sorted(rng.sample(range(n), k))
        elif self.strategy == "uniform":
            indices = self._uniform_indices(n=n, k=k)
        elif self.strategy == "all_if_small":
            if n <= max_n:
                indices = list(range(n))
            else:
                indices = self._uniform_indices(n=n, k=k)
        else:
            raise ValueError(f"未知 bag_sampling_strategy: {self.strategy}")

        if self.random_instance_dropout > 0 and len(indices) > self.min_instances:
            kept: list[int] = []
            for i in indices:
                if rng.random() > self.random_instance_dropout:
                    kept.append(i)
            if len(kept) < self.min_instances:
                needed = self.min_instances - len(kept)
                remain = [i for i in indices if i not in kept]
                kept.extend(remain[:needed])
            indices = sorted(kept)

        if len(indices) < self.min_instances:
            if len(indices) == 0:
                indices = [0]
            while len(indices) < self.min_instances:
                indices.append(indices[len(indices) % len(indices)])
            indices = sorted(indices)
        return indices

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        image_paths_all: list[str] = record["image_paths"]
        rng = random.Random((idx + 1) * 104729 if self.is_train else idx + 17)

        select_idx = self._select_indices(n=len(image_paths_all), rng=rng)
        selected_paths = [image_paths_all[i] for i in select_idx]

        images: list[torch.Tensor] = []
        for p in selected_paths:
            img = Image.open(p).convert("RGB")
            images.append(self.transform(img))
        bag_images = torch.stack(images, dim=0)  # [N, C, H, W]

        if self.task == "gastro_multilabel":
            label = torch.tensor(record["labels"], dtype=torch.float32)
            count_label = torch.tensor(-1, dtype=torch.long)
        elif self.task == "colo_binary":
            label = torch.tensor(record["label"], dtype=torch.long)
            count_label = torch.tensor(record.get("count_label", -1), dtype=torch.long)
        else:
            raise ValueError(f"未知 task: {self.task}")

        return {
            "images": bag_images,
            "label": label,
            "count_label": count_label,
            "exam_dir": record["exam_dir"],
            "image_paths": selected_paths,
            "report_title": record.get("report_title", ""),
            "img_num": int(record.get("img_num", len(image_paths_all))),
            "meta": {
                "gastro_subtype_id": int(record.get("gastro_subtype_id", -1)),
                "gastro_subtype_name": record.get("gastro_subtype_name", ""),
                "tri_label": int(record.get("tri_label", -1)),
            },
        }


def demo_mil_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    bsz = len(batch)
    max_n = max(item["images"].shape[0] for item in batch)
    c, h, w = batch[0]["images"].shape[1:]

    images = torch.zeros((bsz, max_n, c, h, w), dtype=batch[0]["images"].dtype)
    mask = torch.zeros((bsz, max_n), dtype=torch.bool)

    labels: list[torch.Tensor] = []
    count_labels: list[torch.Tensor] = []
    exam_dirs: list[str] = []
    image_paths: list[list[str]] = []
    report_titles: list[str] = []
    img_nums: list[int] = []
    metas: list[dict[str, Any]] = []

    for i, item in enumerate(batch):
        n = item["images"].shape[0]
        images[i, :n] = item["images"]
        mask[i, :n] = True
        labels.append(item["label"])
        count_labels.append(item["count_label"])
        exam_dirs.append(item["exam_dir"])
        image_paths.append(item["image_paths"])
        report_titles.append(item["report_title"])
        img_nums.append(item["img_num"])
        metas.append(item["meta"])

    if labels[0].ndim == 0:
        labels_tensor = torch.stack(labels, dim=0).long()
    else:
        labels_tensor = torch.stack(labels, dim=0).float()

    count_labels_tensor = torch.stack(count_labels, dim=0).long()

    return {
        "images": images,
        "mask": mask,
        "labels": labels_tensor,
        "count_labels": count_labels_tensor,
        "exam_dirs": exam_dirs,
        "image_paths": image_paths,
        "report_titles": report_titles,
        "img_nums": img_nums,
        "metas": metas,
    }
