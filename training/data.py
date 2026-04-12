from __future__ import annotations

import csv
import hashlib
import math
import os
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
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
IMAGE_CACHE_MODES = {"none", "memory", "disk", "memory_and_disk"}


def _iter_progress(iterable, total: int | None = None, desc: str = "处理中"):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def to_int(value: Any) -> int:
    if value is None:
        return 0
    cleaned = re.sub(r"[^0-9]", "", str(value))
    return int(cleaned) if cleaned else 0


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
    rows = load_task_rows(task_csv_path)
    records: list[dict[str, Any]] = []

    for row in _iter_progress(rows, total=len(rows), desc=f"构建{task_name}样本"):
        exam_dir = str(row.get("exam_dir", "")).strip()
        report_title = str(row.get("reportTitle", "")).strip()
        watch_result = str(row.get("watchResult", "")).strip()
        img_num = to_int(row.get("img_num", 0))

        resolved_exam_dir, _ = resolve_exam_dir(exam_dir, dataset_root=dataset_root)

        image_paths = collect_image_paths(resolved_exam_dir)
        if len(image_paths) < min_instances:
            continue

        if task_name == "gastro_multilabel":
            labels = [to_int(row.get(label_name, 0)) for label_name in GASTRO_LABEL_NAMES]
            if sum(labels) <= 0:
                continue
            records.append(
                {
                    "exam_dir": str(resolved_exam_dir),
                    "report_title": report_title,
                    "watch_result": watch_result,
                    "img_num": img_num,
                    "image_paths": image_paths,
                    "labels": labels,
                }
            )
        elif task_name == "colonoscopy_binary":
            label = to_int(row.get("binary_label", -1))
            if label not in {0, 1}:
                continue
            records.append(
                {
                    "exam_dir": str(resolved_exam_dir),
                    "report_title": report_title,
                    "watch_result": watch_result,
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
) -> dict[str, list[dict[str, Any]]]:
    if len(records) == 0:
        return {"train": [], "val": [], "test": []}

    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("ratios 必须和为 1")

    rng = random.Random(seed)
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
        memory_cache_size: int = 0,
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
        self.memory_cache = _LRUImageArrayCache(memory_cache_size if self.use_memory_cache else 0)

    def __len__(self) -> int:
        return len(self.records)

    def _memory_cache_key(self, image_path: str | Path) -> str:
        return str(Path(image_path).expanduser().resolve())

    def _disk_cache_path(self, image_path: str | Path) -> Path | None:
        if self.image_cache_dir is None:
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
        return self.image_cache_dir / digest[:2] / f"{digest}.npy"

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

        cache_path = self._disk_cache_path(image_path)
        if cache_path is not None and cache_path.is_file():
            try:
                with cache_path.open("rb") as file:
                    cached_array = np.load(file, allow_pickle=False)
                if self.use_memory_cache:
                    self.memory_cache.put(memory_key, cached_array)
                return cached_array
            except Exception:
                cache_path.unlink(missing_ok=True)

        image_array = self._load_source_image_array(image_path)
        if cache_path is not None:
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
            cache_path = self._disk_cache_path(image_path)
            if cache_path is not None and cache_path.is_file():
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

    def _select_indices(self, num_instances: int, rng: random.Random) -> list[int]:
        max_instances = self.max_instances if self.max_instances > 0 else num_instances
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

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_paths_all: list[str] = record["image_paths"]
        rng = random.Random((index + 1) * 104729 if self.is_train else index + 17)

        selected_indices = self._select_indices(num_instances=len(image_paths_all), rng=rng)
        selected_paths = [image_paths_all[item] for item in selected_indices]

        images = []
        for path in selected_paths:
            image = Image.fromarray(self._load_image_array(path), mode="RGB")
            images.append(self.transform(image))
        bag_images = torch.stack(images, dim=0)

        if self.task_name == "gastro_multilabel":
            label = torch.tensor(record["labels"], dtype=torch.float32)
        elif self.task_name == "colonoscopy_binary":
            label = torch.tensor(record["label"], dtype=torch.long)
        else:
            raise ValueError(f"未知 task_name: {self.task_name}")

        return {
            "images": bag_images,
            "label": label,
            "exam_dir": record["exam_dir"],
            "image_paths": selected_paths,
            "report_title": record.get("report_title", ""),
            "img_num": int(record.get("img_num", len(image_paths_all))),
            "meta": {},
        }


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

    if labels[0].ndim == 0:
        labels_tensor = torch.stack(labels, dim=0).long()
    else:
        labels_tensor = torch.stack(labels, dim=0).float()

    return {
        "images": images,
        "mask": mask,
        "labels": labels_tensor,
        "exam_dirs": exam_dirs,
        "image_paths": image_paths,
        "report_titles": report_titles,
        "img_nums": img_nums,
        "metas": metas,
    }


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
        self.max_instances_per_bag = max(1, int(max_instances_per_bag))
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
            num_instances = min(num_instances, self.max_instances_per_bag)
            num_instances = max(num_instances, self.min_instances_per_bag)
            self.instance_counts.append(num_instances)

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
