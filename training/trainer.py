from __future__ import annotations

import csv
import inspect
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from .losses import build_binary_criterion, build_multilabel_criterion
from .metrics import compute_binary_metrics, compute_multilabel_metrics, to_builtin_type
from .visualization import save_confusion_matrix, save_loss_curve, save_pr_curve, save_roc_curve

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@dataclass
class TrainerConfig:
    task_type: str
    max_epochs: int
    patience: int
    lr: float
    optimizer_name: str
    weight_decay: float
    warmup_ratio: float
    grad_accum_steps: int
    amp: bool
    monitor_metric: str
    monitor_mode: str
    topk_evidence: int
    loss_name: str
    pos_weight: list[float] | None
    aux_loss_weights: dict[str, float]
    use_multi_gpu: bool
    resume_path: str | None
    run_test: bool


def _is_scalar_metric(value: Any) -> bool:
    return isinstance(value, (int, float, bool, np.generic))


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _format_metric(value: Any) -> str:
    numeric = _to_float(value)
    if math.isnan(numeric):
        return "nan"
    return f"{numeric:.4f}"


def _format_epoch_value(value: Any) -> str:
    try:
        epoch = int(value)
    except Exception:
        return "epoch---"
    if epoch <= 0:
        return "epoch---"
    return f"epoch{epoch:03d}"


def _format_metric_field(name: str, value: Any, *, highlight: bool = False) -> str:
    suffix = "*" if highlight else ""
    return f"{name}{suffix}={_format_metric(value)}"


def _extract_scalar_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in metrics.items():
        if _is_scalar_metric(value):
            flattened[key] = to_builtin_type(value)
    return flattened


class CSVLogger:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = ["epoch", "split", "loss", "lr", "monitor_name", "monitor_value"]
        self.rows: list[dict[str, Any]] = []
        if self.csv_path.is_file():
            with self.csv_path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames:
                    self.fieldnames = list(reader.fieldnames)
                self.rows = [dict(row) for row in reader]

    def write(
        self,
        *,
        epoch: int,
        split: str,
        loss: float,
        lr: float,
        monitor_name: str,
        monitor_value: float,
        metrics: dict[str, Any],
    ) -> None:
        row = {
            "epoch": int(epoch),
            "split": split,
            "loss": loss,
            "lr": lr,
            "monitor_name": monitor_name,
            "monitor_value": monitor_value,
        }
        row.update(_extract_scalar_metrics(metrics))
        self.rows.append(row)

        for key in row.keys():
            if key not in self.fieldnames:
                self.fieldnames.append(key)

        with self.csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()
            for item in self.rows:
                writer.writerow({field: item.get(field, "") for field in self.fieldnames})


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        cfg: TrainerConfig,
        run_dir: Path,
        train_loader,
        val_loader,
        test_loader,
        label_names: list[str],
        class_names: list[str],
        seed: int,
        on_validation_epoch_end: Callable[[int, float, dict[str, Any]], None] | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.run_dir = run_dir
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.label_names = label_names
        self.class_names = class_names
        self.on_validation_epoch_end = on_validation_epoch_end

        set_seed(seed)

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.logger = CSVLogger(self.run_dir / "log.csv")
        self.loss_curve_path = self.run_dir / "loss_curve.png"
        self.last_confusion_matrix_path = self.run_dir / "last_confusion_matrix.png"
        self.loss_history = self._restore_loss_history()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        if self.cfg.use_multi_gpu and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        pos_weight_tensor = None
        if self.cfg.pos_weight is not None:
            pos_weight_tensor = torch.tensor(self.cfg.pos_weight, dtype=torch.float32, device=self.device)

        if self.cfg.task_type == "gastro_multilabel":
            self.criterion = build_multilabel_criterion(loss_name=self.cfg.loss_name, pos_weight=pos_weight_tensor)
        elif self.cfg.task_type == "colonoscopy_binary":
            self.criterion = build_binary_criterion(loss_name=self.cfg.loss_name, pos_weight=pos_weight_tensor)
        else:
            raise ValueError(f"未知 task_type: {self.cfg.task_type}")

        self.optimizer = self._build_optimizer()

        total_steps = max(
            1,
            math.ceil(len(self.train_loader) / max(1, self.cfg.grad_accum_steps)) * max(1, self.cfg.max_epochs),
        )
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.scaler = GradScaler(enabled=self.cfg.amp and self.device.type == "cuda")

        self.start_epoch = 1
        self.primary_best_metric = -float("inf") if self.cfg.monitor_mode == "max" else float("inf")
        self.primary_best_epoch = -1
        self.last_path = self.ckpt_dir / "last.ckpt"
        self.best_trackers = {
            "best_macro_f1": {
                "metric_name": "macro_f1",
                "mode": "max",
                "best_value": -float("inf"),
                "best_epoch": -1,
                "ckpt_path": self.ckpt_dir / "best_macro_f1.ckpt",
                "artifact_dir": self.run_dir / "test_macro_f1",
            },
            "best_micro_f1": {
                "metric_name": "micro_f1",
                "mode": "max",
                "best_value": -float("inf"),
                "best_epoch": -1,
                "ckpt_path": self.ckpt_dir / "best_micro_f1.ckpt",
                "artifact_dir": self.run_dir / "test_micro_f1",
            },
            "best_val_loss": {
                "metric_name": "val_loss",
                "mode": "min",
                "best_value": float("inf"),
                "best_epoch": -1,
                "ckpt_path": self.ckpt_dir / "best_val_loss.ckpt",
                "artifact_dir": self.run_dir / "test_val_loss",
            },
        }

        if self.cfg.resume_path:
            self._load_checkpoint(Path(self.cfg.resume_path), strict=False, load_training_state=True)
        self._forward_accepts_labels = "labels" in inspect.signature(self.raw_model.forward).parameters

    @property
    def raw_model(self) -> nn.Module:
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _iter_progress(self, loader, desc: str):
        if tqdm is not None:
            return tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, mininterval=0.8, smoothing=0.05)
        return loader

    def _build_optimizer(self):
        optimizer_name = self.cfg.optimizer_name.strip().lower()
        if optimizer_name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        if optimizer_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        if optimizer_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                momentum=0.9,
                nesterov=True,
            )
        raise ValueError(f"未知 optimizer_name: {self.cfg.optimizer_name}")

    def _restore_loss_history(self) -> list[dict[str, float]]:
        if not self.logger.rows:
            return []

        epoch_state: dict[int, dict[str, float]] = {}
        for row in self.logger.rows:
            try:
                epoch = int(row.get("epoch", 0))
            except Exception:
                continue
            if epoch <= 0:
                continue
            state = epoch_state.setdefault(epoch, {"epoch": epoch})
            split = str(row.get("split", "")).strip()
            if split == "train":
                state["train_loss"] = _to_float(row.get("loss", float("nan")))
            elif split == "val":
                state["val_loss"] = _to_float(row.get("loss", float("nan")))

        restored: list[dict[str, float]] = []
        for epoch in sorted(epoch_state):
            state = epoch_state[epoch]
            if "train_loss" not in state or "val_loss" not in state:
                continue
            restored.append(
                {
                    "epoch": int(state["epoch"]),
                    "train_loss": float(state["train_loss"]),
                    "val_loss": float(state["val_loss"]),
                }
            )
        return restored

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            "images": batch["images"].to(self.device, non_blocking=True),
            "mask": batch["mask"].to(self.device, non_blocking=True),
            "labels": batch["labels"].to(self.device, non_blocking=True),
            "exam_dirs": batch["exam_dirs"],
            "image_paths": batch["image_paths"],
            "report_titles": batch["report_titles"],
            "img_nums": batch["img_nums"],
            "metas": batch["metas"],
        }

    def _forward_model(self, batch: dict[str, Any]) -> dict[str, Any]:
        kwargs = {
            "images": batch["images"],
            "mask": batch["mask"],
        }
        if self._forward_accepts_labels:
            kwargs["labels"] = batch["labels"]
        return self.model(**kwargs)

    def _primary_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = self.criterion(logits, labels.float())
        return loss.mean() if loss.ndim > 0 else loss

    def _aux_loss(self, outputs: dict[str, Any]) -> torch.Tensor:
        aux_losses = outputs.get("aux_losses", {})
        if not isinstance(aux_losses, dict) or not aux_losses:
            return torch.zeros((), device=self.device)

        total = torch.zeros((), device=self.device)
        for key, value in aux_losses.items():
            if not torch.is_tensor(value):
                continue
            current = value.mean() if value.ndim > 0 else value
            weight = self.cfg.aux_loss_weights.get(key, 1.0)
            total = total + weight * current
        return total

    def _extract_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    def _compute_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
        if self.cfg.task_type == "gastro_multilabel":
            return compute_multilabel_metrics(y_true=y_true, y_prob=y_prob, label_names=self.label_names, threshold=0.5)
        return compute_binary_metrics(y_true=y_true, y_prob=y_prob, class_names=self.class_names, threshold=0.5)

    def _monitor_value(self, metrics: dict[str, Any], loss: float | None = None) -> float:
        if self.cfg.monitor_metric == "val_loss":
            numeric = float(loss) if loss is not None else float("nan")
            if math.isnan(numeric):
                return float("inf")
            return numeric
        value = metrics.get(self.cfg.monitor_metric, float("nan"))
        if _is_scalar_metric(value):
            numeric = float(value)
            if math.isnan(numeric):
                return -float("inf") if self.cfg.monitor_mode == "max" else float("inf")
            return numeric
        return -float("inf") if self.cfg.monitor_mode == "max" else float("inf")

    def _is_improved(self, current: float, best: float, mode: str) -> bool:
        if math.isnan(current):
            return False
        if mode == "max":
            return current > best
        return current < best

    def _serialize_best_trackers(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for alias, tracker in self.best_trackers.items():
            payload[alias] = {
                "metric_name": tracker["metric_name"],
                "mode": tracker["mode"],
                "best_value": tracker["best_value"],
                "best_epoch": tracker["best_epoch"],
            }
        return payload

    def _save_checkpoint(self, path: Path, epoch: int, monitor: float) -> None:
        payload = {
            "epoch": epoch,
            "primary_best_metric": self.primary_best_metric,
            "primary_best_epoch": self.primary_best_epoch,
            "monitor": monitor,
            "model_state": self.raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "cfg": self.cfg.__dict__,
            "best_trackers": self._serialize_best_trackers(),
        }
        torch.save(payload, path)

    def _load_checkpoint(self, path: Path, strict: bool = True, load_training_state: bool = False) -> None:
        if not path.is_file():
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.raw_model.load_state_dict(checkpoint["model_state"], strict=strict)

        if not load_training_state:
            return

        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state"])

        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self.primary_best_metric = float(checkpoint.get("primary_best_metric", self.primary_best_metric))
        self.primary_best_epoch = int(checkpoint.get("primary_best_epoch", self.primary_best_epoch))

        trackers = checkpoint.get("best_trackers", {})
        if isinstance(trackers, dict):
            for alias, saved_tracker in trackers.items():
                if alias not in self.best_trackers or not isinstance(saved_tracker, dict):
                    continue
                self.best_trackers[alias]["best_value"] = float(
                    saved_tracker.get("best_value", self.best_trackers[alias]["best_value"])
                )
                self.best_trackers[alias]["best_epoch"] = int(
                    saved_tracker.get("best_epoch", self.best_trackers[alias]["best_epoch"])
                )

    def _save_best_artifacts(self, alias: str, epoch: int, val_metrics: dict[str, Any]) -> None:
        save_confusion_matrix(
            val_metrics,
            self.run_dir / f"{alias}_val_confusion_matrix.png",
            title=f"Validation Confusion Matrix - Epoch {epoch}",
        )

    def _update_best_checkpoints(self, epoch: int, val_loss: float, val_metrics: dict[str, Any]) -> set[str]:
        improved_aliases: set[str] = set()
        for alias, tracker in self.best_trackers.items():
            metric_name = str(tracker["metric_name"])
            current_value = val_loss if metric_name == "val_loss" else _to_float(val_metrics.get(metric_name, float("nan")))
            if tracker["best_epoch"] < 0 or self._is_improved(current_value, float(tracker["best_value"]), str(tracker["mode"])):
                tracker["best_value"] = current_value
                tracker["best_epoch"] = epoch
                improved_aliases.add(alias)
                self._save_checkpoint(Path(tracker["ckpt_path"]), epoch, current_value)
                self._save_best_artifacts(alias, epoch, val_metrics)
        return improved_aliases

    def _log_epoch(
        self,
        *,
        epoch: int,
        lr: float,
        train_loss: float,
        train_metrics: dict[str, Any],
        val_loss: float,
        val_metrics: dict[str, Any],
    ) -> None:
        train_monitor = self._monitor_value(train_metrics, loss=train_loss)
        val_monitor = self._monitor_value(val_metrics, loss=val_loss)
        self.logger.write(
            epoch=epoch,
            split="train",
            loss=train_loss,
            lr=lr,
            monitor_name=self.cfg.monitor_metric,
            monitor_value=train_monitor,
            metrics=train_metrics,
        )
        self.logger.write(
            epoch=epoch,
            split="val",
            loss=val_loss,
            lr=lr,
            monitor_name=self.cfg.monitor_metric,
            monitor_value=val_monitor,
            metrics=val_metrics,
        )

    def _update_epoch_artifacts(self, epoch: int, train_loss: float, val_loss: float, val_metrics: dict[str, Any]) -> None:
        self.loss_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        save_loss_curve(self.loss_history, self.loss_curve_path)
        save_confusion_matrix(
            val_metrics,
            self.last_confusion_matrix_path,
            title=f"Validation Confusion Matrix - Epoch {epoch}",
        )

    def _print_epoch_summary(
        self,
        *,
        epoch: int,
        lr: float,
        train_loss: float,
        train_metrics: dict[str, Any],
        val_loss: float,
        val_metrics: dict[str, Any],
        improved_aliases: set[str],
    ) -> None:
        print(f"Epoch {epoch:03d}/{self.cfg.max_epochs:03d} | lr={lr:.6e}")
        print(
            "  train "
            f"{_format_metric_field('loss', train_loss)} "
            f"{_format_metric_field('macro_f1', train_metrics.get('macro_f1'))} "
            f"{_format_metric_field('micro_f1', train_metrics.get('micro_f1'))}"
        )
        val_fields = [
            _format_metric_field("loss", val_loss, highlight="best_val_loss" in improved_aliases),
            _format_metric_field("macro_f1", val_metrics.get("macro_f1"), highlight="best_macro_f1" in improved_aliases),
            _format_metric_field("micro_f1", val_metrics.get("micro_f1"), highlight="best_micro_f1" in improved_aliases),
        ]
        print(
            "  val   "
            + " ".join(val_fields)
        )
        best_fields = [
            _format_metric_field("loss", self.best_trackers["best_val_loss"]["best_value"]),
            _format_metric_field("macro_f1", self.best_trackers["best_macro_f1"]["best_value"]),
            _format_metric_field("micro_f1", self.best_trackers["best_micro_f1"]["best_value"]),
        ]
        print(
            "  best  "
            + " ".join(best_fields)
        )
        epoch_fields = [
            _format_epoch_value(self.best_trackers["best_val_loss"]["best_epoch"]).ljust(len(best_fields[0])),
            _format_epoch_value(self.best_trackers["best_macro_f1"]["best_epoch"]).ljust(len(best_fields[1])),
            _format_epoch_value(self.best_trackers["best_micro_f1"]["best_epoch"]).ljust(len(best_fields[2])),
        ]
        print(
            "        "
            + " ".join(epoch_fields)
        )

    def _topk_paths_multilabel(
        self,
        attention: torch.Tensor,
        mask: torch.Tensor,
        image_paths: list[list[str]],
        topk: int,
    ) -> list[dict[str, list[str]]]:
        results: list[dict[str, list[str]]] = []
        for batch_index in range(attention.shape[0]):
            valid_num = int(mask[batch_index].sum().item())
            paths = image_paths[batch_index]
            current: dict[str, list[str]] = {}
            for label_index, label_name in enumerate(self.label_names):
                k = min(topk, valid_num)
                if k <= 0:
                    current[label_name] = []
                    continue
                indices = torch.topk(attention[batch_index, label_index, :valid_num], k=k, dim=-1).indices
                current[label_name] = [paths[idx] for idx in indices.detach().cpu().tolist() if idx < len(paths)]
            results.append(current)
        return results

    def _topk_paths_binary(
        self,
        attention: torch.Tensor,
        mask: torch.Tensor,
        image_paths: list[list[str]],
        topk: int,
    ) -> list[list[str]]:
        results: list[list[str]] = []
        for batch_index in range(attention.shape[0]):
            valid_num = int(mask[batch_index].sum().item())
            k = min(topk, valid_num)
            if k <= 0:
                results.append([])
                continue
            indices = torch.topk(attention[batch_index, :valid_num], k=k, dim=-1).indices.detach().cpu().tolist()
            paths = image_paths[batch_index]
            results.append([paths[idx] for idx in indices if idx < len(paths)])
        return results

    def _run_one_epoch(
        self,
        loader,
        epoch: int,
        train_mode: bool,
        split: str,
    ) -> tuple[float, dict[str, Any]]:
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_batches = 0
        y_true_list: list[np.ndarray] = []
        y_prob_list: list[np.ndarray] = []

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch_cpu in enumerate(self._iter_progress(loader, desc=f"{split}-epoch{epoch}"), start=1):
            batch = self._move_batch_to_device(batch_cpu)

            with torch.set_grad_enabled(train_mode):
                with autocast(enabled=self.cfg.amp and self.device.type == "cuda"):
                    outputs = self._forward_model(batch)
                    logits = outputs["logits"]
                    loss_main = self._primary_loss(logits=logits, labels=batch["labels"])
                    loss_aux = self._aux_loss(outputs)
                    loss = loss_main + loss_aux

                if train_mode:
                    loss_step = loss / max(1, self.cfg.grad_accum_steps)
                    self.scaler.scale(loss_step).backward()
                    if step % max(1, self.cfg.grad_accum_steps) == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.scheduler.step()

            total_loss += float(loss.detach().cpu().item())
            total_batches += 1

            probs = self._extract_probabilities(logits.detach())
            y_prob = probs.detach().cpu().numpy()
            y_true = batch["labels"].detach().cpu().numpy()
            y_prob_list.append(y_prob)
            y_true_list.append(y_true)

        if train_mode and total_batches % max(1, self.cfg.grad_accum_steps) != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

        if total_batches == 0:
            return float("nan"), {self.cfg.monitor_metric: float("nan")}

        avg_loss = total_loss / max(1, total_batches)
        y_true_all = np.concatenate(y_true_list, axis=0)
        y_prob_all = np.concatenate(y_prob_list, axis=0)
        metrics = self._compute_metrics(y_true=y_true_all, y_prob=y_prob_all)

        return avg_loss, metrics

    def _build_test_result_rows(self, test_results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for alias, payload in test_results.items():
            row = {
                "checkpoint_alias": alias,
                "checkpoint_path": payload["checkpoint_path"],
                "best_epoch": payload["best_epoch"],
                "selection_metric": payload["selection_metric"],
                "selection_value": payload["selection_value"],
                "test_loss": payload["test_loss"],
            }
            row.update(_extract_scalar_metrics(payload["metrics"]))
            rows.append(row)
        return rows

    def _write_test_result_csv(self, test_results: dict[str, dict[str, Any]]) -> None:
        rows = self._build_test_result_rows(test_results)
        if not rows:
            return

        fieldnames: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        for csv_path in (self.run_dir / "test_result.csv", self.run_dir / "test_report.csv"):
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({field: row.get(field, "") for field in fieldnames})

    def _save_test_artifacts(self, directory: Path, metrics: dict[str, Any], checkpoint_alias: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        save_confusion_matrix(
            metrics,
            directory / f"{checkpoint_alias}_test_confusion_matrix.png",
            title=f"Test Confusion Matrix - {checkpoint_alias}",
        )
        save_roc_curve(metrics, directory / "roc_curve.png", title=f"Test ROC Curve - {checkpoint_alias}")
        save_pr_curve(metrics, directory / "pr_curve.png", title=f"Test PR Curve - {checkpoint_alias}")

    def _write_test_metrics_file(self, directory: Path, payload: dict[str, Any]) -> None:
        metrics_path = directory / "metrics.json"
        metrics_path.write_text(
            json.dumps(to_builtin_type(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _evaluate_best_checkpoints(self) -> dict[str, dict[str, Any]]:
        test_results: dict[str, dict[str, Any]] = {}

        for alias, tracker in self.best_trackers.items():
            ckpt_path = Path(tracker["ckpt_path"])
            if not ckpt_path.is_file():
                continue

            self._load_checkpoint(ckpt_path, strict=True, load_training_state=False)
            test_loss, test_metrics = self._run_one_epoch(
                loader=self.test_loader,
                epoch=max(1, int(tracker["best_epoch"])),
                train_mode=False,
                split=f"test-{alias}",
            )

            test_dir = Path(tracker["artifact_dir"])
            self._save_test_artifacts(test_dir, test_metrics, alias)

            test_results[alias] = {
                "checkpoint_alias": alias,
                "checkpoint_path": str(ckpt_path),
                "best_epoch": int(tracker["best_epoch"]),
                "selection_metric": str(tracker["metric_name"]),
                "selection_value": float(tracker["best_value"]),
                "test_loss": test_loss,
                "metrics": to_builtin_type(test_metrics),
                "result_dir": str(test_dir),
            }
            self._write_test_metrics_file(test_dir, test_results[alias])

        return test_results

    def _metric_value_from_payload(self, payload: dict[str, Any], metric_key: str) -> Any:
        if metric_key == "test_loss":
            return payload.get("test_loss", float("nan"))
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            return float("nan")
        return metrics.get(metric_key, float("nan"))

    def _print_named_metric_lines(
        self,
        payload: dict[str, Any],
        metric_mapping: list[tuple[str, str]],
        *,
        indent: str = "  ",
    ) -> None:
        for display_name, metric_key in metric_mapping:
            value = self._metric_value_from_payload(payload, metric_key)
            print(f"{indent}{display_name:<22} {_format_metric(value)}")

    def _print_per_class_metric_block(
        self,
        payload: dict[str, Any],
        metric_prefix: str,
        title: str,
        names: list[str],
    ) -> None:
        if not names:
            return
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            return
        print(f"  {title}")
        for name in names:
            metric_key = f"{metric_prefix}_{name}"
            if metric_key not in metrics:
                continue
            print(f"    {name:<18} {_format_metric(metrics.get(metric_key))}")

    def _print_test_summary(self, test_results: dict[str, dict[str, Any]]) -> None:
        if not test_results:
            return

        print("-" * 72)
        print("测试结果")
        ordered_aliases = ("best_macro_f1", "best_micro_f1", "best_val_loss")
        for alias in ordered_aliases:
            payload = test_results.get(alias)
            if not isinstance(payload, dict):
                continue
            label = f"  {alias:<13} "
            print(
                f"{label}"
                f"loss={_format_metric(payload.get('test_loss'))} "
                f"macro_f1={_format_metric(payload.get('metrics', {}).get('macro_f1'))} "
                f"micro_f1={_format_metric(payload.get('metrics', {}).get('micro_f1'))}"
            )
        print("-" * 72)

    def fit(self) -> dict[str, Any]:
        no_improve = 0

        for epoch in range(self.start_epoch, self.cfg.max_epochs + 1):
            train_loss, train_metrics = self._run_one_epoch(
                loader=self.train_loader,
                epoch=epoch,
                train_mode=True,
                split="train",
            )
            val_loss, val_metrics = self._run_one_epoch(
                loader=self.val_loader,
                epoch=epoch,
                train_mode=False,
                split="val",
            )

            lr = self._current_lr()
            val_monitor = self._monitor_value(val_metrics, loss=val_loss)

            self._log_epoch(
                epoch=epoch,
                lr=lr,
                train_loss=train_loss,
                train_metrics=train_metrics,
                val_loss=val_loss,
                val_metrics=val_metrics,
            )
            self._update_epoch_artifacts(epoch, train_loss, val_loss, val_metrics)

            if self.primary_best_epoch < 0 or self._is_improved(val_monitor, self.primary_best_metric, self.cfg.monitor_mode):
                self.primary_best_metric = val_monitor
                self.primary_best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1

            improved_aliases = self._update_best_checkpoints(epoch, val_loss, val_metrics)
            self._save_checkpoint(self.last_path, epoch, val_monitor)
            self._print_epoch_summary(
                epoch=epoch,
                lr=lr,
                train_loss=train_loss,
                train_metrics=train_metrics,
                val_loss=val_loss,
                val_metrics=val_metrics,
                improved_aliases=improved_aliases,
            )

            if self.on_validation_epoch_end is not None:
                self.on_validation_epoch_end(epoch, val_loss, val_metrics)

            if no_improve >= self.cfg.patience:
                print(f"提前停止：验证集 {self.cfg.monitor_metric} 连续 {self.cfg.patience} 个 epoch 未提升。")
                break

        test_results: dict[str, dict[str, Any]] = {}
        if self.cfg.run_test:
            test_results = self._evaluate_best_checkpoints()
            self._write_test_result_csv(test_results)
            self._print_test_summary(test_results)
        else:
            print("自动探索模式：跳过测试集评估，仅使用验证集结果进行参数选择。")

        result = {
            "primary_monitor_metric": self.cfg.monitor_metric,
            "primary_monitor_mode": self.cfg.monitor_mode,
            "primary_best_epoch": self.primary_best_epoch,
            "primary_best_value": self.primary_best_metric,
            "best_checkpoints": {
                alias: {
                    "metric_name": tracker["metric_name"],
                    "mode": tracker["mode"],
                    "best_value": tracker["best_value"],
                    "best_epoch": tracker["best_epoch"],
                    "checkpoint_path": str(tracker["ckpt_path"]),
                    "artifact_dir": str(tracker["artifact_dir"]),
                }
                for alias, tracker in self.best_trackers.items()
            },
            "test_results": test_results,
        }
        return result
