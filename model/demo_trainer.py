from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from .demo_losses import build_binary_or_multiclass_criterion, build_multilabel_criterion
from .demo_metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    compute_multilabel_metrics,
    to_builtin_type,
)

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainerConfig:
    task_type: str
    model_family: str
    num_classes: int
    num_labels: int
    max_epochs: int
    patience: int
    lr: float
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


class DemoCSVLogger:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "split", "loss", "monitor", "metrics_json"])
                writer.writeheader()

    def write(self, epoch: int, split: str, loss: float, monitor: float, metrics: dict[str, Any]) -> None:
        row = {
            "epoch": epoch,
            "split": split,
            "loss": loss,
            "monitor": monitor,
            "metrics_json": json.dumps(to_builtin_type(metrics), ensure_ascii=False),
        }
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "split", "loss", "monitor", "metrics_json"])
            writer.writerow(row)


class DemoTrainer:
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
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.run_dir = run_dir
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.label_names = label_names
        self.class_names = class_names

        set_seed(seed)

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir = self.run_dir / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        self.logger = DemoCSVLogger(self.run_dir / "train_log.csv")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        if self.cfg.use_multi_gpu and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        pos_weight_tensor = None
        if self.cfg.pos_weight is not None:
            pos_weight_tensor = torch.tensor(self.cfg.pos_weight, dtype=torch.float32, device=self.device)

        if self.cfg.task_type == "gastro_multilabel":
            self.criterion = build_multilabel_criterion(
                loss_name=self.cfg.loss_name,
                pos_weight=pos_weight_tensor,
            )
        else:
            self.criterion = build_binary_or_multiclass_criterion(
                num_classes=self.cfg.num_classes,
                loss_name=self.cfg.loss_name,
                pos_weight=pos_weight_tensor,
            )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        total_steps = max(1, math.ceil(len(self.train_loader) / max(1, self.cfg.grad_accum_steps)) * self.cfg.max_epochs)
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.scaler = GradScaler(enabled=self.cfg.amp and self.device.type == "cuda")

        self.start_epoch = 1
        self.best_metric = -float("inf") if self.cfg.monitor_mode == "max" else float("inf")
        self.best_epoch = -1
        self.best_path = self.ckpt_dir / "best.pt"
        self.last_path = self.ckpt_dir / "last.pt"

        if self.cfg.resume_path:
            self._load_checkpoint(Path(self.cfg.resume_path), strict=False)

    @property
    def _raw_model(self) -> nn.Module:
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _iter_progress(self, loader, desc: str):
        if tqdm is not None:
            return tqdm(loader, desc=desc, leave=False)
        return loader

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            "images": batch["images"].to(self.device, non_blocking=True),
            "mask": batch["mask"].to(self.device, non_blocking=True),
            "labels": batch["labels"].to(self.device, non_blocking=True),
            "count_labels": batch["count_labels"].to(self.device, non_blocking=True),
            "exam_dirs": batch["exam_dirs"],
            "image_paths": batch["image_paths"],
            "report_titles": batch["report_titles"],
            "img_nums": batch["img_nums"],
            "metas": batch["metas"],
        }

    def _forward(self, batch: dict[str, Any], train_mode: bool) -> dict[str, Any]:
        images = batch["images"]
        mask = batch["mask"]

        if self.cfg.model_family == "gastro_advanced":
            subtype_ids = torch.tensor(
                [int(m.get("gastro_subtype_id", -1)) for m in batch["metas"]],
                dtype=torch.long,
                device=self.device,
            )
            return self.model(
                images=images,
                mask=mask,
                subtype_ids=subtype_ids,
                labels=batch["labels"] if train_mode else None,
            )

        if self.cfg.model_family == "colo_advanced":
            return self.model(
                images=images,
                mask=mask,
                labels=batch["labels"] if train_mode else None,
                count_labels=batch["count_labels"] if train_mode else None,
            )

        return self.model(images=images, mask=mask)

    def _primary_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.cfg.task_type == "gastro_multilabel":
            return self.criterion(logits, labels.float())

        if self.cfg.num_classes == 2 and logits.ndim == 1:
            return self.criterion(logits, labels.float())
        return self.criterion(logits, labels.long())

    def _aux_loss(self, outputs: dict[str, Any]) -> torch.Tensor:
        aux = outputs.get("aux_losses", {})
        if not isinstance(aux, dict) or len(aux) == 0:
            return torch.zeros((), device=self.device)

        total = torch.zeros((), device=self.device)
        for k, v in aux.items():
            if not torch.is_tensor(v):
                continue
            w = self.cfg.aux_loss_weights.get(k, 1.0)
            total = total + w * v
        return total

    def _extract_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        if self.cfg.task_type == "gastro_multilabel":
            return torch.sigmoid(logits)

        if self.cfg.num_classes == 2 and logits.ndim == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

    def _compute_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
        if self.cfg.task_type == "gastro_multilabel":
            return compute_multilabel_metrics(
                y_true=y_true,
                y_prob=y_prob,
                label_names=self.label_names,
                threshold=0.5,
            )

        if self.cfg.num_classes == 2 and y_prob.ndim == 1:
            return compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=0.5)

        return compute_multiclass_metrics(y_true=y_true, y_prob=y_prob, class_names=self.class_names)

    def _monitor_value(self, metrics: dict[str, Any]) -> float:
        v = metrics.get(self.cfg.monitor_metric, float("nan"))
        if isinstance(v, (int, float)):
            if math.isnan(v):
                return -float("inf") if self.cfg.monitor_mode == "max" else float("inf")
            return float(v)
        return -float("inf") if self.cfg.monitor_mode == "max" else float("inf")

    def _is_improved(self, val: float) -> bool:
        if self.cfg.monitor_mode == "max":
            return val > self.best_metric
        return val < self.best_metric

    def _save_checkpoint(self, path: Path, epoch: int, monitor: float) -> None:
        payload = {
            "epoch": epoch,
            "best_metric": self.best_metric,
            "monitor": monitor,
            "model_state": self._raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "cfg": self.cfg.__dict__,
        }
        torch.save(payload, path)

    def _load_checkpoint(self, path: Path, strict: bool = True) -> None:
        if not path.is_file():
            return
        ckpt = torch.load(path, map_location=self.device)
        self._raw_model.load_state_dict(ckpt["model_state"], strict=strict)
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state"])
        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.best_metric = float(ckpt.get("best_metric", self.best_metric))

    def _topk_paths_from_attention(
        self,
        attn: torch.Tensor,
        mask: torch.Tensor,
        image_paths: list[list[str]],
        topk: int,
    ) -> list[list[str]]:
        # 单头 attention: [B, N]
        outputs: list[list[str]] = []
        for i in range(attn.shape[0]):
            valid_n = int(mask[i].sum().item())
            k = min(topk, valid_n)
            if k <= 0:
                outputs.append([])
                continue
            idx = torch.topk(attn[i, :valid_n], k=k, dim=-1).indices.detach().cpu().tolist()
            paths = image_paths[i]
            outputs.append([paths[j] for j in idx if j < len(paths)])
        return outputs

    def _topk_paths_multilabel(
        self,
        attn: torch.Tensor,
        mask: torch.Tensor,
        image_paths: list[list[str]],
        topk: int,
    ) -> list[dict[str, list[str]]]:
        # 多头 attention: [B, L, N]
        rows: list[dict[str, list[str]]] = []
        for i in range(attn.shape[0]):
            valid_n = int(mask[i].sum().item())
            paths = image_paths[i]
            item: dict[str, list[str]] = {}
            for l_idx, label_name in enumerate(self.label_names):
                k = min(topk, valid_n)
                if k <= 0:
                    item[label_name] = []
                    continue
                idx = torch.topk(attn[i, l_idx, :valid_n], k=k, dim=-1).indices.detach().cpu().tolist()
                item[label_name] = [paths[j] for j in idx if j < len(paths)]
            rows.append(item)
        return rows

    def _run_one_epoch(
        self,
        loader,
        epoch: int,
        train_mode: bool,
        split: str,
        save_evidence: bool = False,
    ) -> tuple[float, dict[str, Any]]:
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_batches = 0

        y_true_list: list[np.ndarray] = []
        y_prob_list: list[np.ndarray] = []

        evidence_records: list[dict[str, Any]] = []
        expert_weights_all: list[np.ndarray] = []

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch_cpu in enumerate(self._iter_progress(loader, desc=f"{split}-epoch{epoch}"), start=1):
            batch = self._move_batch_to_device(batch_cpu)
            with torch.set_grad_enabled(train_mode):
                with autocast(enabled=self.cfg.amp and self.device.type == "cuda"):
                    outputs = self._forward(batch=batch, train_mode=train_mode)
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
            y_true_list.append(y_true)
            y_prob_list.append(y_prob)

            if "expert_weights" in outputs and torch.is_tensor(outputs["expert_weights"]):
                expert_weights_all.append(outputs["expert_weights"].detach().cpu().numpy())

            if save_evidence:
                mask_cpu = batch["mask"].detach().cpu()
                if self.cfg.task_type == "gastro_multilabel":
                    attn = outputs["attention"].detach().cpu()  # [B, L, N]
                    topk_map = self._topk_paths_multilabel(
                        attn=attn,
                        mask=mask_cpu,
                        image_paths=batch["image_paths"],
                        topk=self.cfg.topk_evidence,
                    )
                    for i in range(attn.shape[0]):
                        rec: dict[str, Any] = {
                            "exam_dir": batch["exam_dirs"][i],
                            "report_title": batch["report_titles"][i],
                            "topk_attention": topk_map[i],
                            "pred_prob": {
                                self.label_names[j]: float(y_prob[i, j])
                                for j in range(len(self.label_names))
                            },
                            "gt": {
                                self.label_names[j]: int(y_true[i, j])
                                for j in range(len(self.label_names))
                            },
                        }
                        if "expert_weights" in outputs and torch.is_tensor(outputs["expert_weights"]):
                            rec["expert_weights"] = outputs["expert_weights"][i].detach().cpu().tolist()
                        if "prototype_scores" in outputs and torch.is_tensor(outputs["prototype_scores"]):
                            rec["prototype_scores"] = outputs["prototype_scores"][i].detach().cpu().tolist()
                        evidence_records.append(rec)
                else:
                    attn = outputs["attention"].detach().cpu()  # [B, N]
                    topk_paths = self._topk_paths_from_attention(
                        attn=attn,
                        mask=mask_cpu,
                        image_paths=batch["image_paths"],
                        topk=self.cfg.topk_evidence,
                    )
                    for i in range(attn.shape[0]):
                        rec = {
                            "exam_dir": batch["exam_dirs"][i],
                            "report_title": batch["report_titles"][i],
                            "topk_attention": topk_paths[i],
                            "pred_prob": float(y_prob[i] if np.ndim(y_prob[i]) == 0 else y_prob[i, 1]),
                            "gt": int(y_true[i]),
                        }
                        if "prototype_similarity" in outputs and torch.is_tensor(outputs["prototype_similarity"]):
                            rec["prototype_similarity"] = outputs["prototype_similarity"][i].detach().cpu().tolist()
                        evidence_records.append(rec)

        if train_mode and total_batches % max(1, self.cfg.grad_accum_steps) != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

        avg_loss = total_loss / max(1, total_batches)

        y_true_all = np.concatenate(y_true_list, axis=0)
        y_prob_all = np.concatenate(y_prob_list, axis=0)
        metrics = self._compute_metrics(y_true=y_true_all, y_prob=y_prob_all)

        if expert_weights_all:
            ew = np.concatenate(expert_weights_all, axis=0)
            metrics["expert_usage_distribution"] = ew.mean(axis=0).tolist()

        if save_evidence:
            out_path = self.evidence_dir / f"{split}_evidence.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for rec in evidence_records:
                    f.write(json.dumps(to_builtin_type(rec), ensure_ascii=False) + "\n")

        return avg_loss, metrics

    def fit(self) -> dict[str, Any]:
        no_improve = 0

        for epoch in range(self.start_epoch, self.cfg.max_epochs + 1):
            train_loss, train_metrics = self._run_one_epoch(
                loader=self.train_loader,
                epoch=epoch,
                train_mode=True,
                split="train",
                save_evidence=False,
            )
            val_loss, val_metrics = self._run_one_epoch(
                loader=self.val_loader,
                epoch=epoch,
                train_mode=False,
                split="val",
                save_evidence=True,
            )

            train_monitor = self._monitor_value(train_metrics)
            val_monitor = self._monitor_value(val_metrics)

            self.logger.write(epoch, "train", train_loss, train_monitor, train_metrics)
            self.logger.write(epoch, "val", val_loss, val_monitor, val_metrics)

            self._save_checkpoint(self.last_path, epoch, val_monitor)

            if self._is_improved(val_monitor):
                self.best_metric = val_monitor
                self.best_epoch = epoch
                self._save_checkpoint(self.best_path, epoch, val_monitor)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.cfg.patience:
                break

        if self.best_path.is_file():
            self._load_checkpoint(self.best_path, strict=True)

        test_loss, test_metrics = self._run_one_epoch(
            loader=self.test_loader,
            epoch=self.best_epoch if self.best_epoch > 0 else self.cfg.max_epochs,
            train_mode=False,
            split="test",
            save_evidence=True,
        )

        result = {
            "best_epoch": self.best_epoch,
            "best_val_metric": self.best_metric,
            "test_loss": test_loss,
            "test_metrics": to_builtin_type(test_metrics),
        }

        (self.run_dir / "result_summary.json").write_text(
            json.dumps(to_builtin_type(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.run_dir / "test_metrics.json").write_text(
            json.dumps(to_builtin_type(test_metrics), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return result
