#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    auc,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    multilabel_confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Sampler

# 尽量避免在源码目录下产生 pyc 文件。
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from models import (
    DemoColoCountAwareDebiasMIL,
    DemoColoMILBaseline,
    DemoGastroMILBaseline,
    DemoGastroProtoMoEFormer,
)
from models.demo_data import (
    COLO_BINARY_CLASS_NAMES,
    GASTRO_LABEL_NAMES,
    DemoMILBagDataset,
    build_task_records,
    demo_mil_collate_fn,
    split_records,
)
from models.demo_metrics import to_builtin_type
from models.demo_trainer import DemoTrainer, TrainerConfig

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


MODEL_KEYS = (
    "demo_gastro_mil_baseline",
    "demo_gastro_proto_moe_former",
    "demo_colo_mil_baseline",
    "demo_colo_count_aware_debias_mil",
)


class RichDemoTrainer(DemoTrainer):
    """仅在 demo.py 内扩展训练输出，不修改底层 trainer 文件。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_csv_path = self.run_dir / "loss_history.csv"
        self.epoch_metrics_csv_path = self.run_dir / "epoch_metrics.csv"
        self.val_metrics_latest_path = self.run_dir / "val_metrics_latest.json"
        self.best_val_metrics_path = self.run_dir / "best_val_metrics.json"
        self.test_metrics_path = self.run_dir / "test_metrics.json"
        self.test_metrics_meta_path = self.run_dir / "test_metrics_meta.json"
        self.loss_curve_path = self.run_dir / "loss_curve.png"
        self.val_cm_path = self.run_dir / "val_confusion_matrix.png"
        self.test_cm_path = self.run_dir / "test_confusion_matrix.png"
        self.val_roc_path = self.run_dir / "val_roc_curve.png"
        self.test_roc_path = self.run_dir / "test_roc_curve.png"
        self.val_pr_path = self.run_dir / "val_pr_curve.png"
        self.test_pr_path = self.run_dir / "test_pr_curve.png"

        self.loss_history: list[dict[str, Any]] = []
        self.metrics_history: list[dict[str, Any]] = []

    def _iter_progress(self, loader, desc: str):
        if tqdm is not None:
            return tqdm(loader, desc=desc, leave=True, dynamic_ncols=True)
        return loader

    @staticmethod
    def _nanmean(values: list[float]) -> float:
        if not values:
            return float("nan")
        arr = np.asarray(values, dtype=np.float64)
        if np.isnan(arr).all():
            return float("nan")
        return float(np.nanmean(arr))

    @staticmethod
    def _safe_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        merged = np.concatenate([y_true.reshape(-1), y_pred.reshape(-1)], axis=0)
        if len(np.unique(merged)) < 2:
            return 1.0 if np.array_equal(y_true, y_pred) else 0.0
        score = float(cohen_kappa_score(y_true, y_pred))
        return 0.0 if math.isnan(score) else score

    @staticmethod
    def _fmt_metric(value: Any) -> str:
        if isinstance(value, (int, float)):
            if isinstance(value, float) and math.isnan(value):
                return "nan"
            return f"{float(value):.4f}"
        return str(value)

    def _summary_text(self, metrics: dict[str, Any]) -> str:
        ordered_keys = ["ACC", "Recall", "Precision", "F1", "ROC_AUC", "PR_AUC", "Kappa"]
        parts = [f"{k}={self._fmt_metric(metrics.get(k, float('nan')))}" for k in ordered_keys]
        return ", ".join(parts)

    def _binary_artifacts(self, y_true: np.ndarray, y_prob: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
        y_true = y_true.astype(np.int64).reshape(-1)
        y_prob = y_prob.astype(np.float64).reshape(-1)
        y_pred = (y_prob >= 0.5).astype(np.int64)

        acc = float(accuracy_score(y_true, y_pred))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        kappa = self._safe_kappa(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        roc_auc = float("nan")
        roc_payload: dict[str, Any] | None = None
        if len(np.unique(y_true)) >= 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = float(auc(fpr, tpr))
            roc_payload = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}

        pr_auc = float("nan")
        pr_payload: dict[str, Any] | None = None
        if int(y_true.sum()) > 0:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(recall_curve, precision_curve))
            pr_payload = {
                "precision": precision_curve.tolist(),
                "recall": recall_curve.tolist(),
                "auc": pr_auc,
            }

        metrics = {
            "ACC": acc,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc,
            "Kappa": kappa,
            "accuracy": acc,
            "recall": recall,
            "precision": precision,
            "f1_score": f1,
            "kappa": kappa,
        }
        metrics.update(self._compute_metrics(y_true=y_true, y_prob=y_prob))

        artifact = {
            "mode": "binary",
            "class_names": self.class_names if self.class_names else ["negative", "positive"],
            "confusion_matrix": cm.tolist(),
            "roc_curve": roc_payload,
            "pr_curve": pr_payload,
        }
        return metrics, artifact

    def _multilabel_artifacts(self, y_true: np.ndarray, y_prob: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
        y_true = y_true.astype(np.int64)
        y_prob = y_prob.astype(np.float64)
        y_pred = (y_prob >= 0.5).astype(np.int64)

        acc = float(accuracy_score(y_true, y_pred))
        recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        per_label_precision: dict[str, float] = {}
        per_label_recall: dict[str, float] = {}
        per_label_kappa: dict[str, float] = {}
        roc_aucs: list[float] = []
        pr_aucs: list[float] = []
        kappas: list[float] = []
        roc_curves: list[dict[str, Any]] = []
        pr_curves: list[dict[str, Any]] = []

        for idx, label_name in enumerate(self.label_names):
            yt = y_true[:, idx]
            yp = y_prob[:, idx]
            yhat = y_pred[:, idx]

            label_precision = float(precision_score(yt, yhat, zero_division=0))
            label_recall = float(recall_score(yt, yhat, zero_division=0))
            label_kappa = self._safe_kappa(yt, yhat)

            per_label_precision[label_name] = label_precision
            per_label_recall[label_name] = label_recall
            per_label_kappa[label_name] = label_kappa
            kappas.append(label_kappa)

            if len(np.unique(yt)) >= 2:
                fpr, tpr, _ = roc_curve(yt, yp)
                roc_auc = float(auc(fpr, tpr))
            else:
                fpr, tpr, roc_auc = np.array([]), np.array([]), float("nan")
            roc_aucs.append(roc_auc)
            roc_curves.append(
                {
                    "label": label_name,
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": roc_auc,
                }
            )

            if int(yt.sum()) > 0:
                precision_curve, recall_curve, _ = precision_recall_curve(yt, yp)
                pr_auc = float(auc(recall_curve, precision_curve))
            else:
                precision_curve, recall_curve, pr_auc = np.array([]), np.array([]), float("nan")
            pr_aucs.append(pr_auc)
            pr_curves.append(
                {
                    "label": label_name,
                    "precision": precision_curve.tolist(),
                    "recall": recall_curve.tolist(),
                    "auc": pr_auc,
                }
            )

        metrics = {
            "ACC": acc,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "ROC_AUC": self._nanmean(roc_aucs),
            "PR_AUC": self._nanmean(pr_aucs),
            "Kappa": self._nanmean(kappas),
            "accuracy": acc,
            "recall_macro": recall,
            "precision_macro": precision,
            "f1_macro_required": f1,
            "kappa_macro": self._nanmean(kappas),
            "per_label_precision": per_label_precision,
            "per_label_recall": per_label_recall,
            "per_label_kappa": per_label_kappa,
            "metric_note": "多标签任务中 ACC 为 exact-match accuracy；Recall/Precision/F1/Kappa 为 macro 平均。",
        }
        metrics.update(self._compute_metrics(y_true=y_true, y_prob=y_prob))

        artifact = {
            "mode": "multilabel",
            "label_names": self.label_names,
            "confusion_matrices": [cm.tolist() for cm in multilabel_confusion_matrix(y_true, y_pred)],
            "roc_curves": roc_curves,
            "pr_curves": pr_curves,
        }
        return metrics, artifact

    def _compute_required_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.cfg.task_type == "gastro_multilabel":
            return self._multilabel_artifacts(y_true=y_true, y_prob=y_prob)
        return self._binary_artifacts(y_true=y_true, y_prob=y_prob)

    @staticmethod
    def _draw_heatmap(ax, cm: np.ndarray, labels: list[str], title: str) -> None:
        image = ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = int(cm[i, j])
                ax.text(j, i, str(value), ha="center", va="center", color="black")
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    def _save_confusion_matrix_plot(self, split: str, epoch: int, artifact: dict[str, Any]) -> None:
        path = self.val_cm_path if split == "val" else self.test_cm_path
        mode = artifact.get("mode")

        if mode == "binary":
            cm = np.asarray(artifact["confusion_matrix"], dtype=np.int64)
            labels = list(artifact.get("class_names", ["negative", "positive"]))
            fig, ax = plt.subplots(figsize=(5, 4))
            self._draw_heatmap(ax, cm=cm, labels=labels, title=f"{split} confusion matrix | epoch {epoch}")
        else:
            cms = artifact.get("confusion_matrices", [])
            labels = list(artifact.get("label_names", self.label_names))
            cols = max(1, len(cms))
            fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 4))
            if cols == 1:
                axes = [axes]
            for ax, cm, label_name in zip(axes, cms, labels):
                self._draw_heatmap(
                    ax,
                    cm=np.asarray(cm, dtype=np.int64),
                    labels=["0", "1"],
                    title=f"{label_name} | epoch {epoch}",
                )
            fig.suptitle(f"{split} confusion matrix", fontsize=12)

        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _save_curve_plot(self, split: str, epoch: int, artifact: dict[str, Any], curve_type: str) -> None:
        is_roc = curve_type == "roc"
        path = self.val_roc_path if split == "val" and is_roc else self.test_roc_path if is_roc else self.val_pr_path if split == "val" else self.test_pr_path
        mode = artifact.get("mode")
        fig, ax = plt.subplots(figsize=(6, 5))

        if mode == "binary":
            curve_payload = artifact.get("roc_curve" if is_roc else "pr_curve")
            if curve_payload:
                if is_roc:
                    ax.plot(curve_payload["fpr"], curve_payload["tpr"], label=f"AUC={curve_payload['auc']:.4f}")
                    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)
                    ax.set_xlabel("FPR")
                    ax.set_ylabel("TPR")
                    ax.set_title(f"{split} ROC curve | epoch {epoch}")
                else:
                    ax.plot(curve_payload["recall"], curve_payload["precision"], label=f"AUC={curve_payload['auc']:.4f}")
                    ax.set_xlabel("Recall")
                    ax.set_ylabel("Precision")
                    ax.set_title(f"{split} PR curve | epoch {epoch}")
                ax.legend(loc="best")
            else:
                ax.text(0.5, 0.5, "Curve unavailable", ha="center", va="center")
                ax.set_axis_off()
        else:
            curve_list = artifact.get("roc_curves" if is_roc else "pr_curves", [])
            valid_aucs: list[float] = []
            for item in curve_list:
                curve_auc = float(item.get("auc", float("nan")))
                if math.isnan(curve_auc):
                    continue
                valid_aucs.append(curve_auc)
                if is_roc:
                    ax.plot(item["fpr"], item["tpr"], label=f"{item['label']} | AUC={curve_auc:.4f}")
                else:
                    ax.plot(item["recall"], item["precision"], label=f"{item['label']} | AUC={curve_auc:.4f}")
            if valid_aucs:
                macro_auc = self._nanmean(valid_aucs)
                if is_roc:
                    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)
                    ax.set_xlabel("FPR")
                    ax.set_ylabel("TPR")
                    ax.set_title(f"{split} ROC curve | epoch {epoch} | macro AUC={macro_auc:.4f}")
                else:
                    ax.set_xlabel("Recall")
                    ax.set_ylabel("Precision")
                    ax.set_title(f"{split} PR curve | epoch {epoch} | macro AUC={macro_auc:.4f}")
                ax.legend(loc="best")
            else:
                ax.text(0.5, 0.5, "Curve unavailable", ha="center", va="center")
                ax.set_axis_off()

        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _write_loss_history_csv(self) -> None:
        fieldnames = ["epoch", "train_loss", "val_loss"]
        with self.loss_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.loss_history:
                writer.writerow(row)

    def _write_epoch_metrics_csv(self) -> None:
        fieldnames = [
            "epoch",
            "split",
            "loss",
            "monitor",
            "ACC",
            "Recall",
            "Precision",
            "F1",
            "ROC_AUC",
            "PR_AUC",
            "Kappa",
        ]
        with self.epoch_metrics_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.metrics_history:
                writer.writerow(row)

    def _save_loss_curve(self) -> None:
        if not self.loss_history:
            return
        epochs = [int(row["epoch"]) for row in self.loss_history]
        train_losses = [float(row["train_loss"]) for row in self.loss_history]
        val_losses = [float(row["val_loss"]) for row in self.loss_history]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(epochs, train_losses, marker="o", label="train_loss")
        ax.plot(epochs, val_losses, marker="o", label="val_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss convergence")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.loss_curve_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _save_metrics_snapshot(self, path: Path, epoch: int, loss: float, metrics: dict[str, Any]) -> None:
        payload = {
            "epoch": epoch,
            "loss": loss,
            "metrics": to_builtin_type(metrics),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _record_epoch_outputs(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_metrics: dict[str, Any],
        val_metrics: dict[str, Any],
    ) -> None:
        self.loss_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        train_monitor = self._monitor_value(train_metrics)
        val_monitor = self._monitor_value(val_metrics)
        self.metrics_history.append(
            {
                "epoch": epoch,
                "split": "train",
                "loss": train_loss,
                "monitor": train_monitor,
                "ACC": train_metrics.get("ACC", float("nan")),
                "Recall": train_metrics.get("Recall", float("nan")),
                "Precision": train_metrics.get("Precision", float("nan")),
                "F1": train_metrics.get("F1", float("nan")),
                "ROC_AUC": train_metrics.get("ROC_AUC", float("nan")),
                "PR_AUC": train_metrics.get("PR_AUC", float("nan")),
                "Kappa": train_metrics.get("Kappa", float("nan")),
            }
        )
        self.metrics_history.append(
            {
                "epoch": epoch,
                "split": "val",
                "loss": val_loss,
                "monitor": val_monitor,
                "ACC": val_metrics.get("ACC", float("nan")),
                "Recall": val_metrics.get("Recall", float("nan")),
                "Precision": val_metrics.get("Precision", float("nan")),
                "F1": val_metrics.get("F1", float("nan")),
                "ROC_AUC": val_metrics.get("ROC_AUC", float("nan")),
                "PR_AUC": val_metrics.get("PR_AUC", float("nan")),
                "Kappa": val_metrics.get("Kappa", float("nan")),
            }
        )
        self._write_loss_history_csv()
        self._write_epoch_metrics_csv()
        self._save_loss_curve()

    def _run_one_epoch_detailed(
        self,
        loader,
        epoch: int,
        train_mode: bool,
        split: str,
        save_evidence: bool = False,
    ) -> tuple[float, dict[str, Any], dict[str, Any]]:
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
        progress = self._iter_progress(loader, desc=f"{split}-epoch{epoch}")

        for step, batch_cpu in enumerate(progress, start=1):
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

            if hasattr(progress, "set_postfix"):
                progress.set_postfix(loss=f"{float(loss.detach().cpu().item()):.4f}")

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
                    attn = outputs["attention"].detach().cpu()
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
                    attn = outputs["attention"].detach().cpu()
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

        if total_batches == 0:
            return float("nan"), {self.cfg.monitor_metric: float("nan")}, {}

        avg_loss = total_loss / max(1, total_batches)
        y_true_all = np.concatenate(y_true_list, axis=0)
        y_prob_all = np.concatenate(y_prob_list, axis=0)
        metrics, artifact = self._compute_required_metrics(y_true=y_true_all, y_prob=y_prob_all)

        if expert_weights_all:
            ew = np.concatenate(expert_weights_all, axis=0)
            metrics["expert_usage_distribution"] = ew.mean(axis=0).tolist()

        if save_evidence:
            out_path = self.evidence_dir / f"{split}_evidence.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for rec in evidence_records:
                    f.write(json.dumps(to_builtin_type(rec), ensure_ascii=False) + "\n")

        return avg_loss, metrics, artifact

    def fit(self) -> dict[str, Any]:
        best_val_metrics: dict[str, Any] | None = None
        best_val_loss = float("nan")

        for epoch in range(self.start_epoch, self.cfg.max_epochs + 1):
            print("\n" + "-" * 80)
            print(f"[{self.run_dir.name}] Epoch {epoch}/{self.cfg.max_epochs} 训练")
            train_loss, train_metrics, _ = self._run_one_epoch_detailed(
                loader=self.train_loader,
                epoch=epoch,
                train_mode=True,
                split="train",
                save_evidence=False,
            )
            print(f"[{self.run_dir.name}] 训练完成: loss={self._fmt_metric(train_loss)}, {self._summary_text(train_metrics)}")

            print(f"[{self.run_dir.name}] Epoch {epoch}/{self.cfg.max_epochs} 验证")
            val_loss, val_metrics, val_artifact = self._run_one_epoch_detailed(
                loader=self.val_loader,
                epoch=epoch,
                train_mode=False,
                split="val",
                save_evidence=True,
            )
            print(f"[{self.run_dir.name}] 验证完成: loss={self._fmt_metric(val_loss)}, {self._summary_text(val_metrics)}")

            train_monitor = self._monitor_value(train_metrics)
            val_monitor = self._monitor_value(val_metrics)
            self.logger.write(epoch, "train", train_loss, train_monitor, train_metrics)
            self.logger.write(epoch, "val", val_loss, val_monitor, val_metrics)
            self._record_epoch_outputs(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )
            self._save_metrics_snapshot(self.val_metrics_latest_path, epoch=epoch, loss=val_loss, metrics=val_metrics)
            self._save_confusion_matrix_plot(split="val", epoch=epoch, artifact=val_artifact)
            self._save_curve_plot(split="val", epoch=epoch, artifact=val_artifact, curve_type="roc")
            self._save_curve_plot(split="val", epoch=epoch, artifact=val_artifact, curve_type="pr")

            self._save_checkpoint(self.last_path, epoch, val_monitor)
            if self._is_improved(val_monitor):
                self.best_metric = val_monitor
                self.best_epoch = epoch
                best_val_metrics = dict(val_metrics)
                best_val_loss = val_loss
                self._save_checkpoint(self.best_path, epoch, val_monitor)
                self._save_metrics_snapshot(self.best_val_metrics_path, epoch=epoch, loss=val_loss, metrics=val_metrics)

        if self.best_path.is_file():
            self._load_checkpoint(self.best_path, strict=True)

        test_epoch = self.best_epoch if self.best_epoch > 0 else self.cfg.max_epochs
        print("\n" + "-" * 80)
        print(f"[{self.run_dir.name}] 测试（best epoch = {test_epoch}）")
        test_loss, test_metrics, test_artifact = self._run_one_epoch_detailed(
            loader=self.test_loader,
            epoch=test_epoch,
            train_mode=False,
            split="test",
            save_evidence=True,
        )
        print(f"[{self.run_dir.name}] 测试完成: loss={self._fmt_metric(test_loss)}, {self._summary_text(test_metrics)}")

        self.test_metrics_path.write_text(
            json.dumps(to_builtin_type(test_metrics), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._save_metrics_snapshot(self.test_metrics_meta_path, epoch=test_epoch, loss=test_loss, metrics=test_metrics)
        self._save_confusion_matrix_plot(split="test", epoch=test_epoch, artifact=test_artifact)
        self._save_curve_plot(split="test", epoch=test_epoch, artifact=test_artifact, curve_type="roc")
        self._save_curve_plot(split="test", epoch=test_epoch, artifact=test_artifact, curve_type="pr")

        result = {
            "best_epoch": self.best_epoch,
            "best_val_metric": self.best_metric,
            "best_val_loss": best_val_loss,
            "best_val_metrics": to_builtin_type(best_val_metrics or {}),
            "test_loss": test_loss,
            "test_metrics": to_builtin_type(test_metrics),
        }

        (self.run_dir / "result_summary.json").write_text(
            json.dumps(to_builtin_type(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result


class InstanceAwareBatchSampler(Sampler[list[int]]):
    """按实例总量限制批次，避免检查目录图像数差异大导致内存峰值。"""

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
        self._iter_count = 0

        self.instance_counts: list[int] = []
        for record in self.records:
            n = len(record.get("image_paths", []))
            n = max(1, n)
            n = min(n, self.max_instances_per_bag)
            n = max(n, self.min_instances_per_bag)
            self.instance_counts.append(n)

    def __iter__(self):
        indices = list(range(len(self.records)))
        if self.shuffle:
            rng = random.Random(self.seed + self._iter_count)
            rng.shuffle(indices)
        self._iter_count += 1

        batch: list[int] = []
        batch_instances = 0

        for idx in indices:
            n_inst = self.instance_counts[idx]

            need_flush = False
            if len(batch) >= self.batch_size:
                need_flush = True
            elif batch and (batch_instances + n_inst > self.max_instances_per_batch):
                need_flush = True

            if need_flush:
                yield batch
                batch = []
                batch_instances = 0

            batch.append(idx)
            batch_instances += n_inst

        if batch:
            yield batch

    def __len__(self) -> int:
        if not self.records:
            return 0
        return int(math.ceil(len(self.records) / float(self.batch_size)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键运行 4 个 MIL 模型 demo")
    parser.add_argument("--config", type=str, default="configs/path.yaml", help="路径配置文件")
    parser.add_argument("--demo-config", type=str, default="configs/demo.yaml", help="demo 运行参数配置")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--epochs", type=int, default=30, help="每个模型训练轮数")
    parser.add_argument("--patience", type=int, default=30, help="保留参数兼容；默认与训练轮数一致")
    parser.add_argument("--image-size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--num-workers", type=int, default=-1, help="覆盖 demo.yaml 中的 num_workers；-1 表示不覆盖")
    parser.add_argument("--max-exams-per-task", type=int, default=0, help="每个任务最多样本数，0 表示不限制")
    parser.add_argument("--no-pretrained", action="store_true", help="禁用 ImageNet 预训练")
    parser.add_argument("--disable-multi-gpu", action="store_true", help="禁用 DataParallel")
    return parser.parse_args()


def resolve_config_path(raw_path: Any, config_path: Path) -> str:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def load_path_config(config_path: Path) -> dict[str, str]:
    if not config_path.is_file():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "paths" not in payload:
        raise ValueError("配置文件缺少 paths 字段")
    paths = payload["paths"]
    if not isinstance(paths, dict):
        raise ValueError("paths 字段格式错误")

    required = ["valid_dicts_report_csv", "output_dir"]
    for k in required:
        if k not in paths:
            raise ValueError(f"paths 缺少字段: {k}")

    resolved = {
        "valid_dicts_report_csv": resolve_config_path(paths["valid_dicts_report_csv"], config_path),
        "output_dir": resolve_config_path(paths["output_dir"], config_path),
    }
    if "dataset_root" in paths and str(paths["dataset_root"]).strip():
        resolved["dataset_root"] = resolve_config_path(paths["dataset_root"], config_path)
    return resolved


def _load_per_model_int_map(payload: dict[str, Any], key: str, default: int) -> dict[str, int]:
    raw = payload.get(key, {})
    if isinstance(raw, int):
        return {k: int(raw) for k in MODEL_KEYS}
    if not isinstance(raw, dict):
        raise ValueError(f"{key} 必须是整数或字典")

    out: dict[str, int] = {}
    for k in MODEL_KEYS:
        value = int(raw.get(k, default))
        if value <= 0:
            raise ValueError(f"{key}.{k} 必须 > 0")
        out[k] = value
    return out


def _load_per_model_float_map(payload: dict[str, Any], key: str, default: float) -> dict[str, float]:
    raw = payload.get(key, {})
    if isinstance(raw, (int, float)):
        v = float(raw)
        return {k: v for k in MODEL_KEYS}
    if not isinstance(raw, dict):
        raise ValueError(f"{key} 必须是数字或字典")

    out: dict[str, float] = {}
    for k in MODEL_KEYS:
        value = float(raw.get(k, default))
        if value < 0.0:
            raise ValueError(f"{key}.{k} 必须 >= 0")
        out[k] = value
    return out


def load_demo_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"未找到 demo 配置文件: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("demo 配置文件格式错误")

    gpu_ids_raw = payload.get("gpu_ids", [0, 1, 2])
    if not isinstance(gpu_ids_raw, list) or not gpu_ids_raw:
        raise ValueError("gpu_ids 必须是非空列表")
    gpu_ids = [int(x) for x in gpu_ids_raw]

    num_workers = int(payload.get("num_workers", 6))
    if num_workers < 0:
        raise ValueError("num_workers 必须 >= 0")

    batch_size = _load_per_model_int_map(payload, "batch_size", default=3)
    eval_batch_size = _load_per_model_int_map(payload, "eval_batch_size", default=3)
    train_max_instances = _load_per_model_int_map(payload, "train_max_instances", default=24)
    val_max_instances = _load_per_model_int_map(payload, "val_max_instances", default=24)
    test_max_instances = _load_per_model_int_map(payload, "test_max_instances", default=24)
    train_max_batch_instances = _load_per_model_int_map(payload, "train_max_batch_instances", default=72)
    eval_max_batch_instances = _load_per_model_int_map(payload, "eval_max_batch_instances", default=72)
    random_instance_dropout = _load_per_model_float_map(payload, "random_instance_dropout", default=0.05)

    min_instances = int(payload.get("min_instances", 1))
    if min_instances <= 0:
        raise ValueError("min_instances 必须 > 0")

    train_sampling = str(payload.get("train_sampling_strategy", "random"))
    eval_sampling = str(payload.get("eval_sampling_strategy", "uniform"))

    return {
        "gpu_ids": gpu_ids,
        "num_workers": num_workers,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "train_max_instances": train_max_instances,
        "val_max_instances": val_max_instances,
        "test_max_instances": test_max_instances,
        "train_max_batch_instances": train_max_batch_instances,
        "eval_max_batch_instances": eval_max_batch_instances,
        "random_instance_dropout": random_instance_dropout,
        "min_instances": min_instances,
        "train_sampling_strategy": train_sampling,
        "eval_sampling_strategy": eval_sampling,
    }


def maybe_limit_records(records: list[dict[str, Any]], max_num: int, seed: int) -> list[dict[str, Any]]:
    if max_num <= 0 or len(records) <= max_num:
        return records
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    keep = idx[:max_num]
    return [records[int(i)] for i in keep]


def build_compatible_split(
    records: list[dict[str, Any]],
    seed: int,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    """优先使用常规比例切分；当样本极少时回退到可训练切分。"""
    if len(records) == 0:
        return {"train": [], "val": [], "test": []}, "empty"

    regular_split = split_records(records, seed=seed, ratios=ratios)
    if all(len(regular_split[k]) > 0 for k in ("train", "val", "test")):
        return regular_split, "ratio_split"

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n >= 3:
        # 至少保证每个 split 有 1 个样本。
        fallback_split = {
            "train": shuffled[:-2],
            "val": [shuffled[-2]],
            "test": [shuffled[-1]],
        }
    elif n == 2:
        # 样本过少时允许 test 复用 train，保证流程可跑通。
        fallback_split = {
            "train": [shuffled[0]],
            "val": [shuffled[1]],
            "test": [shuffled[0]],
        }
    else:
        fallback_split = {
            "train": [shuffled[0]],
            "val": [shuffled[0]],
            "test": [shuffled[0]],
        }

    return fallback_split, "small_sample_fallback"


def compute_multilabel_pos_weight(train_records: list[dict[str, Any]]) -> list[float]:
    y = np.array([r["labels"] for r in train_records], dtype=np.float32)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    pw = (neg + 1.0) / (pos + 1.0)
    return pw.astype(np.float32).tolist()


def compute_binary_pos_weight(train_records: list[dict[str, Any]]) -> list[float]:
    y = np.array([r["label"] for r in train_records], dtype=np.int64)
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    pw = (neg + 1.0) / (pos + 1.0)
    return [float(pw)]


def ceil_to_multiple(value: int, divisor: int) -> int:
    if divisor <= 0:
        return value
    return ((value + divisor - 1) // divisor) * divisor


def normalize_batch_size(value: int, active_gpu_count: int) -> int:
    v = max(1, int(value))
    if active_gpu_count <= 1:
        return v
    return ceil_to_multiple(max(v, active_gpu_count), active_gpu_count)


def build_loaders(
    split_data: dict[str, list[dict[str, Any]]],
    task_name: str,
    image_size: int,
    num_workers: int,
    train_batch_size: int,
    eval_batch_size: int,
    train_max_instances: int,
    val_max_instances: int,
    test_max_instances: int,
    min_instances: int,
    train_sampling: str,
    eval_sampling: str,
    random_instance_dropout: float,
    train_max_batch_instances: int,
    eval_max_batch_instances: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    pin_memory = torch.cuda.is_available()

    train_ds = DemoMILBagDataset(
        records=split_data["train"],
        task=task_name,
        max_instances=train_max_instances,
        min_instances=min_instances,
        bag_sampling_strategy=train_sampling,
        is_train=True,
        image_size=image_size,
        random_instance_dropout=random_instance_dropout,
    )
    val_ds = DemoMILBagDataset(
        records=split_data["val"],
        task=task_name,
        max_instances=val_max_instances,
        min_instances=min_instances,
        bag_sampling_strategy=eval_sampling,
        is_train=False,
        image_size=image_size,
        random_instance_dropout=0.0,
    )
    test_ds = DemoMILBagDataset(
        records=split_data["test"],
        task=task_name,
        max_instances=test_max_instances,
        min_instances=min_instances,
        bag_sampling_strategy=eval_sampling,
        is_train=False,
        image_size=image_size,
        random_instance_dropout=0.0,
    )

    train_sampler = InstanceAwareBatchSampler(
        records=split_data["train"],
        max_instances_per_bag=train_max_instances,
        min_instances_per_bag=min_instances,
        batch_size=train_batch_size,
        max_instances_per_batch=train_max_batch_instances,
        shuffle=True,
        seed=seed,
    )
    val_sampler = InstanceAwareBatchSampler(
        records=split_data["val"],
        max_instances_per_bag=val_max_instances,
        min_instances_per_bag=min_instances,
        batch_size=eval_batch_size,
        max_instances_per_batch=eval_max_batch_instances,
        shuffle=False,
        seed=seed + 1,
    )
    test_sampler = InstanceAwareBatchSampler(
        records=split_data["test"],
        max_instances_per_bag=test_max_instances,
        min_instances_per_bag=min_instances,
        batch_size=eval_batch_size,
        max_instances_per_batch=eval_max_batch_instances,
        shuffle=False,
        seed=seed + 2,
    )

    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": demo_mil_collate_fn,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_sampler=test_sampler, **loader_kwargs)
    return train_loader, val_loader, test_loader


def run_single_model(
    model_name: str,
    model,
    trainer_cfg: TrainerConfig,
    split_data: dict[str, list[dict[str, Any]]],
    task_name: str,
    image_size: int,
    num_workers: int,
    run_dir: Path,
    seed: int,
    dl_cfg: dict[str, Any],
    label_names: list[str],
    class_names: list[str],
) -> dict[str, Any]:
    train_loader, val_loader, test_loader = build_loaders(
        split_data=split_data,
        task_name=task_name,
        image_size=image_size,
        num_workers=num_workers,
        train_batch_size=dl_cfg["train_batch_size"],
        eval_batch_size=dl_cfg["eval_batch_size"],
        train_max_instances=dl_cfg["train_max_instances"],
        val_max_instances=dl_cfg["val_max_instances"],
        test_max_instances=dl_cfg["test_max_instances"],
        min_instances=dl_cfg["min_instances"],
        train_sampling=dl_cfg["train_sampling"],
        eval_sampling=dl_cfg["eval_sampling"],
        random_instance_dropout=dl_cfg["random_instance_dropout"],
        train_max_batch_instances=dl_cfg["train_max_batch_instances"],
        eval_max_batch_instances=dl_cfg["eval_max_batch_instances"],
        seed=seed,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "trainer_cfg": asdict(trainer_cfg),
                "dataloader_cfg": dl_cfg,
                "split_stats": {k: len(v) for k, v in split_data.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    trainer = RichDemoTrainer(
        model=model,
        cfg=trainer_cfg,
        run_dir=run_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        label_names=label_names,
        class_names=class_names,
        seed=seed,
    )
    result = trainer.fit()
    return result


def main() -> None:
    args = parse_args()

    path_cfg = load_path_config(Path(args.config))
    demo_cfg = load_demo_config(Path(args.demo_config))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in demo_cfg["gpu_ids"])

    torch.set_float32_matmul_precision("medium")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_multi_gpu = (not args.disable_multi_gpu) and visible_gpu_count > 1
    active_gpu_count = visible_gpu_count if use_multi_gpu else (1 if visible_gpu_count > 0 else 0)

    cfg_workers = int(demo_cfg["num_workers"])
    requested_workers = cfg_workers if args.num_workers < 0 else int(args.num_workers)
    cpu_cap = max(1, (os.cpu_count() or 8) - 2)
    effective_workers = max(0, min(requested_workers, cpu_cap))

    report_csv = Path(path_cfg["valid_dicts_report_csv"])
    output_root = Path(path_cfg["output_dir"]) / "demo"
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("开始构建 demo 数据")
    print(f"报告 CSV: {report_csv}")
    print(f"输出目录: {output_root}")
    print(
        "硬件配置: "
        f"visible_gpu_count={visible_gpu_count}, "
        f"active_gpu_count={active_gpu_count}, "
        f"num_workers={effective_workers}, "
        f"gpu_ids={demo_cfg['gpu_ids']}"
    )
    print("=" * 80)

    gastro_records, colo_records = build_task_records(
        report_csv_path=report_csv,
        min_instances=demo_cfg["min_instances"],
        dataset_root=path_cfg.get("dataset_root"),
    )
    gastro_records = maybe_limit_records(gastro_records, args.max_exams_per_task, args.seed)
    colo_records = maybe_limit_records(colo_records, args.max_exams_per_task, args.seed)

    if len(gastro_records) == 0:
        raise RuntimeError("胃镜可用样本为 0，无法训练")
    if len(colo_records) == 0:
        raise RuntimeError("肠镜可用样本为 0，无法训练")

    gastro_split, gastro_split_mode = build_compatible_split(gastro_records, seed=args.seed, ratios=(0.6, 0.2, 0.2))
    colo_split, colo_split_mode = build_compatible_split(colo_records, seed=args.seed + 1, ratios=(0.6, 0.2, 0.2))

    if gastro_split_mode != "ratio_split":
        print(f"[提示] 胃镜样本较少，启用小样本兼容切分: mode={gastro_split_mode}")
    if colo_split_mode != "ratio_split":
        print(f"[提示] 肠镜样本较少，启用小样本兼容切分: mode={colo_split_mode}")

    print(f"胃镜样本数: train={len(gastro_split['train'])}, val={len(gastro_split['val'])}, test={len(gastro_split['test'])}")
    print(f"肠镜样本数: train={len(colo_split['train'])}, val={len(colo_split['val'])}, test={len(colo_split['test'])}")

    session_dir = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)

    pretrained = not args.no_pretrained

    gastro_pos_weight = compute_multilabel_pos_weight(gastro_split["train"])
    colo_pos_weight = compute_binary_pos_weight(colo_split["train"])

    all_results: dict[str, Any] = {
        "session_dir": str(session_dir),
        "gastro_split": {k: len(v) for k, v in gastro_split.items()},
        "colo_split": {k: len(v) for k, v in colo_split.items()},
        "gastro_split_mode": gastro_split_mode,
        "colo_split_mode": colo_split_mode,
        "models": {},
    }

    # 1) 胃镜 baseline
    key_1 = "demo_gastro_mil_baseline"
    print("\n[1/4] 训练 demo_gastro_mil_baseline")
    model_1 = DemoGastroMILBaseline(
        backbone_name="resnet50",
        pretrained=pretrained,
        freeze_stages=1,
        feature_dim=512,
        attn_dim=256,
        num_labels=3,
        dropout=0.2,
    )
    trainer_cfg_1 = TrainerConfig(
        task_type="gastro_multilabel",
        model_family="gastro_baseline",
        num_classes=3,
        num_labels=3,
        max_epochs=args.epochs,
        patience=args.patience,
        lr=2e-4,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        grad_accum_steps=1 if use_multi_gpu else 2,
        amp=True,
        monitor_metric="macro_auc",
        monitor_mode="max",
        topk_evidence=5,
        loss_name="asymmetric",
        pos_weight=gastro_pos_weight,
        aux_loss_weights={},
        use_multi_gpu=use_multi_gpu,
        resume_path=None,
    )
    dl_cfg_1 = {
        "train_batch_size": normalize_batch_size(demo_cfg["batch_size"][key_1], active_gpu_count),
        "eval_batch_size": normalize_batch_size(demo_cfg["eval_batch_size"][key_1], active_gpu_count),
        "train_max_instances": demo_cfg["train_max_instances"][key_1],
        "val_max_instances": demo_cfg["val_max_instances"][key_1],
        "test_max_instances": demo_cfg["test_max_instances"][key_1],
        "train_max_batch_instances": demo_cfg["train_max_batch_instances"][key_1],
        "eval_max_batch_instances": demo_cfg["eval_max_batch_instances"][key_1],
        "min_instances": demo_cfg["min_instances"],
        "train_sampling": demo_cfg["train_sampling_strategy"],
        "eval_sampling": demo_cfg["eval_sampling_strategy"],
        "random_instance_dropout": demo_cfg["random_instance_dropout"][key_1],
    }
    result_1 = run_single_model(
        model_name=key_1,
        model=model_1,
        trainer_cfg=trainer_cfg_1,
        split_data=gastro_split,
        task_name="gastro_multilabel",
        image_size=args.image_size,
        num_workers=effective_workers,
        run_dir=session_dir / "01_demo_gastro_mil_baseline",
        seed=args.seed,
        dl_cfg=dl_cfg_1,
        label_names=GASTRO_LABEL_NAMES,
        class_names=[],
    )
    all_results["models"][key_1] = result_1

    # 2) 胃镜 advanced
    key_2 = "demo_gastro_proto_moe_former"
    print("\n[2/4] 训练 demo_gastro_proto_moe_former")
    model_2 = DemoGastroProtoMoEFormer(
        backbone_name="resnet50",
        pretrained=pretrained,
        freeze_stages=1,
        feature_dim=512,
        attn_dim=256,
        num_labels=3,
        num_experts=4,
        proto_per_label=8,
        relation_type="transformer",
        relation_layers=2,
        dropout=0.2,
    )
    trainer_cfg_2 = TrainerConfig(
        task_type="gastro_multilabel",
        model_family="gastro_advanced",
        num_classes=3,
        num_labels=3,
        max_epochs=args.epochs,
        patience=args.patience,
        lr=1.5e-4,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        grad_accum_steps=1 if use_multi_gpu else 4,
        amp=True,
        monitor_metric="macro_auc",
        monitor_mode="max",
        topk_evidence=5,
        loss_name="asymmetric",
        pos_weight=gastro_pos_weight,
        aux_loss_weights={
            "proto": 0.3,
            "consistency": 0.2,
            "expert_balance": 0.05,
        },
        use_multi_gpu=use_multi_gpu,
        resume_path=None,
    )
    dl_cfg_2 = {
        "train_batch_size": normalize_batch_size(demo_cfg["batch_size"][key_2], active_gpu_count),
        "eval_batch_size": normalize_batch_size(demo_cfg["eval_batch_size"][key_2], active_gpu_count),
        "train_max_instances": demo_cfg["train_max_instances"][key_2],
        "val_max_instances": demo_cfg["val_max_instances"][key_2],
        "test_max_instances": demo_cfg["test_max_instances"][key_2],
        "train_max_batch_instances": demo_cfg["train_max_batch_instances"][key_2],
        "eval_max_batch_instances": demo_cfg["eval_max_batch_instances"][key_2],
        "min_instances": demo_cfg["min_instances"],
        "train_sampling": demo_cfg["train_sampling_strategy"],
        "eval_sampling": demo_cfg["eval_sampling_strategy"],
        "random_instance_dropout": demo_cfg["random_instance_dropout"][key_2],
    }
    result_2 = run_single_model(
        model_name=key_2,
        model=model_2,
        trainer_cfg=trainer_cfg_2,
        split_data=gastro_split,
        task_name="gastro_multilabel",
        image_size=args.image_size,
        num_workers=effective_workers,
        run_dir=session_dir / "02_demo_gastro_proto_moe_former",
        seed=args.seed,
        dl_cfg=dl_cfg_2,
        label_names=GASTRO_LABEL_NAMES,
        class_names=[],
    )
    all_results["models"][key_2] = result_2

    # 3) 肠镜 baseline
    key_3 = "demo_colo_mil_baseline"
    print("\n[3/4] 训练 demo_colo_mil_baseline")
    model_3 = DemoColoMILBaseline(
        backbone_name="resnet50",
        pretrained=pretrained,
        freeze_stages=1,
        feature_dim=512,
        attn_dim=256,
        num_classes=2,
        dropout=0.2,
    )
    trainer_cfg_3 = TrainerConfig(
        task_type="colo_binary",
        model_family="colo_baseline",
        num_classes=2,
        num_labels=1,
        max_epochs=args.epochs,
        patience=args.patience,
        lr=2e-4,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        grad_accum_steps=1 if use_multi_gpu else 2,
        amp=True,
        monitor_metric="auc",
        monitor_mode="max",
        topk_evidence=5,
        loss_name="focal",
        pos_weight=colo_pos_weight,
        aux_loss_weights={},
        use_multi_gpu=use_multi_gpu,
        resume_path=None,
    )
    dl_cfg_3 = {
        "train_batch_size": normalize_batch_size(demo_cfg["batch_size"][key_3], active_gpu_count),
        "eval_batch_size": normalize_batch_size(demo_cfg["eval_batch_size"][key_3], active_gpu_count),
        "train_max_instances": demo_cfg["train_max_instances"][key_3],
        "val_max_instances": demo_cfg["val_max_instances"][key_3],
        "test_max_instances": demo_cfg["test_max_instances"][key_3],
        "train_max_batch_instances": demo_cfg["train_max_batch_instances"][key_3],
        "eval_max_batch_instances": demo_cfg["eval_max_batch_instances"][key_3],
        "min_instances": demo_cfg["min_instances"],
        "train_sampling": demo_cfg["train_sampling_strategy"],
        "eval_sampling": demo_cfg["eval_sampling_strategy"],
        "random_instance_dropout": demo_cfg["random_instance_dropout"][key_3],
    }
    result_3 = run_single_model(
        model_name=key_3,
        model=model_3,
        trainer_cfg=trainer_cfg_3,
        split_data=colo_split,
        task_name="colo_binary",
        image_size=args.image_size,
        num_workers=effective_workers,
        run_dir=session_dir / "03_demo_colo_mil_baseline",
        seed=args.seed,
        dl_cfg=dl_cfg_3,
        label_names=[],
        class_names=COLO_BINARY_CLASS_NAMES,
    )
    all_results["models"][key_3] = result_3

    # 4) 肠镜 advanced
    key_4 = "demo_colo_count_aware_debias_mil"
    print("\n[4/4] 训练 demo_colo_count_aware_debias_mil")
    model_4 = DemoColoCountAwareDebiasMIL(
        backbone_name="resnet50",
        pretrained=pretrained,
        freeze_stages=1,
        feature_dim=512,
        topk_lesion=8,
        topk_context=8,
        prototype_k=8,
        binary_num_classes=2,
        dropout=0.2,
    )
    trainer_cfg_4 = TrainerConfig(
        task_type="colo_binary",
        model_family="colo_advanced",
        num_classes=2,
        num_labels=1,
        max_epochs=args.epochs,
        patience=args.patience,
        lr=1.5e-4,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        grad_accum_steps=1 if use_multi_gpu else 4,
        amp=True,
        monitor_metric="auc",
        monitor_mode="max",
        topk_evidence=5,
        loss_name="bce",
        pos_weight=colo_pos_weight,
        aux_loss_weights={
            "count": 0.2,
            "proto": 0.25,
            "hard_negative": 0.2,
            "consistency": 0.1,
        },
        use_multi_gpu=use_multi_gpu,
        resume_path=None,
    )
    dl_cfg_4 = {
        "train_batch_size": normalize_batch_size(demo_cfg["batch_size"][key_4], active_gpu_count),
        "eval_batch_size": normalize_batch_size(demo_cfg["eval_batch_size"][key_4], active_gpu_count),
        "train_max_instances": demo_cfg["train_max_instances"][key_4],
        "val_max_instances": demo_cfg["val_max_instances"][key_4],
        "test_max_instances": demo_cfg["test_max_instances"][key_4],
        "train_max_batch_instances": demo_cfg["train_max_batch_instances"][key_4],
        "eval_max_batch_instances": demo_cfg["eval_max_batch_instances"][key_4],
        "min_instances": demo_cfg["min_instances"],
        "train_sampling": demo_cfg["train_sampling_strategy"],
        "eval_sampling": demo_cfg["eval_sampling_strategy"],
        "random_instance_dropout": demo_cfg["random_instance_dropout"][key_4],
    }
    result_4 = run_single_model(
        model_name=key_4,
        model=model_4,
        trainer_cfg=trainer_cfg_4,
        split_data=colo_split,
        task_name="colo_binary",
        image_size=args.image_size,
        num_workers=effective_workers,
        run_dir=session_dir / "04_demo_colo_count_aware_debias_mil",
        seed=args.seed,
        dl_cfg=dl_cfg_4,
        label_names=[],
        class_names=COLO_BINARY_CLASS_NAMES,
    )
    all_results["models"][key_4] = result_4

    summary_path = session_dir / "all_models_summary.json"
    summary_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("4 个模型已完成训练/验证/测试")
    print(f"总输出目录: {session_dir}")
    print(f"汇总文件: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
