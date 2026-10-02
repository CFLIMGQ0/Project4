#!/usr/bin/env python3
"""用原始五折清单核对已完成结果，兼容旧实验缺失split_ids.json的情况。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from tqdm import tqdm

import run_acpe_no_current_fivefold as exp


def read_json(path):
    return json.loads(path.read_text())


def predictions(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    root = exp.ROOTS["mr_rate"]
    protocol = read_json(root / exp.VARIANT / "protocol.json")
    reference_protocol = read_json(root / "123+1/protocol.json")
    # 原训练代码不改动，保留已完成实验的协议哈希。
    for name, digest in protocol["source_sha256"].items():
        assert exp.sha256(exp.ROOT / name) == digest, name
        if "/experiment/" in name:
            assert reference_protocol["source_sha256"][name] == digest, name
    inputs = exp.ROOT / "outputs/mr_rate_1k/experiment"
    folds = read_json(inputs / "patient_folds.json")["folds"]
    samples = read_json(inputs / "samples.json")
    labels = protocol["labels"]
    audits = []
    for fold in tqdm(range(1, 6), desc="核对MR-RATE五折划分与预测"):
        folder = exp.folder("mr_rate", fold)
        reference = root / "123+1" / f"fold_{fold}" / exp.MODEL
        config = read_json(folder / "config.json")
        old = read_json(reference / "config.json")
        params = dict(old["model"])
        params["apro_transition_groups"] = exp.GROUPS
        assert config["model"] == params
        assert config["settings"] == old["settings"]
        assert config["seed"] == old["seed"] == 42 + 100 * fold
        assert config["protocol_sha256"] == protocol["protocol_sha256"]
        test_fold, val_fold = fold - 1, fold % 5
        expected = {
            "train": [i for k, group in enumerate(folds)
                      if k not in (test_fold, val_fold) for i in group],
            "val": folds[val_fold], "test": folds[test_fold],
        }
        assert read_json(folder / "split_ids.json") == expected
        assert sorted(sum(expected.values(), [])) == list(range(1000))
        for split, filename in [("val", "validation_predictions.csv"),
                                ("test", "test_predictions.csv")]:
            current = predictions(folder / filename)
            historical = predictions(reference / filename)
            assert [int(row["patient_id"]) for row in current] == expected[split]
            assert [int(row["patient_id"]) for row in historical] == expected[split]
            for row, old_row in zip(current, historical):
                truths = [int(row[f"true_{label}"]) for label in labels]
                assert truths == samples[int(row["patient_id"])]["labels"]
                assert truths == [int(old_row[f"true_{label}"]) for label in labels]
        metric = read_json(folder / "test_metrics.json")
        assert metric["protocol_sha256"] == protocol["protocol_sha256"]
        assert metric["fold"] == fold
        rows = predictions(folder / "test_predictions.csv")
        f1s = []
        for label in labels:
            truth = [int(row[f"true_{label}"]) for row in rows]
            pred = [int(row[f"pred_{label}"]) for row in rows]
            assert pred == [int(float(row[f"prob_{label}"]) >= metric["thresholds"][label])
                            for row in rows]
            tp = sum(y and p for y, p in zip(truth, pred))
            denominator = sum(truth) + sum(pred)
            f1s.append(2 * tp / denominator if denominator else 0.)
        assert abs(sum(f1s) / len(f1s) - metric["macro_f1"]) < 1e-12
        for name in ("completed.json", "best_model.pt", "history.json"):
            assert (folder / name).stat().st_size > 0
        record = {"dataset": "mr_rate", "fold": fold,
                  "protocol_sha256": protocol["protocol_sha256"],
                  "macro_f1": metric["macro_f1"], "source": str(folder),
                  "splits_match": True}
        marker = exp.OUT / "completed" / f"mr_rate_fold_{fold}.json"
        if marker.exists():
            assert read_json(marker) == record
        else:
            exp.save_json(marker, record)
        audits.append({"fold": fold, "historical_split_file_available":
                       (reference / "split_ids.json").exists(),
                       "canonical_splits_match": True,
                       "historical_prediction_ids_and_targets_match": True,
                       "test_f1_recomputed": sum(f1s) / len(f1s)})
    exp.save_json(exp.OUT / "mr_rate_split_recovery_audit.json", {
        "说明": "训练及测试已完成；原汇总因旧实验后三折缺少划分副本而失败。现核对原始五折清单、旧预测病例及标签，并从保存的测试预测复算F1；未重训或改动预测。",
        "verifier_sha256": exp.sha256(Path(__file__)), "folds": audits})
    exp.summarize()
    print((exp.OUT / "results.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
