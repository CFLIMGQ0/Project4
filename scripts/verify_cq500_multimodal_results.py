#!/usr/bin/env python3
"""独立复核CQ500多模态实验的划分、阈值、F1与已保存权重。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from tqdm import tqdm

import run_cq500_table2_multimodal_baselines as run


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def arrays(records):
    y = np.array([[int(r[f"true_{n}"]) for n in run.LABEL_NAMES] for r in records])
    p = np.array([[float(r[f"prob_{n}"]) for n in run.LABEL_NAMES] for r in records])
    pred = np.array([[int(r[f"pred_{n}"]) for n in run.LABEL_NAMES] for r in records])
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return y, p, pred


def f1(y, pred):
    tp = (y * pred).sum(axis=0)
    fp = ((1-y) * pred).sum(axis=0)
    fn = (y * (1-pred)).sum(axis=0)
    denominator = 2*tp + fp + fn
    return np.divide(2*tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=run.ROOT / "outputs/cq500/table2_multimodal_baselines_uniform64")
    parser.add_argument("--replay-checkpoints", action="store_true")
    args = parser.parse_args()
    folder = args.output_dir
    protocol = json.loads((folder / "protocol.json").read_text())
    state = json.loads((folder / "run_state.json").read_text())
    assert state["status"] == "complete" and state["completed_fold_jobs"] == 20
    assert state["worker_exit_codes"] == [0] * len(state["worker_exit_codes"])
    for name, path in protocol["source_paths"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == protocol["source_sha256"][name], name
    paths = {name: Path(path) for name, path in protocol["source_paths"].items()}
    folds = json.loads(paths["folds"].read_text())["folds"]
    labels = {}
    for row in rows(paths["labels"]):
        patient = int(re.search(r"(\d+)$", row["name"]).group(1))
        labels[patient] = [int(sum(int(row[f"R{i}:{n}"]) for i in (1,2,3)) >= 2)
                           for n in ("IPH", "MassEffect", "MidlineShift")]
    assert len(labels) == 491
    summary = {r["model_key"]: r for r in json.loads((folder / "summary.json").read_text())}
    assert set(summary) == set(run.MODELS)
    if args.replay_checkpoints:
        import torch
        from run_cq500_table2_baselines import load_bags
        torch.set_num_threads(4)
        torch.backends.mha.set_fastpath_enabled(False)
        patient_ids, bags, targets = load_bags(paths["labels"], paths["feature_cache"])
        texts = {int(r["patient_id"]): r["image_description"] for r in rows(paths["descriptions"])}
        text_ids, text_mask = run.encode_descriptions(texts, patient_ids)
        mapping = {int(i): j for j, i in enumerate(patient_ids)}
    verified, replay_max_error = [], 0.0
    for key, fold in tqdm([(k,f) for k in run.MODELS for f in range(1,6)], desc="复核模型与折次"):
        current = folder / f"fold_{fold}" / key
        split = json.loads((current / "split_ids.json").read_text())
        metrics = json.loads((current / "test_metrics.json").read_text())
        marker = json.loads((current / "completed.json").read_text())
        assert marker["protocol_sha256"] == metrics["protocol_sha256"] == protocol["protocol_sha256"]
        assert split["test"] == folds[fold-1] and split["val"] == folds[fold % 5]
        expected_train = [i for j, group in enumerate(folds) if j not in (fold-1, fold % 5) for i in group]
        assert split["train"] == expected_train
        assert len(set(split["train"] + split["val"] + split["test"])) == 491
        val_rows, test_rows = rows(current / "validation_predictions.csv"), rows(current / "test_predictions.csv")
        for name, records in (("val", val_rows), ("test", test_rows)):
            assert [int(r["patient_id"]) for r in records] == split[name]
            y, _, _ = arrays(records)
            assert np.array_equal(y, [labels[i] for i in split[name]])
        val_y, val_p, _ = arrays(val_rows)
        grid = protocol["settings"]["threshold_grid"]
        scores = np.stack([f1(val_y, val_p >= threshold) for threshold in grid])
        thresholds = np.asarray(grid)[scores.argmax(axis=0)]
        assert np.array_equal(thresholds, [metrics["thresholds"][n] for n in run.LABEL_NAMES])
        y, p, pred = arrays(test_rows)
        assert np.array_equal(pred, p >= thresholds)
        np.testing.assert_allclose(f1(y, pred).mean(), metrics["macro_f1"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(f1(y, p >= 0.5).mean(), metrics["macro_f1_fixed_0_5"], rtol=0, atol=1e-12)
        history = json.loads((current / "history.json").read_text())
        assert len(history) == protocol["settings"]["epochs"]
        best = min(history, key=lambda r: r["val_loss"])
        assert best["epoch"] == metrics["best_epoch"] and best["val_loss"] == metrics["best_val_loss"]
        if args.replay_checkpoints:
            checkpoint = torch.load(current / "best_model.pt", map_location="cpu", weights_only=True)
            assert checkpoint["protocol_sha256"] == protocol["protocol_sha256"]
            assert checkpoint["epoch"] == best["epoch"]
            model, _ = run.build_model(key, checkpoint["model_parameters"])
            model.load_state_dict(checkpoint["state_dict"])
            model.to("cuda:0")
            indices = [mapping[i] for i in split["test"]]
            data_loader = run.loader(bags, targets, indices, False, metrics["seed"])
            _, replay_y, replay_p, replay_idx = run.evaluate(model, data_loader, text_ids, text_mask, "cuda:0")
            assert np.array_equal(replay_y, y) and replay_idx.tolist() == indices
            np.testing.assert_allclose(replay_p, p, rtol=1e-5, atol=1e-6)
            replay_max_error = max(replay_max_error, float(np.abs(replay_p-p).max()))
            del model, checkpoint
            torch.cuda.empty_cache()
        verified.append({"model": key, "fold": fold, "cases": len(test_rows), "macro_f1": metrics["macro_f1"]})
    for key in run.MODELS:
        records = rows(folder / f"{key}_oof_predictions.csv")
        assert [int(r["patient_id"]) for r in records] == list(range(491))
        y, p, pred = arrays(records)
        assert np.array_equal(y, [labels[i] for i in range(491)])
        fold_means, fixed_means = [], []
        for fold in range(1,6):
            group = [r for r in records if int(r["fold"]) == fold]
            original = rows(folder / f"fold_{fold}" / key / "test_predictions.csv")
            assert sorted(group, key=lambda r: int(r["patient_id"])) == sorted(
                [{**r, "fold": str(fold)} for r in original], key=lambda r: int(r["patient_id"]))
            fy, fp, f_pred = arrays(group)
            fold_means.append(f1(fy, f_pred).mean())
            fixed_means.append(f1(fy, fp >= 0.5).mean())
        for values, prefix in ((fold_means, "macro_f1"), (fixed_means, "macro_f1_fixed_0_5")):
            np.testing.assert_allclose(np.mean(values), summary[key][prefix+"_mean"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(np.std(values, ddof=1), summary[key][prefix+"_std"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(f1(y, pred).mean(), summary[key]["oof_macro_f1"], rtol=0, atol=1e-12)
    run.save_json(folder / "verification.json", {
        "status": "passed", "verified_fold_jobs": len(verified), "verified_oof_predictions": 4*491,
        "source_hashes_unchanged": True, "patient_fold_membership_exact": True,
        "thresholds_recomputed_from_validation_only": True, "F1_recomputed_from_TP_FP_FN": True,
        "checkpoint_replays": 20 if args.replay_checkpoints else 0,
        "checkpoint_replay_max_probability_error": replay_max_error if args.replay_checkpoints else None,
        "folds": verified,
    })
    print("复核通过：20折、1964条折外预测；原输入与划分保持不变。", flush=True)


if __name__ == "__main__":
    main()
