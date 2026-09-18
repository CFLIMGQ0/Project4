#!/usr/bin/env python3
"""复核AMEF-MIL五折结果，并重放保存权重的多模态预测。"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

import run_cq500_amef_multimodal as run
from run_cq500_table2_baselines import load_bags
from verify_cq500_multimodal_results import arrays, f1, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=run.ROOT / "outputs/cq500/amef_multimodal_uniform64")
    folder = parser.parse_args().output_dir
    run.configure_base()
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    protocol = json.loads((folder / "protocol.json").read_text())
    state = json.loads((folder / "run_state.json").read_text())
    assert state["status"] == "complete" and state["completed_fold_jobs"] == 5
    assert all(code == 0 for code in state["worker_exit_codes"])
    for name, path in protocol["source_paths"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == protocol["source_sha256"][name], name
    paths = {name: Path(path) for name, path in protocol["source_paths"].items()}
    texts, labels, split_source = run.base.load_inputs(SimpleNamespace(
        text_csv=paths["descriptions"], reads=paths["labels"], folds_json=paths["folds"]))
    folds = split_source["folds"]
    patient_ids, bags, targets = load_bags(paths["labels"], paths["feature_cache"])
    assert np.array_equal(targets, [labels[int(i)] for i in patient_ids])
    text_ids, text_mask = run.base.encode_descriptions(texts, patient_ids)
    mapping = {int(i): j for j, i in enumerate(patient_ids)}
    checked, all_test_rows, maximum_error = [], [], 0.0
    for fold in tqdm(range(1, 6), desc="复核与重放AMEF五折"):
        current = folder / f"fold_{fold}" / run.MODEL_KEY
        splits = json.loads((current / "split_ids.json").read_text())
        metrics = json.loads((current / "test_metrics.json").read_text())
        assert metrics["protocol_sha256"] == protocol["protocol_sha256"]
        assert splits["test"] == folds[fold-1] and splits["val"] == folds[fold % 5]
        assert splits["train"] == [i for j, group in enumerate(folds) if j not in (fold-1, fold % 5) for i in group]
        assert len(set(splits["train"] + splits["val"] + splits["test"])) == 491
        val_rows, test_rows = rows(current / "validation_predictions.csv"), rows(current / "test_predictions.csv")
        for name, records in (("val", val_rows), ("test", test_rows)):
            assert [int(r["patient_id"]) for r in records] == splits[name]
            y, _, _ = arrays(records)
            assert np.array_equal(y, [labels[i] for i in splits[name]])
        val_y, val_p, _ = arrays(val_rows)
        grid = protocol["settings"]["threshold_grid"]
        scores = np.stack([f1(val_y, val_p >= t) for t in grid])
        thresholds = np.asarray(grid)[scores.argmax(axis=0)]
        assert np.array_equal(thresholds, [metrics["thresholds"][n] for n in run.base.LABEL_NAMES])
        y, probabilities, predictions = arrays(test_rows)
        assert np.array_equal(predictions, probabilities >= thresholds)
        np.testing.assert_allclose(f1(y, predictions).mean(), metrics["macro_f1"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(f1(y, probabilities >= 0.5).mean(), metrics["macro_f1_fixed_0_5"], atol=1e-12, rtol=0)
        history = json.loads((current / "history.json").read_text())
        assert len(history) == protocol["settings"]["epochs"]
        assert all(np.isfinite([r["train_loss"], r["val_loss"], *r["aux_losses"].values()]).all() for r in history)
        best = min(history, key=lambda r: r["val_loss"])
        assert best["epoch"] == metrics["best_epoch"] and best["val_loss"] == metrics["best_val_loss"]
        checkpoint = torch.load(current / "best_model.pt", map_location="cpu", weights_only=True)
        assert checkpoint["protocol_sha256"] == protocol["protocol_sha256"]
        assert checkpoint["epoch"] == best["epoch"]
        assert checkpoint["thresholds"] == thresholds.tolist()
        model, aux = run.build_model(run.MODEL_KEY, checkpoint["model_parameters"])
        assert aux == {"image_aux": 0.0, "label_query_consistency": 0.01}
        model.load_state_dict(checkpoint["state_dict"])
        model.cuda()
        indices = [mapping[i] for i in splits["test"]]
        loader = run.loader(bags, targets, indices, False, metrics["seed"])
        _, replay_y, replay_p, replay_idx = run.base.evaluate(model, loader, text_ids, text_mask, "cuda:0")
        assert np.array_equal(replay_y, y) and replay_idx.tolist() == indices
        np.testing.assert_allclose(replay_p, probabilities, atol=1e-6, rtol=1e-5)
        maximum_error = max(maximum_error, float(np.abs(replay_p-probabilities).max()))
        with torch.inference_mode():
            batch = next(iter(loader))
            fused = run.forward_batch(model, batch, text_ids, text_mask, "cuda:0")
            assert not torch.allclose(fused["logits"], fused["image_only_logits"])
            changed = list(batch)
            changed[2] = 1-batch[2]
            other = run.forward_batch(model, changed, text_ids, text_mask, "cuda:0")
            assert torch.equal(fused["logits"], other["logits"])
        checked.append(metrics)
        all_test_rows.extend({**r, "fold": str(fold)} for r in test_rows)
        del model, checkpoint
        torch.cuda.empty_cache()
    oof_rows = rows(folder / f"{run.MODEL_KEY}_oof_predictions.csv")
    assert oof_rows == sorted(all_test_rows, key=lambda r: int(r["patient_id"]))
    assert [int(r["patient_id"]) for r in oof_rows] == list(range(491))
    y, probabilities, pred = arrays(oof_rows)
    summary = json.loads((folder / "summary.json").read_text())[0]
    for metric in ("macro_f1", "macro_f1_fixed_0_5"):
        values = [r[metric] for r in checked]
        np.testing.assert_allclose(np.mean(values), summary[metric+"_mean"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.std(values, ddof=1), summary[metric+"_std"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(f1(y, pred).mean(), summary["oof_macro_f1"], atol=1e-12, rtol=0)
    run.base.save_json(folder / "verification.json", {
        "status": "passed", "verified_fold_jobs": 5, "verified_oof_cases": 491,
        "source_hashes_unchanged": True, "exact_original_patient_folds": True,
        "thresholds_recomputed_from_validation_only": True, "F1_recomputed_from_TP_FP_FN": True,
        "multimodal_output_distinct_from_image_output": True,
        "prediction_independent_of_supplied_test_targets": True,
        "checkpoint_replays": 5, "replay_max_probability_error": maximum_error,
    })
    print("通过：5折及491例预测完成复核，保存权重可重现多模态结果。", flush=True)


if __name__ == "__main__":
    main()
