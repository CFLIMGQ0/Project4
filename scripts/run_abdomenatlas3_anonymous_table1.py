#!/usr/bin/env python3
"""采用去器官归属病灶所见运行表1五折，独立保存输入、协议与结果。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

import run_abdomenatlas3_table1 as base
from prepare_abdomenatlas3_table1 import ROOT, save_json, sha256

SOURCE = ROOT / "outputs/abdomenatlas3_mini/anonymous_findings_control"
INPUT = ROOT / "outputs/abdomenatlas3_mini/experiment_anonymous_findings"
OUT = ROOT / "outputs/abdomenatlas3_mini/table1_fivefold_original_uit_anonymous_findings"
ORIGINAL_PROTOCOL = base.protocol
ORIGINAL_TEXT_FOLD = base.text_fold


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prepare_inputs():
    """固定已验证的文本版本，复用原标签、五折和图像缓存。"""
    rows = json.loads((SOURCE / "samples.json").read_text())
    original = json.loads((base.FEATURE_INPUT / "samples.json").read_text())
    assert len(rows) == len(original) == 9261
    for row, old in zip(rows, original):
        for field in ("case_index", "exam_id", "patient_id", "labels"):
            assert row[field] == old[field], (field, row["exam_id"])
        assert row["findings_masked"].strip()
    folds = json.loads((base.FEATURE_INPUT / "patient_folds.json").read_text())
    prior_split = json.loads((SOURCE / "split_ids.json").read_text())
    assert prior_split == {"train": [i for j,g in enumerate(folds["folds"]) if j not in (0,1) for i in g],
                           "val": folds["folds"][1], "test": folds["folds"][0]}
    masking = json.loads((SOURCE / "protocol.json").read_text())
    masking.update(version="anonymous_findings_fivefold_v1",
                   scope="用户授权的表1完整五折，结果独立保存，不自动修改论文",
                   evaluation="固定现有文本规则，沿用同一检查五折；不按测试表现继续调整掩码",
                   development="文本规则曾在第一折验证集上比较；该开发集会在轮换五折中作为测试集，结果属于开发后的探索性五折评价")
    prep = json.loads((base.FEATURE_INPUT / "preparation_protocol.json").read_text())
    prep["feature_preparation_sha256"] = prep.pop("preparation_sha256")
    prep.update(text_variant=masking["version"], text_source=str(SOURCE / "samples.json"),
                samples_sha256=sha256(SOURCE / "samples.json"), masking_sha256=digest(masking))
    prep["preparation_sha256"] = digest(prep)
    for name,value in {"samples.json":rows, "patient_folds.json":folds,
                       "preparation_protocol.json":prep, "masking_protocol.json":masking}.items():
        path = INPUT / name
        if path.exists():
            assert json.loads(path.read_text()) == value, f"禁止覆盖已固定的输入：{path}"
        else:
            save_json(path,value)


def protocol():
    value = ORIGINAL_PROTOCOL()
    value.pop("protocol_sha256")
    value.update(report="官方structured report的固定四器官基础所见，加去除器官归属的病灶尺寸、体积、密度和HU；病灶统一排序且最多六条",
                 report_limit="合成描述与标签共享标注来源；器官定位信息已移除，但仍保留病灶存在与影像所见，不声称不存在间接关联",
                 development="文本规则依据第一折训练/验证诊断固定；后续五折属于探索性比较，不作为全新独立盲测",
                 reuse="复用两项已完成的第一折文本checkpoint，重新核对验证预测并用其验证阈值评价测试折；其余折正常训练")
    for name in ("run_abdomenatlas3_anonymous_table1.py", "abdomenatlas_anonymous_findings_control.py",
                 "abdomenatlas_background_text_control.py"):
        path = ROOT / "src/scripts" / name
        value["source_sha256"][str(path.relative_to(ROOT))] = sha256(path)
    for key in ("textcnn_encoder", "vocab_attention_encoder"):
        for name in ("config.json", "vocabulary.json", "best_model.pt", "validation_metrics.json", "best_validation_predictions.npz"):
            path = SOURCE / key / name
            value["source_sha256"][str(path.relative_to(ROOT))] = sha256(path)
    value["protocol_sha256"] = digest(value)
    return value


def reuse_first_fold(model_name, config, splits, vocabulary, output_dir):
    """已完成的相同输入/配置checkpoint仅补测试评分，避免重复训练。"""
    from torch.utils.data import DataLoader
    from exp_10.models import build_text_classifier
    from exp_10.train_text_classification import build_dataset_class, predict, tune_thresholds, calculate_metrics, save_predictions
    source = SOURCE / model_name
    cfg = json.loads((source / "config.json").read_text())
    for field in ("seed", "data", "training", "model"):
        assert cfg[field] == config[field], f"旧checkpoint配置不匹配：{field}"
    assert vocabulary == json.loads((source / "vocabulary.json").read_text())
    dataset_cls = build_dataset_class()
    loaders = {k: DataLoader(dataset_cls(splits[k], model_name, vocabulary, config["data"]["hash_vocab_size"],
                                        config["data"]["max_length"]), batch_size=config["training"]["batch_size"],
                            shuffle=False, num_workers=0) for k in ("val", "test")}
    model = build_text_classifier(model_name, vocabulary_size=len(vocabulary),
        hash_vocab_size=config["data"]["hash_vocab_size"], num_labels=3,
        max_length=config["data"]["max_length"], model_config=config["model"]).cuda()
    model.load_state_dict(torch.load(source / "best_model.pt", map_location="cpu", weights_only=True))
    val_p,val_y = predict(model, loaders["val"], torch.device("cuda"))
    with np.load(source / "best_validation_predictions.npz") as saved:
        assert np.array_equal(val_y,saved["labels"])
        assert [int(r.patient_id) for r in splits["val"]] == saved["case_indices"].tolist()
        np.testing.assert_allclose(val_p,saved["probabilities"],rtol=1e-5,atol=1e-6)
        thresholds=tune_thresholds(val_y,val_p,config["training"]["threshold_grid"])
        np.testing.assert_allclose(thresholds,saved["thresholds"],atol=1e-7)
    test_p,test_y = predict(model, loaders["test"], torch.device("cuda"))
    metrics = calculate_metrics(test_y,test_p,thresholds,base.LABELS)
    previous = json.loads((source / "validation_metrics.json").read_text())
    metrics.update(model_name=model_name,best_epoch=previous["best_epoch"],
                   best_val_macro_f1=previous["validation_macro_f1"],checkpoint_reused_from=str(source),
                   text_field="findings_masked",answer_masking=True)
    folder=output_dir / model_name
    for name in ("best_model.pt", "history.json", "validation_metrics.json"):
        shutil.copy2(source / name,folder / name)
    save_json(folder / "test_metrics.json",metrics)
    save_predictions(folder / "test_predictions.csv",splits["test"],test_p,metrics,base.LABELS)
    return metrics


def text_fold(key, fold, rows, folds, protocol_digest):
    import exp_10.train_text_classification as trainer
    original = trainer.train_one_model
    if fold == 0 and key in ("textcnn_encoder", "vocab_attention_encoder"):
        trainer.train_one_model = reuse_first_fold
    try:
        ORIGINAL_TEXT_FOLD(key,fold,rows,folds,protocol_digest)
    finally:
        trainer.train_one_model = original
    path=OUT / f"fold_{fold+1}" / key / "test_metrics.json"
    metrics=json.loads(path.read_text())
    metrics["text_source"]="AbdomenAtlas官方结构化所见：固定基础段落及去器官归属的病灶描述"
    save_json(path,metrics)


def activate():
    base.INPUT,base.OUT = INPUT,OUT
    base.protocol,base.text_fold = protocol,text_fold


activate()

if __name__ == "__main__":
    prepare_inputs()
    base.main()
