#!/usr/bin/env python3
"""等待共用CT特征后，比较单折纯图像与ALM-MIL；仅验证集诊断，不改论文结果。"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

import run_abdomenatlas3_table1 as runner
from prepare_abdomenatlas3_table1 import ROOT, OUT as FEATURES, save_json
from training.losses import AsymmetricLossMultiLabel
from exp_10.train_text_classification import tune_thresholds

INPUT = ROOT / "outputs/abdomenatlas3_mini/anonymous_findings_control"
OUT = INPUT / "image_comparison"


def run(key):
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.42)
    torch.backends.mha.set_fastpath_enabled(False)
    runner.configure()
    rows = json.loads((INPUT / "samples.json").read_text())
    split = json.loads((INPUT / "split_ids.json").read_text())
    y = np.asarray([r["labels"] for r in rows], dtype=np.int64)
    seed = 142
    runner.core.image_base.seed_everything(seed)
    runner.core.INPUT = FEATURES
    bags = runner.core.load_features(rows)
    base = runner.core.fusion_base
    token_ids, token_mask = base.encode_descriptions({r["case_index"]: r["findings_masked"] for r in rows}, np.arange(len(rows)))
    loaders = {k: base.loader(bags, y, split[k], k == "train", seed) for k in ("train", "val")}
    folder = OUT / key
    folder.mkdir(parents=True, exist_ok=True)
    params = runner.parameters()[key]
    model, auxiliary = runner.build_model(key, params)
    device = torch.device("cuda:0")
    model.to(device)
    settings = runner.SETTINGS
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    total = len(loaders["train"]) * settings["epochs"]
    warmup = int(total * settings["warmup_ratio"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s+1)/max(1,warmup) if s<warmup else .5*(1+math.cos(math.pi*(s-warmup)/max(1,total-warmup))))
    scaler = torch.amp.GradScaler("cuda")
    criterion = AsymmetricLossMultiLabel()
    history, best_loss = [], math.inf
    save_json(folder / "protocol.json", {"model": key, "parameters": params, "settings": settings,
        "seed": seed, "split_ids": {k: split[k] for k in ("train", "val")},
        "scope": "匿名病灶所见的第一折训练/验证对照，测试折不评分，不进入正式五折结果",
        "selection": "验证ASL最小，沿用此前图像和多模态协议",
        "text_protocol": json.loads((INPUT / "protocol.json").read_text()),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "underlying_protocol_sha256": runner.protocol()["protocol_sha256"]})
    for epoch in tqdm(range(1, settings["epochs"]+1), desc=key):
        model.train(); losses=[]
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                output = base.forward_batch(model, batch, token_ids, token_mask, device, training=True)
                loss = criterion(output["logits"], batch[2].to(device))
                for name, weight in auxiliary.items():
                    loss += float(weight)*output["aux_losses"][name]
            assert torch.isfinite(loss), (key, epoch)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();scheduler.step()
            losses.append(float(loss.detach()))
        val_loss, target, probability, indices = base.evaluate(model, loaders["val"], token_ids, token_mask, device)
        assert indices.tolist() == split["val"]
        thresholds = tune_thresholds(target, probability, settings["threshold_grid"])
        f1 = float(f1_score(target, probability>=thresholds, average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": val_loss, "validation_f1": f1})
        if val_loss < best_loss:
            best_loss = val_loss
            result = {"model": runner.MODELS[key], "epoch": epoch, "validation_loss": val_loss, "validation_macro_f1": f1,
                      "per_label_f1": dict(zip(runner.LABELS, f1_score(target, probability>=thresholds, average=None, zero_division=0).tolist())),
                      "validation_fixed_0_5_f1": float(f1_score(target, probability>=.5, average="macro", zero_division=0)),
                      "thresholds": thresholds.tolist()}
            torch.save({"state_dict": model.state_dict(), "model_parameters": params, "epoch": epoch}, folder/"best_model.pt")
            np.savez_compressed(folder/"validation_predictions.npz", case_indices=indices, labels=target, probabilities=probability, thresholds=thresholds)
            save_json(folder/"validation_metrics.json",result)
        save_json(folder/"history.json",history)
        print(f"{key} epoch={epoch} validation_f1={f1:.4f}",flush=True)
    save_json(folder/"completed.json",result)


def supervise():
    OUT.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (OUT/"run.lock").open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.time()
        expected=len(json.loads((INPUT/"samples.json").read_text()))
        while True:
            count=sum(1 for p in (FEATURES/"features").glob('*.npz') if not p.name.endswith('.tmp.npz'))
            state={"status":"waiting_for_features","pid":os.getpid(),"completed_feature_cases":count,"expected_feature_cases":expected,
                   "models":["attention_mil","amef_multimodal"],"gpus":[2,3],"started_unix":started,"updated_unix":time.time()}
            save_json(OUT/"run_state.json",state)
            if count==expected:break
            time.sleep(20)
        active=[]
        for gpu,key in [(2,'attention_mil'),(3,'amef_multimodal')]:
            if (OUT/key/'completed.json').exists():continue
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            handle=(OUT/(key+'.log')).open('a')
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--model',key],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            active.append((p,handle,key,gpu))
        state.update(status='running',active=[{'pid':p.pid,'model':key,'gpu':gpu} for p,_,key,gpu in active])
        save_json(OUT/'run_state.json',state)
        while any(p.poll() is None for p,_,_,_ in active):time.sleep(20)
        for _,h,_,_ in active:h.close()
        state.update(status='complete' if all(p.returncode==0 for p,_,_,_ in active) else 'incomplete',
                     exit_codes={key:p.returncode for p,_,key,_ in active},updated_unix=time.time())
        save_json(OUT/'run_state.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['attention_mil','amef_multimodal'])
    args=parser.parse_args()
    if args.model:run(args.model)
    else:supervise()
