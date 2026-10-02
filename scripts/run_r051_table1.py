#!/usr/bin/env python3
"""固定 r051 配置，在表1的三个公开队列上完成五折训练和测试。"""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/acpe_research_200_20260928/workspace/src"
sys.path.insert(0, str(SOURCE))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from auto_research.model import ResearchModel
from auto_research.train import Bags, collate, cuda_batch, digest, evaluate, label_f1, loss_fn, write_json
from training.data import _encode_text_fields

OUT = ROOT / "outputs/r051_table1_20260929"
DATASETS = {"ct_rate": "ct_rate_680", "amos_mm": "amos_mm", "mr_rate_1k": "mr_rate_1k"}
NAMES = {"ct_rate": "CT-RATE", "amos_mm": "AMOS-MM", "mr_rate_1k": "MR-RATE-1K"}
CONFIG = {"position_warmup": 8, "descriptor": "no_current"}
SETTINGS = {"epochs": 30, "batch_size": 16, "lr": 2e-4, "weight_decay": .02,
            "warmup_ratio": .2, "instance_dropout": .25, "lqd_weight": .01,
            "image_weight": 0., "grad_clip": 1., "precision": "AMP float16",
            "max_instances": 64, "text_max_length": 512, "text_vocab_size": 8192,
            "checkpoint_selection": "minimum_validation_masked_ASL",
            "threshold_selection": "validation_labelwise_F1_grid_0.10_to_0.90_step_0.05",
            "seed_rule": "42 + 100 * fold (fold=1..5)"}
ASSIGNMENTS = {
    worker: [(ds, fold) for ds in ("amos_mm", "ct_rate", "mr_rate_1k")]
    for worker, fold in [("204_gpu0", 1), ("204_gpu2", 2), ("204_gpu3", 3),
                         ("202_gpu0", 4), ("202_gpu1", 5)]
}


def splits(dataset, fold):
    folder = ROOT / "outputs" / DATASETS[dataset] / "experiment"
    rows = {r["case_index"]: r for r in json.loads((folder / "samples.json").read_text())}
    metadata = [folder / "samples.json"]
    if dataset == "amos_mm":
        path = folder / "splits.json"
        obj = json.loads(path.read_text())
        exclusion = folder / "image_exclusions.json"
        excluded = {r["case_index"] for r in json.loads(exclusion.read_text())["excluded"]}
        val = set(obj["validation_folds"][fold - 1]) - excluded
        train = set(obj["development"]) - val - excluded
        test = set(obj["test"])
        assert not test & excluded and len(test) == 200
        metadata.extend([path, exclusion])
    else:
        path = folder / "patient_folds.json"
        groups = json.loads(path.read_text())["folds"]
        test = set(groups[fold - 1])
        val = set(groups[fold % 5])
        train = set().union(*map(set, groups)) - test - val
        metadata.append(path)
        patient_key = "patient_id" if dataset == "ct_rate" else "patient_uid"
        patients = [{rows[i][patient_key] for i in ids} for ids in (train, val, test)]
        assert not (patients[0] & patients[1] or patients[0] & patients[2] or patients[1] & patients[2])
    assert train and val and test and not (train & val or train & test or val & test)
    return folder, rows, {"train": sorted(train), "validation": sorted(val), "test": sorted(test)}, metadata


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {str(p.relative_to(SOURCE)): digest(p) for p in sorted(SOURCE.rglob("*.py"))}
    jobs = []
    for worker, tasks in ASSIGNMENTS.items():
        for ds, fold in tasks:
            _, _, ids, metadata = splits(ds, fold)
            jobs.append({"id": f"{ds}_fold{fold}", "dataset": ds, "fold": fold,
                         "seed": 42 + 100 * fold, "worker": worker, "split_ids": ids,
                         "input_hashes": {str(p.relative_to(ROOT)): digest(p) for p in metadata}})
    assert len(jobs) == len({j['id'] for j in jobs}) == 15
    plan = {"model": "ALM-MIL r051", "config": CONFIG, "settings": SETTINGS, "jobs": jobs,
            "source_sha256": sources, "runner_sha256": digest(Path(__file__)),
            "scope": "仅固定r051；不恢复200轮搜索；三个数据集各五次独立训练，测试集不参与选择",
            "amos_protocol": "五个开发验证折；同一个200例人工标签测试集，不是互斥测试折",
            "training_origin": "保留r051的冻结特征、AMP、梯度裁剪、损失和最低验证ASL选模；使用表1划分及种子规则",
            "historical_comparison": "历史表1的部分实现使用FP32及不同梯度裁剪，不能把差值全部归因于ACPE改动",
            "test_exposure": "此前搜索在原五折的开发数据中进行，且历史实验看过测试结果；此次是内部评估，不宣称独立外部验证",
            "gpu_policy": "用户已解除两主机GPU0限制；每个分配GPU仅一个本轮worker进程；不终止其它程序"}
    plan["protocol_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    path = OUT / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise RuntimeError("协议已存在但不同，拒绝混用结果")
    else:
        write_json(path, plan)
    return plan


def tune(val):
    grid = np.round(np.arange(.1, .901, .05), 2)
    scores = np.stack([label_f1(val['y'], val['p'], val['known'], t) for t in grid])
    thresholds = grid[scores.argmax(0)]
    for j in range(len(thresholds)):
        observed = val['y'][val['known'][:, j], j]
        if len(observed) == 0 or len(np.unique(observed)) < 2:
            thresholds[j] = .5
    return thresholds


def metrics(value, thresholds):
    y, p, known = value['y'] > .5, value['p'], value['known']
    pred = p >= thresholds
    tp, fp, fn = [(x & known).sum() for x in (pred & y, pred & ~y, ~pred & y)]
    return {"macro_f1": float(label_f1(value['y'], p, known, thresholds).mean()),
            "macro_f1_fixed_0_5": float(label_f1(value['y'], p, known, .5).mean()),
            "per_label_f1": label_f1(value['y'], p, known, thresholds).tolist(),
            "micro_f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "loss": float(value['loss']), "n": len(y)}


def save_checkpoint(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def load_cache(folder, rows, ids, cache, hashes):
    for index in tqdm([i for i in ids if i not in cache], desc="读取特征", mininterval=10):
        path = folder / 'features' / f'{index:04d}.npz'
        with np.load(path, allow_pickle=False) as z:
            feature = z['features'].astype(np.float32)
            positions = z['slice_indices'].astype(np.int64)
            count = int(z['original_count'])
        assert 0 < len(feature) <= 64 and feature.shape[1] == 768 and np.isfinite(feature).all()
        assert len(positions) == len(feature) and np.all(np.diff(positions) > 0)
        assert positions.min() >= 0 and positions.max() < count
        tokens, mask = _encode_text_fields({'watch': rows[index]['findings_masked']}, ('watch',),
                                          max_length=512, vocab_size=8192)
        cache[index] = feature, positions, count, tokens, mask
        hashes[str(index)] = digest(path)


def train_job(job, plan, shared):
    ds, fold, seed = job['dataset'], job['fold'], job['seed']
    output = OUT / ds / f'fold_{fold}'
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / 'result.json').exists():
        assert json.loads((output / 'result.json').read_text())['protocol_sha256'] == plan['protocol_sha256']
        return
    folder, rows, ids, metadata = splits(ds, fold)
    assert ids == job['split_ids']
    assert {str(p.relative_to(ROOT)): digest(p) for p in metadata} == job['input_hashes']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cache, hashes = shared.setdefault(ds, ({}, {}))
    load_cache(folder, rows, ids['train'] + ids['validation'], cache, hashes)
    write_json(output / 'protocol.json', {"job": job, "config": CONFIG, "settings": SETTINGS,
                                        "protocol_sha256": plan['protocol_sha256']})
    write_json(output / 'split_ids.json', ids)
    model = ResearchModel(len(rows[ids['train'][0]]['labels']), CONFIG).cuda()
    torch.manual_seed(seed + 1000); torch.cuda.manual_seed_all(seed + 1000)
    groups = [{'params': [p for n,p in model.named_parameters() if 'apro_positioner' not in n]},
              {'params': [p for n,p in model.named_parameters() if 'apro_positioner' in n]}]
    optimizer = torch.optim.AdamW(groups, lr=SETTINGS['lr'], weight_decay=SETTINGS['weight_decay'])
    def loader(indices, training=False):
        return DataLoader(Bags(rows, indices, cache, CONFIG, training=training), batch_size=16,
                          shuffle=training, num_workers=0, collate_fn=collate)
    train_loader, val_loader = loader(ids['train'], True), loader(ids['validation'])
    steps = SETTINGS['epochs'] * len(train_loader)
    warmup = max(1, int(.2 * steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        (step+1)/warmup if step < warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup))))
    scaler = torch.amp.GradScaler('cuda')
    best, best_epoch, history, start_epoch = float('inf'), 0, [], 0
    started = time.monotonic()
    if (output / 'last.pt').exists():
        saved = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
        assert saved['protocol_sha256'] == plan['protocol_sha256']
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
        best, best_epoch, history, start_epoch = saved['best'], saved['best_epoch'], saved['history'], saved['epoch']
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
    previous_seconds = history[-1]['seconds'] if history else 0.
    worker_status = OUT / 'workers' / (job['worker']+'.json')
    print(f"开始 {job['id']}，训练/验证/测试={len(ids['train'])}/{len(ids['validation'])}/{len(ids['test'])}", flush=True)
    for epoch in range(start_epoch, SETTINGS['epochs']):
        model.train(); model.epoch = epoch
        sums = {k:0. for k in ('total','main','lqd')}
        for batch in train_loader:
            inputs, target, known = cuda_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                out = model(**inputs, labels=(target,known))
            main = loss_fn(out['logits'],target,known)
            lqd = out['research_lqd']
            total = main + .01*lqd
            if not torch.isfinite(total): raise FloatingPointError('非有限训练损失')
            scaler.scale(total).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            for k,v in [('main',main),('lqd',lqd),('total',total)]:sums[k]+=float(v.detach())
        val = evaluate(model,val_loader)
        if not np.isfinite(val['p']).all() or not math.isfinite(val['loss']):
            raise FloatingPointError('非有限验证输出')
        if val['loss'] < best:
            best,best_epoch = val['loss'],epoch+1
            save_checkpoint(output/'best.pt',{'model':model.state_dict(),'epoch':best_epoch,
                                             'config':CONFIG,'protocol_sha256':plan['protocol_sha256']})
        row = {'epoch':epoch+1,'train':{k:v/len(train_loader) for k,v in sums.items()},
               'validation_loss':float(val['loss']),
               'validation_f1_05':float(label_f1(val['y'],val['p'],val['known'],.5).mean()),
               'seconds':previous_seconds+time.monotonic()-started}
        history.append(row)
        write_json(output/'history.json',history)
        save_checkpoint(output/'last.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch+1,
            'best':best,'best_epoch':best_epoch,'history':history,'python_rng':random.getstate(),
            'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state(),'protocol_sha256':plan['protocol_sha256']})
        write_json(worker_status,{'pid':os.getpid(),'state':'training','job':job['id'],
                                  'epoch':epoch+1,'updated':time.time()})
        print(json.dumps({'job':job['id'],**row},ensure_ascii=False),flush=True)
    saved = torch.load(output/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(saved['model']); model.epoch = saved['epoch']-1
    val = evaluate(model,val_loader)
    thresholds = tune(val)
    # 阈值和权重先固定，随后仅评估测试集一次。
    write_json(output/'frozen_selection.json',{'best_epoch':saved['epoch'],'thresholds':thresholds.tolist(),
              'validation_metrics':metrics(val,thresholds),'best_sha256':digest(output/'best.pt')})
    np.savez_compressed(output/'validation_predictions.npz',**{k:val[k] for k in ('y','p','known','ids')},thresholds=thresholds)
    load_cache(folder,rows,ids['test'],cache,hashes)
    test = evaluate(model,loader(ids['test']))
    if not np.isfinite(test['p']).all(): raise FloatingPointError('非有限测试输出')
    np.savez_compressed(output/'test_predictions.npz',**{k:test[k] for k in ('y','p','known','ids')},thresholds=thresholds)
    write_json(output/'cache_hashes.json',{str(i):hashes[str(i)] for i in sorted(set().union(*map(set,ids.values())))})
    result = {'status':'completed','dataset':ds,'fold':fold,'seed':seed,'model':'ALM-MIL r051',
              'best_epoch':saved['epoch'],'position_scale':min(1.,saved['epoch']/8.),
              'thresholds':thresholds.tolist(),'protocol_sha256':plan['protocol_sha256'],
              'seconds':previous_seconds+time.monotonic()-started,**metrics(test,thresholds)}
    write_json(output/'result.json',result)
    print('完成 '+json.dumps(result,ensure_ascii=False),flush=True)
    del model,optimizer,scheduler,scaler,saved,out,total,main,lqd
    gc.collect();torch.cuda.empty_cache()


def aggregate(plan):
    results=[];summaries=[]
    for ds in DATASETS:
        group=[]
        for fold in range(1,6):
            path=OUT/ds/f'fold_{fold}'/'result.json'
            if path.exists():
                item=json.loads(path.read_text())
                assert item['protocol_sha256']==plan['protocol_sha256']
                group.append(item);results.append(item)
        item={'dataset':ds,'completed_folds':len(group)}
        if len(group)==5:
            item.update(macro_f1_mean=statistics.mean(r['macro_f1'] for r in group),
                        macro_f1_std=statistics.stdev(r['macro_f1'] for r in group),
                        macro_f1_fixed_0_5_mean=statistics.mean(r['macro_f1_fixed_0_5'] for r in group),
                        macro_f1_fixed_0_5_std=statistics.stdev(r['macro_f1_fixed_0_5'] for r in group))
        summaries.append(item)
    write_json(OUT/'summary.json',{'completed_jobs':len(results),'expected_jobs':15,'datasets':summaries,
                                 'protocol_sha256':plan['protocol_sha256'],'std_ddof':1})
    report=['# r051：Table 1 三个公开数据集分类结果','','固定 no_current 与8轮位置warmup；各数据集五次独立训练。',
            '沿用历史表1划分；AMOS-MM为五次训练评估同一个200例人工测试集。',
            '训练保持r051方案（冻结ConvNeXt缓存、AMP、ASL+0.01 LQD、最低验证ASL选模），与历史实现并非严格单因素对照。',
            '阈值仅在各折验证集选择；没有删除实验；不按测试结果修改模型或筛选折。','',
            '| 数据集 | 完成 | F1（%，均值±样本标准差） | 固定0.5阈值F1（%） |',
            '|---|---:|---:|---:|']
    for r in summaries:
        if r['completed_folds']==5:
            report.append(f"| {NAMES[r['dataset']]} | 5/5 | {r['macro_f1_mean']*100:.2f} ± {r['macro_f1_std']*100:.2f} | {r['macro_f1_fixed_0_5_mean']*100:.2f} ± {r['macro_f1_fixed_0_5_std']*100:.2f} |")
        else:report.append(f"| {NAMES[r['dataset']]} | {r['completed_folds']}/5 | - | - |")
    report.extend(['','| 数据集 | 折 | 测试F1（%） | 最佳epoch |','|---|---:|---:|---:|'])
    for r in results:report.append(f"| {NAMES[r['dataset']]} | {r['fold']} | {r['macro_f1']*100:.2f} | {r['best_epoch']} |")
    (OUT/'results.md').write_text('\n'.join(report)+'\n')
    if results:
        with (OUT/'fold_results.csv').open('w',newline='') as stream:
            keys=['dataset','fold','seed','macro_f1','macro_f1_fixed_0_5','best_epoch','seconds']
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
            writer.writerows({k:r[k] for k in keys} for r in results)
    print(f"已完成 {len(results)}/15",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--aggregate',action='store_true')
    parser.add_argument('--worker',choices=list(ASSIGNMENTS))
    args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False);torch.backends.cudnn.benchmark=False
    plan=prepare()
    if args.prepare or args.aggregate:aggregate(plan)
    if args.worker:
        (OUT/'workers').mkdir(exist_ok=True)
        lock=(OUT/'workers'/(args.worker+'.lock')).open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        shared={}
        for job in plan['jobs']:
            if job['worker']!=args.worker:continue
            try:train_job(job,plan,shared)
            except Exception:
                error=traceback.format_exc()
                write_json(OUT/'workers'/(args.worker+'.json'),{'state':'failed','pid':os.getpid(),
                    'job':job['id'],'error':error,'updated':time.time()})
                raise
        write_json(OUT/'workers'/(args.worker+'.json'),{'state':'completed','pid':os.getpid(),'updated':time.time()})
    if not(args.prepare or args.aggregate or args.worker):parser.error('请指定运行方式')


if __name__=='__main__':main()
