#!/usr/bin/env python3
"""复用 r051 五折权重，训练配对 Original PE 并评估完整原序列删除网格。"""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/scripts'))
import run_r051_table1 as base
import numpy as np
import torch
from tqdm import tqdm
from paper_block_deletion import block_sampling, RATIOS, BLOCKS

OUT = ROOT / 'outputs/r051_3d_fivefold_20260929'
R051 = ROOT / 'outputs/r051_table1_20260929'
FEATURES = ROOT / 'outputs/paper_results/figure1_features'
GRID = [(r, b) for r in RATIOS for b in ([1] if r == 0 else BLOCKS)]
CONFIG = {'position': 'original_pe'}
WORKERS = {'204_gpu0': [1, 4], '204_gpu2': [2], '202_gpu1': [3, 5]}
DATASETS = ['ct_rate', 'mr_rate_1k', 'amos_mm']
write_json = base.write_json
digest = base.digest


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    """固定划分、模型选择与删除协议，禁止将旧单折网格伪装成五折。"""
    reference = read(R051 / 'protocol.json')
    assert reference['config'] == {'position_warmup': 8, 'descriptor': 'no_current'}
    assert reference['settings'] == base.SETTINGS
    for name, expected in reference['source_sha256'].items():
        assert digest(base.SOURCE / name) == expected, name
    jobs = []
    for worker, folds in WORKERS.items():
        for ds in DATASETS:
            for fold in folds:
                saved = R051 / ds / f'fold_{fold}'
                result = read(saved / 'result.json')
                selection = read(saved / 'frozen_selection.json')
                assert result['status'] == 'completed'
                assert result['protocol_sha256'] == reference['protocol_sha256']
                assert digest(saved / 'best.pt') == selection['best_sha256']
                folder, rows, ids, metadata = base.splits(ds, fold)
                assert ids == read(saved / 'split_ids.json')
                original = next(j for j in reference['jobs'] if j['dataset'] == ds and j['fold'] == fold)
                assert original['split_ids'] == ids
                jobs.append(dict(id=f'{ds}_fold{fold}', dataset=ds, fold=fold,
                                 seed=original['seed'], worker=worker, split_ids=ids,
                                 input_hashes=original['input_hashes'],
                                 r051_checkpoint_sha256=selection['best_sha256']))
    plan = dict(model='Original PE', config=CONFIG, settings=base.SETTINGS,
                jobs=jobs, source_sha256=reference['source_sha256'],
                r051_protocol_sha256=reference['protocol_sha256'],
                training_runner_sha256=digest(ROOT / 'src/scripts/run_r051_table1.py'),
                scope='只补相同训练协议的Original PE；复用现有15个r051权重；不启动自动搜索',
                evaluation=dict(split='test', source_sequence='full_before_deletion',
                                indices='original_acquisition_indices', mask_seed=42,
                                ratios=list(RATIOS), blocks=list(BLOCKS), conditions_per_fold=65,
                                target_instances=64, num_folds=5, std_ddof=1,
                                thresholds='saved_clean_validation_no_retuning'),
                amos_protocol='五个开发验证折训练的模型评估同一个200例固定测试集，非五个互斥测试折',
                position_warmup='r051保留8轮位置warmup；Original PE沿用标准正弦位置注入，不增加新机制',
                comparison_scope='比较r051方案和标准Original PE，不能将差异全部归因于描述符或排除warmup影响',
                exposure='固定既有模型与历史划分，历史开发期间接触过相关数据；不是全新外部验证')
    plan['protocol_sha256'] = hashlib.sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'protocol.json'
    if target.exists():
        assert read(target) == plan, '既有五折协议不同，拒绝混用'
    else:
        write_json(target, plan)
    return plan


def train_control(job, plan, shared):
    """复用逐行相同的训练函数，仅替换位置配置与输出路径。"""
    base.OUT = OUT / 'original_pe'
    base.CONFIG = CONFIG
    base.train_job(job, plan, shared)
    path = base.OUT / job['dataset'] / f"fold_{job['fold']}" / 'result.json'
    result = read(path)
    # 旧通用训练函数的描述字段写死为r051，在此纠正；不改变任何训练或预测结果。
    result.update(model='Original PE', position_scale=1.)
    write_json(path, result)


def records_for(job):
    ds = job['dataset']
    folder, rows, ids, _ = base.splits(ds, job['fold'])
    records, excluded, hashes = {}, [], {}
    for case in tqdm(ids['test'], desc=job['id'] + ' 核验完整删除网格', mininterval=5):
        initial = folder / 'features' / f'{case:04d}.npz'
        with np.load(initial, allow_pickle=False) as z:
            total = int(z['original_count'])
            old_features = z['features'].astype(np.float32)
            old_indices = z['slice_indices'].astype(np.int64)
        try:
            configs = {f'{r:02d}_B{b}': block_sampling(total, r, b, case) for r, b in GRID}
        except ValueError as error:
            excluded.append(dict(case_index=case, source_count=total, reason=str(error)))
            continue
        path = FEATURES / ds / f'{case:04d}.npz'
        with np.load(path, allow_pickle=False) as z:
            assert int(z['original_count']) == total
            indices = z['source_indices'].astype(np.int64)
            features = z['features'].astype(np.float32)
        assert features.shape == (len(indices), 768) and np.isfinite(features).all()
        selections = {}
        for key, obj in configs.items():
            selected = np.asarray(obj['selected_raw_indices'], dtype=np.int64)
            local = np.searchsorted(indices, selected)
            assert np.all(local < len(indices)) and np.array_equal(indices[local], selected), (ds, case, key)
            selections[key] = (selected, local)
        common, ia, ib = np.intersect1d(indices, old_indices, return_indices=True)
        assert len(common) and np.array_equal(features[ia], old_features[ib]), (ds, case, '编码缓存不同')
        tokens = base._encode_text_fields({'watch': rows[case]['findings_masked']}, ('watch',),
                                          max_length=512, vocab_size=8192)
        records[case] = dict(features=features, selections=selections, count=total, tokens=tokens)
        hashes[str(case)] = digest(path)
    assert records and not (set(records) & set(ids['train']) or set(records) & set(ids['validation']))
    output = OUT / 'grids' / ds / f"fold_{job['fold']}"
    write_json(output / 'eligibility.json', dict(test_ids=ids['test'], eligible_ids=sorted(records),
               excluded=excluded, rule='同一测试折的全部65条件共用满足非空非相邻B-block删除的病例'))
    write_json(output / 'feature_hashes.json', hashes)
    return rows, records


def checkpoint_models(job, plan):
    ds, fold = job['dataset'], job['fold']
    models, sources = {}, {}
    for name, root, config, expected in [
        ('r051', R051, {'position_warmup': 8, 'descriptor': 'no_current'}, plan['r051_protocol_sha256']),
        ('original_pe', OUT / 'original_pe', CONFIG, plan['protocol_sha256'])]:
        folder = root / ds / f'fold_{fold}'
        checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        selection = read(folder / 'frozen_selection.json')
        result = read(folder / 'result.json')
        assert checkpoint['config'] == config and checkpoint['protocol_sha256'] == expected
        assert result['protocol_sha256'] == expected
        assert digest(folder / 'best.pt') == selection['best_sha256']
        labels = len(selection['thresholds'])
        model = base.ResearchModel(labels, config)
        model.load_state_dict(checkpoint['model'], strict=True)
        model.epoch = int(checkpoint['epoch']) - 1
        models[name] = model.cuda().eval()
        sources[name] = dict(path=str((folder / 'best.pt').relative_to(ROOT)),
                             sha256=selection['best_sha256'], config=config,
                             best_epoch=checkpoint['epoch'], thresholds=selection['thresholds'],
                             position_scale=min(1., checkpoint['epoch'] / config.get('position_warmup', 1)))
    return models, sources


def evaluate_grid(job, plan, batch_size=16):
    output = OUT / 'grids' / job['dataset'] / f"fold_{job['fold']}"
    path = output / 'grid.json'
    if path.exists():
        saved = read(path)
        assert saved['protocol_sha256'] == plan['protocol_sha256']
        if len(saved['results']) == 65:
            return
    rows, records = records_for(job)
    models, sources = checkpoint_models(job, plan)
    cases = sorted(records)
    header = dict(dataset=job['dataset'], fold=job['fold'], seed=job['seed'],
                  evaluation_split='test', num_cases=len(cases), case_indices=cases,
                  index_protocol='original_acquisition_indices', source_sequence='full_before_deletion',
                  protocol_sha256=plan['protocol_sha256'], checkpoints=sources)
    results = {}
    if path.exists():
        saved = read(path)
        assert all(saved[k] == v for k, v in header.items())
        results = saved['results']
    for ratio, blocks in tqdm(GRID, desc=job['id'] + ' 65条件配对评估', mininterval=5):
        key = f'{ratio:02d}_B{blocks}'
        if key in results:
            continue
        probs = {name: [] for name in models}
        accuracy = {name: [] for name in models}
        labels, knowns, selected_all, counts_all = [], [], [], []
        with torch.inference_mode():
            for start in range(0, len(cases), batch_size):
                subset = cases[start:start + batch_size]
                entries = []
                for case in subset:
                    record = records[case]
                    selected, local = record['selections'][key]
                    row = rows[case]
                    entries.append((record['features'][local], selected, record['count'],
                                    *record['tokens'], np.asarray(row['labels'], np.float32),
                                    np.asarray(row.get('known_mask', [True] * len(row['labels'])), bool), case))
                    padded = np.full(64, -1, dtype=np.int64)
                    padded[:len(selected)] = selected
                    selected_all.append(padded)
                    counts_all.append(record['count'])
                batch = base.collate(entries)
                labels.append(batch['targets'].numpy())
                knowns.append(batch['known'].numpy())
                inputs, _, _ = base.cuda_batch(batch)
                for name, model in models.items():
                    with torch.autocast('cuda', dtype=torch.float16):
                        values = model(**inputs)
                    logits = values['logits'].float()
                    assert torch.isfinite(logits).all()
                    probs[name].append(logits.sigmoid().cpu().numpy())
                    contexts = values.get('apro_context_coordinates')
                    for j, case in enumerate(subset):
                        selected, _ = records[case]['selections'][key]
                        truth = selected.astype(np.float64) / (records[case]['count'] - 1)
                        if name == 'r051':
                            context = contexts[j, :len(selected)].float().cpu().numpy()
                            assert np.isfinite(context).all() and np.all(np.diff(context) >= -1e-7)
                            assert context[-1] - context[0] > 1e-12
                            pred = (context - context[0]) / (context[-1] - context[0])
                        else:
                            pred = np.linspace(0., 1., len(selected))
                        accuracy[name].append(float((np.abs(pred - truth)[1:-1] <= .05).mean()))
        y, known = np.concatenate(labels), np.concatenate(knowns)
        arrays = dict(case_indices=np.asarray(cases), labels=y, known_mask=known,
                      selected_raw_indices=np.stack(selected_all), original_counts=np.asarray(counts_all))
        scores = {}
        for name in models:
            probability = np.concatenate(probs[name])
            thresholds = np.asarray(sources[name]['thresholds'])
            scores[name] = dict(macro_f1=float(base.label_f1(y, probability, known, thresholds).mean()),
                                acc005=float(np.mean(accuracy[name])))
            arrays[name + '_probabilities'] = probability
            arrays[name + '_case_acc005'] = np.asarray(accuracy[name])
            arrays[name + '_thresholds'] = thresholds
        prediction = output / 'predictions' / (key + '.npz')
        prediction.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prediction, **arrays)
        results[key] = dict(ratio=ratio, blocks=blocks, num_cases=len(cases), status='COMPLETE',
                            **scores, delta_f1=scores['r051']['macro_f1'] - scores['original_pe']['macro_f1'],
                            delta_acc005=scores['r051']['acc005'] - scores['original_pe']['acc005'],
                            prediction_file=str(prediction.relative_to(ROOT)), prediction_sha256=digest(prediction))
        write_json(path, dict(**header, results=results, updated=time.time()))
        write_json(OUT / 'workers' / (job['worker'] + '.json'), dict(state='evaluating', pid=os.getpid(),
                   job=job['id'], completed_conditions=len(results), updated=time.time()))
    del models, records, values, logits, inputs
    gc.collect()
    torch.cuda.empty_cache()


def aggregate(plan):
    rows, summaries, completed = [], [], 0
    for ds in base.DATASETS:
        folds = []
        for fold in range(1, 6):
            path = OUT / 'grids' / ds / f'fold_{fold}' / 'grid.json'
            if not path.exists():
                continue
            obj = read(path)
            assert obj['protocol_sha256'] == plan['protocol_sha256']
            if len(obj['results']) == 65:
                completed += 1
                folds.append(obj)
            for r in obj['results'].values():
                rows.append(dict(Dataset=ds, Fold=fold, Seed=obj['seed'], DeletionRatio=r['ratio'],
                                 Blocks=r['blocks'], NumCases=r['num_cases'],
                                 R051_MacroF1=r['r051']['macro_f1'], OriginalPE_MacroF1=r['original_pe']['macro_f1'],
                                 Delta_MacroF1=r['delta_f1'], R051_Acc005=r['r051']['acc005'],
                                 OriginalPE_Acc005=r['original_pe']['acc005'], Delta_Acc005=r['delta_acc005']))
        if len(folds) != 5:
            continue
        for ratio, blocks in GRID:
            key = f'{ratio:02d}_B{blocks}'
            item = dict(dataset=ds, ratio=ratio, blocks=blocks, num_folds=5,
                        case_counts=[obj['num_cases'] for obj in folds])
            for field in ['delta_f1', 'delta_acc005']:
                vals = [obj['results'][key][field] for obj in folds]
                item[field + '_mean'] = statistics.mean(vals)
                item[field + '_std'] = statistics.stdev(vals)
            for name in ['r051', 'original_pe']:
                for field in ['macro_f1', 'acc005']:
                    vals = [obj['results'][key][name][field] for obj in folds]
                    item[name + '_' + field + '_mean'] = statistics.mean(vals)
                    item[name + '_' + field + '_std'] = statistics.stdev(vals)
            summaries.append(item)
    if rows:
        with (OUT / 'grid_folds.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    if summaries:
        with (OUT / 'grid_mean_std.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader(); writer.writerows(summaries)
    summary = dict(status='completed' if completed == 15 else 'running', completed_grids=completed,
                   expected_grids=15, measured_paired_conditions=len(rows), expected_paired_conditions=975,
                   protocol_sha256=plan['protocol_sha256'], std_ddof=1, conditions=summaries,
                   amos_protocol=plan['amos_protocol'], updated=time.time())
    write_json(OUT / 'summary.json', summary)
    print(f'完成网格 {completed}/15；实测配对条件 {len(rows)}/975', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--worker', choices=list(WORKERS))
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cudnn.benchmark = False
    plan = prepare()
    if args.prepare or args.aggregate:
        aggregate(plan)
    if args.worker:
        (OUT / 'workers').mkdir(exist_ok=True)
        lock = (OUT / 'workers' / (args.worker + '.lock')).open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(OUT / 'workers' / (args.worker + '.code.json'), dict(runner_sha256=digest(__file__),
                   inherited_trainer_sha256=digest(ROOT / 'src/scripts/run_r051_table1.py')))
        shared = {}
        for job in plan['jobs']:
            if job['worker'] != args.worker:
                continue
            try:
                if not args.evaluate_only:
                    train_control(job, plan, shared)
                evaluate_grid(job, plan)
            except Exception:
                write_json(OUT / 'workers' / (args.worker + '.json'), dict(state='failed', pid=os.getpid(),
                           job=job['id'], error=traceback.format_exc(), updated=time.time()))
                raise
        write_json(OUT / 'workers' / (args.worker + '.json'),
                   dict(state='completed', pid=os.getpid(), updated=time.time()))
    if not (args.prepare or args.aggregate or args.worker):
        parser.error('请指定准备、汇总或执行worker')


if __name__ == '__main__':
    main()
