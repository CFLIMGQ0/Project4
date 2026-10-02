#!/usr/bin/env python3
"""固定三个ACPE进阶版，完成表1测试及图2五折完整删除网格。"""
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
import shutil
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/acpe_advanced_table1_figure2_20260930'
ARCHIVE = ROOT / 'outputs/acpe_advanced_versions'
os.environ['PROJECT4_ROOT'] = str(ROOT)
sys.path.insert(0, str(ROOT / 'src/scripts'))
sys.path.insert(0, str(OUT))
import run_r051_table1 as base
from frozen_acpe.model import Model
from frozen_acpe.train import StructuredBags
from paper_block_deletion import block_sampling
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

DATASETS = ['ct_rate', 'mr_rate_1k', 'amos_mm']
VERSIONS = ['original_pe', 'advanced_1', 'advanced_2', 'advanced_3']
GRID = [(0, 'B1')] + [(r, p) for r in range(10, 81, 10)
                               for p in [*[f'B{b}' for b in range(1, 8)], 'Rand']]
TRAIN_CONDITIONS = [(40, 1), (60, 1), (80, 1), (80, 4), (80, 8)]
ORIGINAL_BAGS = base.Bags
write = base.write_json
digest = base.digest


def read(path):
    return json.loads(Path(path).read_text())


def identity(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def selection(total, ratio, pattern, case):
    """连续块沿用原掩码；Rand实际均匀随机选删，不将八块伪称随机。"""
    if pattern != 'Rand':
        return np.asarray(block_sampling(total, ratio, int(pattern[1:]), case)['selected_raw_indices'])
    remove = math.floor(total * ratio / 100)
    if total - remove < 3:
        raise ValueError('随机删除后不足三张图像')
    rng = np.random.default_rng(np.random.SeedSequence([42, case, ratio, 20260930]))
    keep = np.ones(total, dtype=bool)
    keep[rng.choice(total, size=remove, replace=False)] = False
    remaining = np.flatnonzero(keep)
    slots = np.rint(np.linspace(0, len(remaining)-1, min(64, len(remaining)))).astype(np.int64)
    assert int((~keep).sum()) == remove
    return remaining[slots]


def candidates():
    registry = read(ARCHIVE / 'registry.json')
    values = {'original_pe': read(ARCHIVE / 'matched_original_pe/candidate.json')}
    for version in registry['versions']:
        values[version['key']] = read(ARCHIVE / version['key'] / 'candidate.json')
    return values


def archived_run(version, dataset, fold, definitions):
    directory = 'matched_original_pe' if version == 'original_pe' else version
    return ARCHIVE / directory / 'runs' / f"{definitions[version]['id']}_{dataset}_f{fold}"


def prepare():
    path = OUT / 'protocol.json'
    if path.exists():
        plan = read(path)
        assert plan['runner_sha256'] == digest(__file__), '协议冻结后禁止修改评估实现'
        return plan
    definitions = candidates()
    research = read(ARCHIVE / 'shared/protocol.json')
    for name, expected in research['baseline_source_hashes'].items():
        assert digest(base.SOURCE / name) == expected, name
    for name, expected in research['source_hashes'].items():
        assert digest(OUT / 'frozen_acpe' / name) == expected, name
    jobs = []
    for ds in ['amos_mm', 'ct_rate', 'mr_rate_1k']:
        for fold in range(1, 6):
            folder, rows, ids, metadata = base.splits(ds, fold)
            reuse = {}
            for version in VERSIONS:
                source = archived_run(version, ds, fold, definitions)
                old = read(source / 'protocol.json')
                compatible = (old['split']['train'] == ids['train'] and
                              old['split']['validation'] == ids['validation'] and
                              old['excluded_test'] == ids['test'] and
                              old['split']['seed'] == 42 + 100*fold and
                              old['spec']['config'] == definitions[version]['config'])
                if compatible:
                    result = read(source / 'result.json')
                    assert digest(source / 'best.pt') == result['best_sha256']
                    reuse[version] = {'path': str(source.relative_to(ROOT)),
                                      'checkpoint_sha256': result['best_sha256']}
            jobs.append({'id': f'{ds}_fold{fold}', 'dataset': ds, 'fold': fold,
                         'seed': 42 + 100*fold, 'split_ids': ids, 'reuse': reuse,
                         'input_hashes': {str(p.relative_to(ROOT)): digest(p) for p in metadata}})
    plan = {'versions': definitions, 'datasets': DATASETS, 'jobs': jobs,
            'settings': base.SETTINGS, 'grid': GRID,
            'training_augmentation': '保持候选原有训练增强：概率0.5采用40/60/80%的B1、80%的B4/B8原序列删除视图',
            'random_deletion': '独立固定种子均匀选删原序列图像；确切数量floor(N*ratio/100)，可删除端点',
            'baseline': 'Original PE使用与进阶版相同的结构化缺失训练',
            'test_protocol': 'CT/MR五个互斥测试折；AMOS五个开发验证折训练后评估同一200例固定测试集',
            'selection': '最低验证ASL选权重；干净验证集逐标签F1选阈值，随后冻结',
            'clean_vs_grid': '表1测试完整测试折；图2只用全部65条件共同可评估病例；二者分别汇报',
            'coordinate_metric': '内部图像的端点归一化预测坐标与原始索引/(N-1)误差不超过0.05，先病例内后病例间平均',
            'display': '三个版本各一组2x3曲面，列CT-RATE/MR-RATE-1K/AMOS-MM，上F1下Acc@0.05；temp1.png合为6x3',
            'source_protocol_sha256': research['sha256'],
            'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                [Path(__file__), ROOT/'src/scripts/run_r051_table1.py',
                 ROOT/'src/scripts/paper_block_deletion.py',
                 *sorted((OUT/'frozen_acpe').glob('*.py')),
                 *[base.SOURCE/n for n in research['baseline_source_hashes']]]},
            'runner_sha256': digest(__file__), 'gpu_policy': '仅GPU2/3，每卡最多两个项目GPU任务',
            'note': '内部研究评估，不修改论文，不将开发验证预测或旧图的人为偏移作为测试结果'}
    plan['protocol_sha256'] = identity(plan)
    write(path, plan)
    external = [{'kind': 'external', 'key': 'advanced_benchmark_'+j['id'],
                 'protocol_sha256': plan['protocol_sha256'],
                 'completion_path': str(OUT/'jobs'/j['id']/'completed.json'),
                 'arguments': [str(Path(__file__).resolve()), '--job', j['id']]} for j in jobs]
    write(OUT/'jobs.json', external)
    reusable = sum(len(j['reuse']) for j in jobs)
    print(f'协议已固定：复用{reusable}/60份权重，其余{60-reusable}份需按表1划分训练。', flush=True)
    return plan


def get_features(ds, case, folder, required, supplement=False):
    """复用冻结编码结果；缺失随机条件切片只补提，不覆写旧缓存。"""
    with np.load(folder/'features'/f'{case:04d}.npz', allow_pickle=False) as z:
        total = int(z['original_count'])
        initial_indices = z['slice_indices'].astype(np.int64)
        initial = z['features'].astype(np.float32)
    required = np.asarray(sorted(set(map(int, required))), dtype=np.int64)
    values = {int(i): f for i, f in zip(initial_indices, initial)}
    sources = []
    paths = [ROOT/'outputs/paper_results/figure1_features'/ds/f'{case:04d}.npz',
             ROOT/'outputs/acpe_fivefold_research_20260929/features'/ds/f'{case:04d}.npz',
             OUT/'random_features'/ds/f'{case:04d}.npz']
    for path in paths:
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as z:
            assert int(z['original_count']) == total
            ix = z['source_indices'].astype(np.int64)
            ff = z['features'].astype(np.float32)
        assert ff.shape == (len(ix), 768) and np.isfinite(ff).all()
        common, ia, ib = np.intersect1d(ix, initial_indices, return_indices=True)
        if len(common):
            assert np.array_equal(ff[ia], initial[ib]), (ds, case, '缓存编码不一致')
        for i, feature in zip(ix, ff):
            if int(i) not in values:
                values[int(i)] = feature
        sources.append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path)})
        if all(int(i) in values for i in required):
            break
    missing = np.asarray([i for i in required if int(i) not in values], dtype=np.int64)
    if len(missing):
        assert supplement, (ds, case, '训练增强缓存缺失', len(missing))
        import prepare_figure1_features as builder
        from prepare_public_block3_features import FrozenEncoder, prepare_amos, prepare_mr
        # 只在需要补提的病例中实例化GPU编码器，提取后释放。
        encoder = FrozenEncoder('cuda:0')
        rows = {r['case_index']:r for r in read(folder/'samples.json')}
        if ds == 'ct_rate':
            count, fresh = builder.encode_ct_slices(rows[case], missing, encoder, ds)
        elif ds == 'amos_mm':
            count, fresh = prepare_amos(rows[case], missing, encoder)
        else:
            count, fresh = prepare_mr(rows[case], missing, encoder)
        assert count == total and fresh.shape == (len(missing), 768)
        path = paths[-1]
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_suffix('.lock').open('a')
        with lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            previous = {}
            if path.exists():
                with np.load(path, allow_pickle=False) as z:
                    previous = {int(i):f for i,f in zip(z['source_indices'],z['features'])}
            previous.update({int(i):f for i,f in zip(missing,fresh)})
            ix = np.asarray(sorted(previous),dtype=np.int64)
            temp = path.with_name(path.stem+f'.{os.getpid()}.tmp.npz')
            np.savez_compressed(temp,source_indices=ix,features=np.stack([previous[int(i)] for i in ix]),original_count=total)
            temp.replace(path)
        values.update({int(i):f for i,f in zip(missing,fresh)})
        sources.append({'path':str(path.relative_to(ROOT)),'sha256':digest(path)})
        del encoder
        gc.collect();torch.cuda.empty_cache()
    features = np.stack([values[int(i)] for i in required])
    assert np.isfinite(features).all()
    return total, required, features, sources


def training_views(ds, folder, ids):
    full = {}
    for case in tqdm(ids, desc=ds+' 训练增强缓存', mininterval=10):
        with np.load(folder/'features'/f'{case:04d}.npz',allow_pickle=False) as z:
            total = int(z['original_count'])
        try:
            # 沿用候选训练时的共同可评估病例定义。
            for r in range(10,81,10):
                for b in range(1,9):
                    block_sampling(total,r,b,case)
            selections = {f'd{r:02d}_b{b}':np.asarray(block_sampling(total,r,b,case)['selected_raw_indices'])
                          for r,b in TRAIN_CONDITIONS}
        except ValueError:
            continue
        union = np.unique(np.concatenate(list(selections.values())))
        count, ix, features, _ = get_features(ds,case,folder,union)
        full[case] = {'count':count,'features':features,
                      'selections':{k:(v,np.searchsorted(ix,v)) for k,v in selections.items()}}
    return full


def reuse_weight(job, plan, version, shared):
    ds,fold = job['dataset'],job['fold']
    output = OUT/'models'/version/ds/f'fold_{fold}'
    output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists():
        assert read(output/'result.json')['protocol_sha256'] == plan['protocol_sha256']
        return
    source = ROOT/job['reuse'][version]['path']
    assert digest(source/'best.pt') == job['reuse'][version]['checkpoint_sha256']
    old = read(source/'result.json')
    config = plan['versions'][version]['config']
    saved = torch.load(source/'best.pt',map_location='cpu',weights_only=True)
    assert saved['config'] == config
    folder,rows,ids,_ = base.splits(ds,fold)
    model = Model(len(rows[ids['train'][0]]['labels']),config).cuda().eval()
    model.load_state_dict(saved['model'],strict=True);model.epoch=saved['epoch']-1
    cache,hashes = shared.setdefault(ds,({},{}))
    base.load_cache(folder,rows,ids['validation']+ids['test'],cache,hashes)
    loader = lambda values:DataLoader(ORIGINAL_BAGS(rows,values,cache,config),batch_size=16,num_workers=0,collate_fn=base.collate)
    val=base.evaluate(model,loader(ids['validation']));thresholds=base.tune(val)
    expected = np.load(source/'validation_predictions.npz',allow_pickle=False)
    assert np.array_equal(val['ids'],expected['clean_ids'])
    assert np.allclose(val['p'],expected['clean_p'],rtol=3e-3,atol=3e-4), '复用权重前向校验失败'
    expected.close()
    base.save_checkpoint(output/'best.pt',{'model':saved['model'],'epoch':saved['epoch'],
                         'config':config,'protocol_sha256':plan['protocol_sha256'],
                         'source_checkpoint_sha256':job['reuse'][version]['checkpoint_sha256']})
    write(output/'split_ids.json',ids)
    write(output/'protocol.json',{'job':job,'config':config,'settings':base.SETTINGS,
          'protocol_sha256':plan['protocol_sha256'],'reused_from':str(source.relative_to(ROOT))})
    write(output/'frozen_selection.json',{'best_epoch':saved['epoch'],'thresholds':thresholds.tolist(),
          'validation_metrics':base.metrics(val,thresholds),'best_sha256':digest(output/'best.pt')})
    np.savez_compressed(output/'validation_predictions.npz',**{k:val[k] for k in ('y','p','known','ids')},thresholds=thresholds)
    test=base.evaluate(model,loader(ids['test']))
    np.savez_compressed(output/'test_predictions.npz',**{k:test[k] for k in ('y','p','known','ids')},thresholds=thresholds)
    write(output/'result.json',{'status':'completed','dataset':ds,'fold':fold,'version':version,
          'seed':job['seed'],'best_epoch':saved['epoch'],'thresholds':thresholds.tolist(),
          'protocol_sha256':plan['protocol_sha256'],'reused_archive':True,**base.metrics(test,thresholds)})
    write(output/'cache_hashes.json',hashes)
    del model,saved,test,val
    gc.collect();torch.cuda.empty_cache()


def grid_records(job):
    ds,fold=job['dataset'],job['fold']
    folder,rows,ids,_=base.splits(ds,fold)
    records,excluded,sources={},[],{}
    for case in tqdm(ids['test'],desc=job['id']+' 完整65条件缓存',mininterval=10):
        with np.load(folder/'features'/f'{case:04d}.npz',allow_pickle=False) as z:
            total=int(z['original_count'])
        try:
            selected={f'{r:02d}_{pattern}':selection(total,r,pattern,case) for r,pattern in GRID}
        except ValueError as exc:
            excluded.append({'case_index':case,'count':total,'reason':str(exc)});continue
        union=np.unique(np.concatenate(list(selected.values())))
        count,ix,features,files=get_features(ds,case,folder,union,supplement=True)
        records[case]={'count':count,'features':features,
            'selections':{k:(v,np.searchsorted(ix,v)) for k,v in selected.items()},
            'tokens':base._encode_text_fields({'watch':rows[case]['findings_masked']},('watch',),max_length=512,vocab_size=8192)}
        sources[str(case)]=files
    assert records and not set(records)&set(ids['train']+ids['validation'])
    write(OUT/'jobs'/job['id']/'eligibility.json',{'test':ids['test'],'eligible':sorted(records),'excluded':excluded})
    write(OUT/'jobs'/job['id']/'grid_feature_sources.json',sources)
    return rows,records


def evaluate_grid(job,plan,version,rows,records):
    ds,fold=job['dataset'],job['fold'];output=OUT/'grids'/version/ds/f'fold_{fold}'
    path=output/'grid.json';results={}
    if path.exists():
        old=read(path);assert old['protocol_sha256']==plan['protocol_sha256'];results=old['results']
        if len(results)==65:return
    source=OUT/'models'/version/ds/f'fold_{fold}'
    saved=torch.load(source/'best.pt',map_location='cpu',weights_only=True)
    frozen=read(source/'frozen_selection.json');assert digest(source/'best.pt')==frozen['best_sha256']
    config=plan['versions'][version]['config'];assert saved['config']==config
    thresholds=np.asarray(frozen['thresholds']);cases=sorted(records)
    model=Model(len(thresholds),config).cuda().eval()
    model.load_state_dict(saved['model'],strict=True);model.epoch=saved['epoch']-1
    for ratio,pattern in tqdm(GRID,desc=f'{job["id"]} {version} 65条件',mininterval=10):
        key=f'{ratio:02d}_{pattern}'
        if key in results:continue
        probabilities,accuracies,labels,knowns,selected_all=[],[],[],[],[]
        with torch.inference_mode():
            for start in range(0,len(cases),16):
                subset=cases[start:start+16];entries=[]
                for case in subset:
                    record=records[case];indices,slots=record['selections'][key];row=rows[case]
                    entries.append((record['features'][slots],indices,record['count'],*record['tokens'],
                         np.asarray(row['labels'],np.float32),np.asarray(row.get('known_mask',[True]*len(row['labels'])),bool),case))
                    padded=np.full(64,-1,dtype=np.int64);padded[:len(indices)]=indices;selected_all.append(padded)
                batch=base.collate(entries);inputs,_,_=base.cuda_batch(batch)
                with torch.autocast('cuda',dtype=torch.float16):values=model(**inputs)
                logits=values['logits'].float();assert torch.isfinite(logits).all()
                probabilities.append(logits.sigmoid().cpu().numpy())
                labels.append(batch['targets'].numpy());knowns.append(batch['known'].numpy())
                for j,case in enumerate(subset):
                    indices,_=records[case]['selections'][key]
                    truth=indices.astype(np.float64)/max(1,records[case]['count']-1)
                    if version=='original_pe':pred=np.linspace(0.,1.,len(indices))
                    else:
                        coord=values['apro_context_coordinates'][j,:len(indices)].float().cpu().numpy()
                        assert np.isfinite(coord).all() and np.all(np.diff(coord)>=-1e-7)
                        assert coord[-1]-coord[0]>1e-12
                        pred=(coord-coord[0])/(coord[-1]-coord[0])
                    accuracies.append(float((np.abs(pred-truth)[1:-1]<=.05).mean()))
        y,p,known=np.concatenate(labels),np.concatenate(probabilities),np.concatenate(knowns)
        prediction=output/'predictions'/f'{key}.npz';prediction.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(prediction,case_indices=np.asarray(cases),labels=y,known_mask=known,
             probabilities=p,case_acc005=np.asarray(accuracies),thresholds=thresholds,
             selected_raw_indices=np.stack(selected_all))
        results[key]={'ratio':ratio,'pattern':pattern,'n':len(cases),
                     'macro_f1':float(base.label_f1(y,p,known,thresholds).mean()),
                     'macro_f1_fixed_0_5':float(base.label_f1(y,p,known,.5).mean()),
                     'acc005':float(np.mean(accuracies)),'prediction_sha256':digest(prediction)}
        write(path,{'dataset':ds,'fold':fold,'version':version,'evaluation_split':'test',
             'protocol_sha256':plan['protocol_sha256'],'case_indices':cases,'checkpoint_sha256':frozen['best_sha256'],
             'results':results,'updated':time.time()})
    del model,saved,values,logits,inputs
    gc.collect();torch.cuda.empty_cache()


def aggregate(plan):
    with (OUT/'aggregate.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        table,grids,fold_rows=[],[],[]
        for ds in DATASETS:
            for version in VERSIONS:
                vals=[]
                for fold in range(1,6):
                    p=OUT/'models'/version/ds/f'fold_{fold}'/'result.json'
                    if p.exists():
                        r=read(p);assert r['protocol_sha256']==plan['protocol_sha256'];vals.append(r)
                item={'dataset':ds,'version':version,'completed_folds':len(vals)}
                if len(vals)==5:
                    for metric in ['macro_f1','macro_f1_fixed_0_5']:
                        a=[r[metric] for r in vals];item[metric]=statistics.mean(a);item[metric+'_std']=statistics.stdev(a)
                table.append(item)
            for fold in range(1,6):
                paths={v:OUT/'grids'/v/ds/f'fold_{fold}'/'grid.json' for v in VERSIONS}
                if not all(p.exists() for p in paths.values()):continue
                objects={v:read(p) for v,p in paths.items()}
                assert all(o['case_indices']==objects['original_pe']['case_indices'] for o in objects.values())
                common=set.intersection(*[set(o['results']) for o in objects.values()])
                for key in sorted(common):
                    baseline=objects['original_pe']['results'][key]
                    for version in VERSIONS[1:]:
                        r=objects[version]['results'][key]
                        fold_rows.append({'dataset':ds,'version':version,'fold':fold,'ratio':r['ratio'],
                          'pattern':r['pattern'],'n':r['n'],'macro_f1':r['macro_f1'],'original_f1':baseline['macro_f1'],
                          'acc005':r['acc005'],'original_acc005':baseline['acc005'],
                          'delta_f1':r['macro_f1']-baseline['macro_f1'],'delta_acc005':r['acc005']-baseline['acc005']})
            for version in VERSIONS[1:]:
                for ratio,pattern in GRID:
                    values=[r for r in fold_rows if r['dataset']==ds and r['version']==version and r['ratio']==ratio and r['pattern']==pattern]
                    if len(values)!=5:continue
                    item={'dataset':ds,'version':version,'ratio':ratio,'pattern':pattern,'folds':5}
                    for metric in ['macro_f1','original_f1','acc005','original_acc005','delta_f1','delta_acc005']:
                        a=[r[metric] for r in values];item[metric+'_mean']=statistics.mean(a);item[metric+'_std']=statistics.stdev(a)
                    grids.append(item)
        finished=sum((OUT/'jobs'/j['id']/'completed.json').exists() for j in plan['jobs'])
        summary={'status':'complete' if finished==15 and len(grids)==585 else 'running',
                 'completed_jobs':finished,'expected_jobs':15,'completed_grid_summaries':len(grids),
                 'expected_grid_summaries':585,'protocol_sha256':plan['protocol_sha256'],
                 'table1':table,'grid':grids,'updated':time.time()}
        write(OUT/'summary.json',summary)
        for filename,rows in [('table1_results.csv',table),('grid_folds.csv',fold_rows),('grid_mean_std.csv',grids)]:
            if rows:
                fields=list(dict.fromkeys(k for r in rows for k in r))
                with (OUT/filename).open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        lines=['# ACPE进阶版：表1五折测试结果','',
               '阈值由干净验证集确定；数值为F1百分比的五折均值±样本标准差。',
               'CT/MR使用五个互斥测试折；AMOS使用五次开发折训练及同一个固定测试集。','',
               '| 版本 | CT-RATE | MR-RATE-1K | AMOS-MM |','|---|---:|---:|---:|']
        for version in VERSIONS:
            cells=[]
            for ds in DATASETS:
                r=next(x for x in table if x['dataset']==ds and x['version']==version)
                cells.append(f"{100*r['macro_f1']:.2f} ± {100*r['macro_f1_std']:.2f}" if r['completed_folds']==5 else f"待完成（{r['completed_folds']}/5）")
            name='Original PE（匹配训练）' if version=='original_pe' else 'ACPE进阶版'+version[-1]
            lines.append('| '+name+' | '+' | '.join(cells)+' |')
        (OUT/'table1_results.md').write_text('\n'.join(lines)+'\n')
        print(f'已完成{finished}/15组，五折曲面条件{len(grids)}/585。',flush=True)
        return summary


def run_job(job,plan):
    output=OUT/'jobs'/job['id'];output.mkdir(parents=True,exist_ok=True)
    with (output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'completed.json').exists():return
        for path,expected in plan['source_sha256'].items():assert digest(ROOT/path)==expected,path
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.backends.mha.set_fastpath_enabled(False);torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(.45)
        shared={};views=None;ds,fold=job['dataset'],job['fold']
        folder,rows,ids,metadata=base.splits(ds,fold);assert ids==job['split_ids']
        assert {str(p.relative_to(ROOT)):digest(p) for p in metadata}==job['input_hashes']
        for version in VERSIONS:
            write(output/'status.json',{'phase':'clean_test_or_training','version':version,'updated':time.time(),'pid':os.getpid()})
            if version in job['reuse']:
                reuse_weight(job,plan,version,shared)
            else:
                if views is None:views=training_views(ds,folder,ids['train'])
                def bags(rows,indices,cache,config,training=False):
                    if training and config.get('structured_training'):
                        return StructuredBags(rows,indices,cache,config,training=True,full=views)
                    return ORIGINAL_BAGS(rows,indices,cache,config,training=training)
                base.Bags=bags;base.ResearchModel=Model;base.CONFIG=plan['versions'][version]['config']
                base.OUT=OUT/'models'/version
                current={**job,'worker':version+'_'+job['id']}
                base.train_job(current,plan,shared)
                result_path=base.OUT/ds/f'fold_{fold}'/'result.json'
                result=read(result_path)
                result.update(model='ALM-MIL '+version,version=version,
                              candidate_id=plan['versions'][version]['id'],reused_archive=False)
                write(result_path,result)
            aggregate(plan)
        del views,shared
        gc.collect();torch.cuda.empty_cache()
        rows,records=grid_records(job)
        for version in VERSIONS:
            write(output/'status.json',{'phase':'full_grid','version':version,'updated':time.time(),'pid':os.getpid()})
            evaluate_grid(job,plan,version,rows,records)
        write(output/'completed.json',{'protocol_sha256':plan['protocol_sha256'],'job':job['id'],'updated':time.time()})
        write(output/'status.json',{'phase':'complete','updated':time.time(),'pid':os.getpid()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true')
    p.add_argument('--aggregate',action='store_true');p.add_argument('--job');args=p.parse_args()
    plan=prepare()
    if args.job:
        job=next(j for j in plan['jobs'] if j['id']==args.job)
        try:run_job(job,plan)
        except Exception:
            write(OUT/'jobs'/job['id']/'failure.json',{'error':traceback.format_exc(),'updated':time.time()});raise
    summary=aggregate(plan)
    if summary['status']=='complete':
        subprocess.run([sys.executable,str(ROOT/'src/scripts/plot_acpe_advanced_benchmark.py')],check=True)


if __name__=='__main__':main()
