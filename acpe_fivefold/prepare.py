"""冻结五折开发协议；准备原序列删除后的特征，不读取保留测试图像。"""
import argparse
import collections
import shutil
import warnings
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from .common import ROOT, OUT, BASE, DATASETS, CONDITIONS, folder, read, write, digest, identity, key
from .plan import BREADTH, MAX_CANDIDATES
from scripts.paper_block_deletion import block_sampling


def split_dataset(ds):
    root = folder(ds)
    rows = {r['case_index']: r for r in read(root / 'samples.json')}
    if ds == 'amos_mm':
        obj = read(root / 'splits.json')
        exclusions = {r['case_index'] for r in read(root / 'image_exclusions.json')['excluded']}
        development = set(obj['development']) - exclusions
        heldout = set(obj['test'])
        folds = [sorted(set(f)-exclusions) for f in obj['validation_folds']]
        patient = {i: rows[i]['scan_id'] for i in rows}
        files = ['samples.json', 'splits.json', 'image_exclusions.json']
    else:
        old = read(root / 'patient_folds.json')['folds']
        heldout = set(old[0])
        development = set().union(*map(set, old[1:]))
        patient_key = 'patient_id' if ds == 'ct_rate' else 'patient_uid'
        patient = {i: rows[i][patient_key] for i in rows}
        indices = sorted(development)
        labels = [sum(int(v) << j for j, v in enumerate(rows[i]['labels'])) for i in indices]
        groups = [patient[i] for i in indices]
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260929)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            folds = [[indices[j] for j in validation] for _, validation in
                     splitter.split(np.zeros(len(indices)), labels, groups)]
        files = ['samples.json', 'patient_folds.json']
    assert len(folds) == 5 and set().union(*map(set, folds)) == development
    assert sum(map(len, folds)) == len(development) and not development & heldout
    assert not {patient[i] for i in development} & {patient[i] for i in heldout}
    result = []
    for fold, values in enumerate(folds, 1):
        train = sorted(development - set(values))
        assert not {patient[i] for i in train} & {patient[i] for i in values}
        result.append({'fold': fold, 'seed': 42+100*fold, 'train': train, 'validation': sorted(values),
                       'validation_positive_counts': np.asarray([rows[i]['labels'] for i in values]).sum(0).tolist()})
    return {'development': sorted(development), 'excluded_test': sorted(heldout), 'folds': result,
            'input_hashes': {name: digest(root/name) for name in files}}


def prepare():
    if (OUT / 'protocol.json').exists():
        return read(OUT / 'protocol.json')
    OUT.mkdir(parents=True, exist_ok=True)
    splits = {ds: split_dataset(ds) for ds in DATASETS}
    records = {}
    for ds, split in splits.items():
        entries = {}
        for case in tqdm(split['development'], desc=ds+' 开发集删除清单', mininterval=5):
            with np.load(folder(ds)/'features'/f'{case:04d}.npz', allow_pickle=False) as z:
                total = int(z['original_count'])
            try:
                # 与后续完整65条件保持同一病例集合，不随研究子网格改变纳入对象。
                for ratio in range(10,81,10):
                    for blocks in range(1,9):
                        block_sampling(total,ratio,blocks,case)
                configs = {key(r,b): block_sampling(total,r,b,case) for r,b in CONDITIONS}
            except ValueError as error:
                entries[str(case)] = {'eligible': False, 'source_count': total, 'reason': str(error)}
                continue
            entries[str(case)] = {'eligible': True, 'source_count': total,
                                  'selections': {k:v['selected_raw_indices'] for k,v in configs.items()}}
        records[ds] = entries
    write(OUT/'deletion_manifest.json', records)
    code_files = ['auto_research/model.py', 'auto_research/train.py', 'exp_8/models.py',
                  'training/data.py', 'training/losses.py', 'scripts/paper_block_deletion.py']
    protocol = {'version': 1, 'datasets': splits, 'epochs': 30, 'batch_size': 16, 'lr': .0002,
                'weight_decay': .02, 'instance_dropout': .25, 'lqd_weight': .01,
                'selection': 'minimum_validation_masked_ASL',
                'primary_metric': 'validation_macro_F1_fixed_threshold_0.5',
                'secondary_metric': 'clean_validation_tuned_thresholds_F1; exploratory_only',
                'deletion_conditions': list(map(list, CONDITIONS)), 'deletion_seed': 42,
                'deletion_source': 'full_original_sequence_before_uniform_sampling_up_to_64',
                'eligibility': '全部65条件均可构造的开发病例，13条件共用同一集合；正常F1另报完整开发验证折',
                'deletion_manifest_sha256': digest(OUT/'deletion_manifest.json'),
                'screening': '每方案三个数据集各五折，15个训练全部完成才排名；同折配对Original PE',
                'ranking': '0.25*mean(clean_delta)+0.75*mean(severe_delta)-0.1*mean(std_fold_severe_delta)',
                'severe_conditions': [key(r,b) for r in (60,80) for b in (1,4)],
                'trend_diagnostics': '逐数据集报告各删除率和B的配对差值，以及80%-20%、B1-B8的优势变化；不预设单调结果',
                'admissible': '至少2个数据集严重缺失F1提高且各至少3/5折提高、平均提高、各数据集正常F1下降不超过0.5个百分点',
                'max_candidates': MAX_CANDIDATES,
                'test_policy': '本轮不读取测试图像或预测；历史已接触这些测试数据，不能宣称从未使用的外部测试集',
                'split_policy': 'CT/MR保留旧fold0，余下患者重新分成5个开发折；AMOS沿用5个开发折并隔离200例测试',
                'full_grid_policy': '研究阶段13个预注册条件；最终候选另行确认完整65条件，本轮不自动接触测试集',
                'gpu_policy': '按最新指令先只用204主机GPU0，每GPU最多一个本轮CUDA进程，不终止其它用户程序',
                'baseline_source_hashes': {f:digest(BASE/f) for f in code_files},
                'feature_encoder_hash': digest(ROOT/'src/scripts/prepare_public_block3_features.py'),
                'source_hashes': {p.name:digest(p) for p in sorted((ROOT/'src/acpe_fivefold').glob('*.py'))}}
    protocol['sha256'] = identity(protocol)
    write(OUT/'protocol.json', protocol)
    candidates = [{'id': f'c{i:03d}_{name}', 'name': name, 'config': config, 'hypothesis': hypothesis,
                   'kind': 'breadth', 'parent': None} for i,(name,config,hypothesis) in enumerate(BREADTH)]
    assert len({identity(c['config']) for c in candidates}) == len(candidates)
    write(OUT/'candidates.json', candidates)
    snapshot = OUT/'source'
    snapshot.mkdir(exist_ok=True)
    for p in (ROOT/'src/acpe_fivefold').glob('*.py'):
        shutil.copy2(p, snapshot/p.name)
    return protocol


def make_features(encode=False):
    manifest = read(OUT/'deletion_manifest.json')
    encoder = None
    missing = []
    for ds in DATASETS:
        rows = {r['case_index']:r for r in read(folder(ds)/'samples.json')}
        completed = 0
        for textcase, obj in tqdm(manifest[ds].items(), desc=ds+' 删除特征', mininterval=5):
            if not obj['eligible']:
                continue
            case = int(textcase)
            dest = OUT/'features'/ds/f'{case:04d}.npz'
            union = np.asarray(sorted(set().union(*map(set, obj['selections'].values()))), dtype=np.int64)
            if dest.exists():
                with np.load(dest, allow_pickle=False) as z:
                    assert np.array_equal(z['source_indices'], union)
                    assert int(z['original_count']) == obj['source_count']
                completed += 1
                continue
            available = {}
            sources = [folder(ds)/'features'/f'{case:04d}.npz',
                       ROOT/'outputs/paper_results/figure1_features'/ds/f'{case:04d}.npz',
                       ROOT/'outputs/r051_3d_20260928/features'/ds/f'{case:04d}.npz']
            hashes = {}
            for source in sources:
                if not source.exists():
                    continue
                hashes[str(source.relative_to(ROOT))] = digest(source)
                with np.load(source, allow_pickle=False) as z:
                    assert int(z['original_count']) == obj['source_count']
                    ix = z['source_indices'] if 'source_indices' in z else z['slice_indices']
                    for index, vector in zip(ix, z['features']):
                        if int(index) not in union:
                            continue
                        if int(index) in available:
                            assert np.array_equal(available[int(index)], vector), (ds,case,int(index))
                        available[int(index)] = vector.astype(np.float32)
            needed = np.asarray([i for i in union if int(i) not in available], dtype=np.int64)
            if len(needed) and not encode:
                missing.append({'dataset': ds, 'case': case, 'slices': len(needed)})
                continue
            if len(needed):
                assert ds == 'amos_mm', 'CT和MR预期可完全复用既有完整网格缓存'
                from scripts.prepare_public_block3_features import FrozenEncoder, prepare_amos
                if encoder is None:
                    encoder = FrozenEncoder('cuda:0')
                count, values = prepare_amos(rows[case], needed, encoder)
                assert count == obj['source_count']
                available.update({int(i):v for i,v in zip(needed,values)})
            values = np.stack([available[int(i)] for i in union])
            assert values.shape == (len(union),768) and np.isfinite(values).all()
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix('.tmp.npz')
            np.savez_compressed(temp, features=values, source_indices=union, original_count=obj['source_count'])
            temp.replace(dest)
            write(dest.with_suffix('.json'), {'source_hashes': hashes, 'new_slices': len(needed),
                                              'feature_sha256': digest(dest)})
            completed += 1
        expected = sum(x['eligible'] for x in manifest[ds].values())
        if completed == expected:
            write(OUT/'features'/ds/'ready.json', {'cases': completed, 'manifest_sha256': digest(OUT/'deletion_manifest.json')})
    write(OUT/'feature_preparation.json', {'missing': missing, 'mode': 'encode' if encode else 'reuse'})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--encode', action='store_true')
    args = p.parse_args()
    prepare()
    make_features(args.encode)


if __name__ == '__main__':
    main()
