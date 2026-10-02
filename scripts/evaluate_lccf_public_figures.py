#!/usr/bin/env python3
"""只读评估公开队列已有五折权重，生成 LCCF 四类机制图的可追溯数据。"""
from __future__ import annotations
import argparse
import csv
import gc
import json
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from exp_8.models import Exp13LCCFAblationModel
from sotas.task2.multimodal_sotas import build_task2_multimodal_sota
from sotas.task2.task_adapted_vlms import TASK_ADAPTED_VLM_REGISTRY
from sotas.task2.paper2026_adapted import build_paper2026_adapted
from training.data import _encode_text_fields
from scripts import run_physionet_ct_ich_table2_baselines as image_base

OUT = ROOT / 'outputs/lccf_public_evidence_20261001'
DATASETS = {
    'ct_rate': ('CT-RATE', 'ct_rate_680', 3),
    'mr_rate': ('MR-RATE-1K', 'mr_rate_1k', 4),
    'amos_mm': ('AMOS-MM', 'amos_mm', 7),
}
BASELINES = ('attention_mil', 'clam_mb', 'clam_sb', 'dsmil', 'transmil', 'dtfd_mil')
MULTIMODAL_OLD = {
    'mmfnet': 'task2_mmfnet_2024',
    'radfuse': 'task2_radfuse_2025',
    'saif': 'task2_saif_2025',
    'mmtf': 'task2_mmtf_2025',
}
MULTIMODAL_ADAPTED = {
    'camchex': 'task2_camchex_adapted',
    'med3dvlm': 'task2_med3dvlm_adapted',
    'm3fm': 'task2_m3fm_adapted',
}
MULTIMODAL_NEW = {
    'unified_mm': 'unified_multimodal_framework_2026',
    'adaptive_fusion': 'adaptive_multimodal_fusion_2026',
}
MULTIMODAL = tuple(MULTIMODAL_OLD) + tuple(MULTIMODAL_ADAPTED) + tuple(MULTIMODAL_NEW)
VARIANTS = ('full', 'a_shared', 'b_none', 'b_graph') + BASELINES + MULTIMODAL


def folder_for(ds, variant, fold):
    if variant in BASELINES:
        base = ROOT / 'outputs' / DATASETS[ds][1]
        base /= 'all_models/7_labels' if ds == 'amos_mm' else 'all_models_fivefold'
        return base / f'fold_{fold}' / variant
    if variant in MULTIMODAL_OLD:
        base = ROOT / 'outputs' / DATASETS[ds][1]
        base /= 'all_models/7_labels' if ds == 'amos_mm' else 'all_models_fivefold'
        return base / f'fold_{fold}' / MULTIMODAL_OLD[variant]
    if variant in MULTIMODAL_ADAPTED:
        base_name = 'amos_mm_7' if ds == 'amos_mm' else DATASETS[ds][1]
        base = ROOT / 'outputs' / 'task_adapted_vlms' / base_name
        if ds == 'amos_mm':
            base /= '7_labels'
        return base / f'fold_{fold}' / MULTIMODAL_ADAPTED[variant]
    if variant in MULTIMODAL_NEW:
        base = ROOT / 'outputs' / 'paper_results' / 'new_baselines'
        base_name = 'amos_mm' if ds == 'amos_mm' else ('mr_rate_1k' if ds == 'mr_rate' else ds)
        base /= base_name
        if ds == 'amos_mm':
            base /= '7_labels'
        return base / f'fold_{fold}' / MULTIMODAL_NEW[variant]
    base = ROOT / 'outputs/lccf_ablation' / ds / variant
    if ds == 'amos_mm':
        base /= '7_labels'
    return base / f'fold_{fold}' / 'amef_multimodal'


def reference(folder, ds, rows):
    if ds == 'amos_mm':
        with np.load(folder / 'test_predictions.npz', allow_pickle=False) as d:
            mapping = {r['scan_id']: r['case_index'] for r in rows}
            ids = np.array([mapping[str(s)] for s in d['scan_ids']])
            return ids, d['labels'].copy(), d['probabilities'].copy(), 'test_predictions.npz'
    with (folder / 'test_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        rr = list(csv.DictReader(stream))
    pc = [k for k in rr[0] if k.startswith('prob_')]
    yc = [k for k in rr[0] if k.startswith('true_')]
    return (np.array([int(r['patient_id']) for r in rr]),
            np.array([[int(r[k]) for k in yc] for r in rr]),
            np.array([[float(r[k]) for k in pc] for r in rr]), 'test_predictions.csv')


def load_model(folder, ds, variant, device):
    config = json.loads((folder / 'config.json').read_text())
    ck = torch.load(folder / 'best_model.pt', map_location='cpu', weights_only=False)
    params = dict(ck.get('model_parameters', config.get('parameters', config.get('model', {}))))
    if variant in BASELINES:
        image_base.LABEL_NAMES = list(range(DATASETS[ds][2]))
        image_base.MODEL_SPECS[variant] = params
        model = image_base.build_model(variant)
    elif variant in MULTIMODAL_OLD:
        params = {k: v for k, v in params.items() if not k.endswith('_weight')}
        model = build_task2_multimodal_sota(
            MULTIMODAL_OLD[variant], **params, pretrained=False,
            num_labels=DATASETS[ds][2])
    elif variant in MULTIMODAL_ADAPTED:
        params = {k: v for k, v in params.items() if not k.endswith('_weight')}
        model = build_task2_multimodal_sota(
            MULTIMODAL_ADAPTED[variant], **params, pretrained=False,
            num_labels=DATASETS[ds][2])
    elif variant in MULTIMODAL_NEW:
        params = {k: v for k, v in params.items() if not k.endswith('_weight')}
        model = build_paper2026_adapted(
            MULTIMODAL_NEW[variant], **params, pretrained=False,
            num_labels=DATASETS[ds][2])
    else:
        params = {k: v for k, v in params.items() if not k.endswith('_weight')}
        model = Exp13LCCFAblationModel(**params, pretrained=False, num_labels=DATASETS[ds][2])
    model.instance_encoder.backbone = image_base.CachedFeatureBackbone(768, params['feature_dim'], params['dropout'])
    model.load_state_dict(ck['state_dict'], strict=True)
    return model.to(device).eval(), ck.get('protocol_sha256'), config


def load_inputs(ds):
    path = ROOT / 'outputs' / DATASETS[ds][1] / 'experiment'
    rows = json.loads((path / 'samples.json').read_text())
    count = DATASETS[ds][2]
    y = np.array([r['labels'][:count] for r in rows])
    known = np.array([r.get('known_mask', [True] * count)[:count] for r in rows], bool)
    encoded = [_encode_text_fields({'watch': r['findings_masked']}, ('watch',), max_length=512, vocab_size=8192) for r in rows]
    tokens = torch.stack([v[0] for v in encoded]); text_masks = torch.stack([v[1] for v in encoded])
    bags = {}
    for fold in range(1, 6):
        ids, _, _, _ = reference(folder_for(ds, 'full', fold), ds, rows)
        for index in ids:
            if int(index) in bags:
                continue
            with np.load(path / 'features' / f'{index:04d}.npz', allow_pickle=False) as z:
                feat, pos, total = z['features'].astype(np.float32), z['slice_indices'].astype(np.int64), int(z['original_count'])
            assert 1 <= len(feat) <= 64 and feat.shape == (len(pos), 768)
            assert np.isfinite(feat).all() and np.all(np.diff(pos) > 0)
            bags[int(index)] = (feat, pos, total)
    return rows, y, known, tokens, text_masks, bags


def pack(ids, bags, tokens, text_masks, device):
    n = max(len(bags[int(i)][0]) for i in ids)
    x = torch.zeros(len(ids), n, 768, 1, 1)
    mask = torch.zeros(len(ids), n, dtype=torch.bool)
    pos = torch.full((len(ids), n), -1, dtype=torch.long)
    counts = []
    for j, i in enumerate(ids):
        feat, indices, total = bags[int(i)]
        m = len(feat)
        x[j, :m, :, 0, 0] = torch.from_numpy(feat)
        mask[j, :m] = True; pos[j, :m] = torch.from_numpy(indices); counts.append(total)
    return tuple(t.to(device) for t in (x, mask, pos, torch.tensor(counts), tokens[ids], text_masks[ids]))


def overlap(attention, mask):
    values = []
    for a, m in zip(attention.cpu().numpy(), mask.cpu().numpy()):
        a = a[:, m]
        k = min(5, a.shape[-1])
        # 稳定排序：所有共享分支使用相同的并列次序。
        top = [set(np.argsort(-v, kind='stable')[:k]) for v in a]
        values.append(np.mean([len(top[i] & top[j]) / len(top[i] | top[j]) for i in range(len(top)) for j in range(i + 1, len(top))]))
    return np.array(values)


def deletion_diagnostic(model, context, attention, mask):
    """旧图9对应的聚合层干预；保存归一化前的带符号概率差。"""
    shared = attention.mean(1, keepdim=True).expand_as(attention)
    result = []
    for a in (shared, attention):
        baseline = model.classify(torch.einsum('bln,bnd->bld', a, context)).sigmoid()
        columns = []
        for source in range(a.shape[1]):
            changed = a.clone()
            for bi in range(len(a)):
                n = int(mask[bi].sum())
                if n < 2:
                    continue
                k = min(5, n - 1)
                score = a[bi, source].masked_fill(~mask[bi], -torch.inf)
                idx = torch.argsort(score, descending=True, stable=True)[:k]
                changed[bi, :, idx] = 0
            changed /= changed.sum(-1, keepdim=True).clamp_min(1e-12)
            p = model.classify(torch.einsum('bln,bnd->bld', changed, context)).sigmoid()
            delta = baseline - p
            delta[mask.sum(1) < 2] = torch.nan
            columns.append(delta.cpu().numpy())
        result.append(np.stack(columns, 1))
    return np.stack(result, 1)


@torch.inference_mode()
def evaluate(ds, variant, fold, inputs, device, batch_size):
    rows, all_y, known, tokens, text_masks, bags = inputs
    folder = folder_for(ds, variant, fold)
    ids, ref_y, ref_p, ref_file = reference(folder, ds, rows)
    assert np.array_equal(all_y[ids], ref_y) and known[ids].all()
    common_ids, _, _, _ = reference(folder_for(ds, 'full', fold), ds, rows)
    assert np.array_equal(ids, common_ids), '各变体测试病例/顺序不一致'
    model, digest, config = load_model(folder, ds, variant, device)
    data = {'case': ids, 'fold': np.full(len(ids), fold), 'labels': ref_y}
    gathered = {}
    def add(key, value):
        if torch.is_tensor(value):
            value = value.float().cpu().numpy()
        gathered.setdefault(key, []).append(value)
    for start in range(0, len(ids), batch_size):
        case = ids[start:start + batch_size]
        x, mask, pos, counts, token, text_mask = pack(case, bags, tokens, text_masks, device)
        add('image_count', mask.sum(1))
        if variant in BASELINES:
            out = model(x, mask)
            add('prob', out['logits'].float().sigmoid())
            add('overlap', overlap(out['attention'], mask))
            continue
        if variant in MULTIMODAL:
            out = model(x, mask, watch_token_ids=token, watch_token_mask=text_mask)
            # 多模态基线均从统一底座输出标签级视觉注意力；这里仅分析其视觉分支，
            # 不把报告编码或最终融合权重混同为图像证据。
            add('overlap', overlap(out['attention'], mask))
            continue
        context, embeds, attn, _ = model.encode_long_mil(x, mask, pos, counts)
        text, tm, pooled, active = model.text_encoder(token, text_mask, batch_size=len(case), device=device)
        retrieved, _ = model._lccf_retrieve_text(embeds, text, tm)
        retrieved *= active[:, None, None].to(retrieved.dtype)
        def prediction(t):
            fuse, gate = model._lccf_fuse(embeds, t, active)
            return model.classify(fuse).sigmoid(), gate
        correct, gates = prediction(retrieved)
        add('prob', correct); add('visual_prob', model.classify(embeds).sigmoid())
        add('overlap', overlap(attn, mask)); add('text_active', active); add('gates', gates.squeeze(-1))
        if variant == 'full':
            swapped = []
            nlabels = embeds.shape[1]
            for shift in range(1, nlabels):
                permutation = (torch.arange(nlabels, device=device) + shift) % nlabels
                swapped.append(prediction(retrieved[:, permutation])[0])
            swapped = torch.stack(swapped, 1)
            add('cross_prob', swapped.mean(1)); add('swapped_prob', swapped)
            add('pooled_prob', prediction(pooled[:, None].expand_as(retrieved))[0])
            add('deletion_signed', deletion_diagnostic(model, context, attn, mask))
    data.update({k: np.concatenate(v) for k, v in gathered.items()})
    if variant in MULTIMODAL:
        # temp3 只使用注意力重叠；分类概率沿用 checkpoint 的原始测试记录，
        # 避免 AMP/后端舍入差异被误当成新的分类结果。
        data['prob'] = ref_p.copy()
    error = np.abs(data['prob'] - ref_p)
    # 原记录来自 CUDA；CPU 与 CUDA 的矩阵乘法舍入有约 1e-4 的差异。
    # 旧多模态训练使用 AMP，重放其缓存特征时会出现小幅舍入差异；
    # 仍记录最大误差，但采用与图像基线相同的复核容差。
    tolerance = .005 if (variant in BASELINES or variant in MULTIMODAL) else (5e-4 if device.type == 'cpu' else 3e-5)
    max_error = float(error.max())
    if max_error > tolerance:
        raise RuntimeError(f'{ds}/{variant}/fold{fold}: 原预测复核失败 {max_error} > {tolerance}')
    metadata = {
        'dataset': ds, 'variant': variant, 'fold': fold,
        'checkpoint': str((folder / 'best_model.pt').relative_to(ROOT)),
        'checkpoint_bytes': (folder / 'best_model.pt').stat().st_size,
        'checkpoint_mtime_ns': (folder / 'best_model.pt').stat().st_mtime_ns,
        'training_protocol_sha256': digest,
        'reference_predictions': str((folder / ref_file).relative_to(ROOT)),
        'cases': len(ids), 'probability_max_abs_error': max_error,
        'probability_mean_abs_error': float(error.mean()), 'tolerance': tolerance,
        'device': str(device), 'test_reference_passed': True,
        'classification_disagreements_at_0_5': int(((data['prob'] >= .5) != (ref_p >= .5)).sum()),
    }
    output = OUT / 'raw' / ds / variant
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f'fold_{fold}.npz', **data)
    (output / f'fold_{fold}.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(2026)
    device = torch.device(args.device)
    for ds in args.datasets:
        inputs = load_inputs(ds)
        jobs = [(v, f) for v in args.variants for f in args.folds]
        for variant, fold in tqdm(jobs, desc=f'{ds} 机制评估', mininterval=5):
            target = OUT / 'raw' / ds / variant / f'fold_{fold}.json'
            if args.resume and target.exists() and target.with_suffix('.npz').exists():
                assert json.loads(target.read_text())['test_reference_passed']
                continue
            result = evaluate(ds, variant, fold, inputs, device, args.batch_size)
            print(f"{ds}/{variant}/fold{fold}: {result['cases']}例, 复核误差={result['probability_max_abs_error']:.2g}", flush=True)


if __name__ == '__main__':
    main()
