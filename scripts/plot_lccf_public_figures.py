#!/usr/bin/env python3
"""使用可追溯的公开五折推理结果绘制 temp3--temp6，不更改论文。"""
from __future__ import annotations
import csv
import json
import string
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from evaluate_lccf_public_figures import ROOT, OUT, DATASETS, VARIANTS

SKILL = Path('/home/Lim/.agents/skills/nature-figure/scripts')
sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

NAMES = {
    'a_shared': 'Shared', 'attention_mil': 'Attn.', 'clam_mb': 'C-MB',
    'clam_sb': 'C-SB', 'dsmil': 'DSMIL', 'transmil': 'TransMIL', 'dtfd_mil': 'DTFD',
    'mmfnet': 'MMF', 'radfuse': 'RadF.', 'saif': 'SAIF', 'mmtf': 'MMTF',
    'camchex': 'CaM', 'med3dvlm': 'Med3D', 'm3fm': 'M3FM',
    'unified_mm': 'U-MM', 'adaptive_fusion': 'A-Fus.', 'full': 'LCCF',
}
ORDER = ['a_shared', 'attention_mil', 'clam_sb', 'clam_mb', 'dsmil', 'transmil', 'dtfd_mil',
         'mmfnet', 'radfuse', 'saif', 'mmtf', 'camchex', 'med3dvlm', 'm3fm',
         'unified_mm', 'adaptive_fusion', 'full']
SHORT_LABELS = {
    'ct_rate': ['Emph.', 'Atel.', 'Fibrotic'],
    'mr_rate': ['Uns.', 'Neuro.', 'Cerebro.', 'Neo.'],
    'amos_mm': ['L', 'K', 'G', 'S', 'B', 'P', 'T'],
}
COLORS = {'full': '#C44748', 'b_none': '#A4A4A4', 'b_graph': '#E5A619', 'correct': '#C44748', 'cross': '#4C78A8', 'pooled': '#999999'}
IMAGE_METHODS = {'attention_mil', 'clam_sb', 'clam_mb', 'dsmil', 'transmil', 'dtfd_mil'}
MULTIMODAL_METHODS = {'mmfnet', 'radfuse', 'saif', 'mmtf', 'camchex', 'med3dvlm', 'm3fm', 'unified_mm', 'adaptive_fusion'}
SOURCE = []


def load():
    result = {}
    for ds in DATASETS:
        result[ds] = {}
        for method in VARIANTS:
            parts = []
            for fold in range(1, 6):
                p = OUT / 'raw' / ds / method / f'fold_{fold}.npz'
                meta = json.loads(p.with_suffix('.json').read_text())
                assert meta['test_reference_passed']
                with np.load(p, allow_pickle=False) as z:
                    parts.append({k: z[k] for k in z.files})
            result[ds][method] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        for method in VARIANTS:
            assert np.array_equal(result[ds][method]['case'], result[ds]['full']['case'])
            assert np.array_equal(result[ds][method]['labels'], result[ds]['full']['labels'])
    return result


def collapse_cases(case, values):
    """AMOS 的重复测试先按病例平均，避免散点伪重复。"""
    return np.stack([np.mean(values[case == c], axis=0) for c in np.unique(case)])


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.sans-serif': ['DejaVu Sans'], 'svg.fonttype': 'none', 'font.size': 9,
        'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8,
        'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.linewidth': .75,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})


def decorate(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E2E5E9', linewidth=.65)
    ax.spines[['top', 'right']].set_visible(False)


def save(fig, axes, name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=list(np.asarray(axes).flat),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
        json_out=OUT / f'{name}.alignment.json')
    fig.savefig(OUT / f'{name}.pdf', dpi=300)
    fig.savefig(OUT / f'{name}.svg', dpi=300)
    fig.savefig(OUT / f'{name}.png', dpi=300)
    # 仅完成渲染后替换用户指定的四个项目根目录预览。
    (ROOT / f'{name}.png').write_bytes((OUT / f'{name}.png').read_bytes())
    plt.close(fig)


def fig_overlap(raw):
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.15), sharey=True)
    fig.subplots_adjust(left=.045, right=.995, bottom=.36, top=.82, wspace=.12)
    rng = np.random.default_rng(2026)
    for di, (ax, ds) in enumerate(zip(axes, DATASETS)):
        d = raw[ds]
        arrays = [collapse_cases(d[m]['case'], d[m]['overlap']) for m in ORDER]
        for j, (method, values) in enumerate(zip(ORDER, arrays)):
            color = ('#C44748' if method == 'full' else '#868E96' if method == 'a_shared'
                     else '#58A6A6' if method in MULTIMODAL_METHODS else '#4C78A8')
            ax.scatter(j + rng.uniform(-.16, .16, len(values)), values, s=3,
                       color=color, alpha=.12, edgecolors='none', rasterized=True)
            ax.boxplot([values], positions=[j], widths=.56, manage_ticks=False, showfliers=False,
                patch_artist=True, boxprops={'facecolor': color, 'alpha': .27, 'edgecolor': color},
                medianprops={'color': '#222222', 'linewidth': .9},
                whiskerprops={'color': color, 'linewidth': .7}, capprops={'color': color, 'linewidth': .7})
            ax.scatter(j, values.mean(), marker='D', s=13, facecolor=color, edgecolor='white', linewidth=.4, zorder=4)
            SOURCE.append({'figure': 'temp3', 'dataset': ds, 'condition': method,
                           'group': 'all examinations', 'n_unique': len(values), 'value': float(values.mean())})
        ax.set_title(f'{string.ascii_lowercase[di]}  {DATASETS[ds][0]} (n={len(arrays[0])})', loc='left', fontweight='bold', pad=12)
        # 分开显示图像基线、多模态基线和 LCCF，避免把不同证据来源混为一组。
        ax.axvspan(0.5, 6.5, color='#4C78A8', alpha=.035, zorder=0)
        ax.axvspan(6.5, 15.5, color='#58A6A6', alpha=.035, zorder=0)
        ax.set_xticks(range(len(ORDER)), [NAMES[m] for m in ORDER], rotation=65, ha='right', rotation_mode='anchor', fontsize=6.2)
        ax.set_xlim(-.65, len(ORDER)-.35); ax.set_ylim(-.045, 1.08)
        ax.set_yticks(np.arange(0, 1.01, .2)); decorate(ax)
    axes[0].set_ylabel('Pairwise top-5 Jaccard overlap')
    fig.text(.53, .055, 'Blue: image-only baselines  |  Teal: multimodal baselines  |  Red: LCCF', ha='center', fontsize=8, color='#555555')
    fig.text(.53, .018, 'Boxes: median and IQR  |  Diamonds: mean  |  Points: examinations', ha='center', fontsize=8, color='#555555')
    save(fig, axes, 'temp3')


def matrix_summary(d, condition):
    matrix = np.zeros((d['labels'].shape[1], d['labels'].shape[1]))
    for source in range(len(matrix)):
        selected = (d['labels'][:, source] > 0) & (d['image_count'] > 1)
        changes = np.abs(d['deletion_signed'][selected, condition, source])
        means = collapse_cases(d['case'][selected], changes).mean(0)
        matrix[source] = 100 * means / means.sum() if means.sum() > 0 else np.nan
    return matrix


def fig_deletion(raw):
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 7.0))
    fig.subplots_adjust(left=.105, right=.88, top=.89, bottom=.15, wspace=.52, hspace=.84)
    for row in range(2):
        for col, ds in enumerate(DATASETS):
            ax = axes[row, col]; mat = matrix_summary(raw[ds]['full'], row)
            im = ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=100, interpolation='nearest')
            mode = 'Shared control' if row == 0 else 'Label-wise attention'
            diag = np.diag(mat).mean()
            ax.set_title(f'{string.ascii_lowercase[row*3+col]}  {DATASETS[ds][0]}\n{mode}', loc='left', fontsize=9, pad=9)
            labels = SHORT_LABELS[ds]
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right', rotation_mode='anchor', fontsize=7.3)
            ax.set_yticks(range(len(labels)), labels, fontsize=7.3)
            ax.tick_params(length=0)
            for i in range(len(mat)):
                for j in range(len(mat)):
                    ax.text(j, i, f'{mat[i,j]:.1f}', ha='center', va='center', fontsize=7 if len(mat)>4 else 9,
                        fontweight='bold' if i == j else 'normal', color='white' if mat[i,j] >= 55 else '#262626')
                    SOURCE.append({'figure': 'temp4', 'dataset': ds, 'condition': mode, 'group': labels[i],
                                   'target': labels[j], 'value': float(mat[i,j])})
    cax = fig.add_axes([.92, .23, .018, .50])
    cb = fig.colorbar(im, cax=cax); cb.set_label('Row-normalized absolute decision impact (%)', fontsize=8.5)
    fig.text(.018, .53, 'Deleted-evidence label', rotation=90, va='center', fontsize=10)
    fig.text(.49, .059, 'Affected output label', ha='center', fontsize=10)
    fig.text(.49, .025, 'Top-5 evidence removal at visual aggregation; fixed contextual features', ha='center', fontsize=8, color='#555555')
    save(fig, axes, 'temp4')


def per_exam_confidence(p, y):
    count = y.sum(1)
    return (p * y).sum(1) / np.maximum(count, 1)


def fold_means(d, field, group):
    y = d['labels']; pos = y.sum(1)
    selected = pos == 1 if group == 'Single-positive' else pos >= 2 if group == 'Co-positive' else pos > 0
    if field != 'visual_prob':
        selected &= d['text_active'] > 0
    values = per_exam_confidence(d[field], y)
    result = np.array([values[selected & (d['fold'] == f)].mean() for f in range(1,6)])
    return result, len(np.unique(d['case'][selected]))


def fig_reasoning(raw):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), sharey=True)
    fig.subplots_adjust(left=.075, right=.99, top=.80, bottom=.22, wspace=.20)
    methods = ['b_none', 'b_graph', 'full']; groups = ['Single-positive', 'Co-positive']
    labels = ['No reasoning', 'Ordinary graph', 'Label hypergraph']
    width = .23
    for di, (ax, ds) in enumerate(zip(axes, DATASETS)):
        counts = []
        for mi, method in enumerate(methods):
            means, stds, local_counts = [], [], []
            for group in groups:
                f, n = fold_means(raw[ds][method], 'visual_prob', group)
                means.append(f.mean()); stds.append(f.std(ddof=1)); local_counts.append(n)
                for fi, val in enumerate(f, 1):
                    SOURCE.append({'figure': 'temp5', 'dataset': ds, 'condition': labels[mi], 'group': group,
                                   'fold': fi, 'n_unique': n, 'value': float(val)})
            x = np.arange(2) + (mi-1)*width
            ax.bar(x, means, width, color=COLORS[method], edgecolor='white', linewidth=.5, label=labels[mi])
            ax.errorbar(x, means, yerr=stds, fmt='none', ecolor='#333333', capsize=3, elinewidth=.9)
            counts = local_counts
        ax.set_title(f'{string.ascii_lowercase[di]}  {DATASETS[ds][0]}', loc='left', fontweight='bold', pad=12)
        ax.set_xticks(range(2), [f'{g}\n(n={n})' for g,n in zip(groups, counts)])
        ax.set_ylim(0,1.08); ax.set_yticks(np.arange(0,1.01,.2)); decorate(ax)
    axes[0].set_ylabel('Mean positive-label confidence')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.53,.985), ncol=3, frameon=False)
    fig.text(.53,.045,'Visual pathway; bars: five-fold mean; error bars: fold-to-fold SD',ha='center',fontsize=8,color='#555555')
    save(fig, axes, 'temp5')


def fig_retrieval(raw):
    fig, axes = plt.subplots(1,3,figsize=(10.2,3.3),sharey=True)
    fig.subplots_adjust(left=.075,right=.99,top=.84,bottom=.24,wspace=.20)
    conditions = [('prob','Correct','#C44748'),('cross_prob','Cross-label','#4C78A8'),('pooled_prob','Pooled','#999999')]
    all_bounds=[]; collection={}
    for ds in DATASETS:
        collection[ds]=[fold_means(raw[ds]['full'],key,'All positive examinations')[0] for key,_,_ in conditions]
        for v in collection[ds]: all_bounds.extend([v.mean()-v.std(ddof=1),v.mean()+v.std(ddof=1)])
    lo=max(0.,min(all_bounds)-.04); hi=min(1.03,max(all_bounds)+.065)
    for di,(ax,ds) in enumerate(zip(axes,DATASETS)):
        values=collection[ds]; means=[v.mean() for v in values]
        ax.plot(range(3),means,color='#333333',linewidth=1.4,zorder=2)
        for j,(v,(_,name,color)) in enumerate(zip(values,conditions)):
            mean,sd=v.mean(),v.std(ddof=1)
            ax.errorbar(j,mean,yerr=sd,fmt='o',color=color,markersize=7,capsize=4,elinewidth=1.0,
                        markeredgecolor='white',markeredgewidth=.7,zorder=3)
            for fi,val in enumerate(v,1):
                SOURCE.append({'figure':'temp6','dataset':ds,'condition':name,'group':'positive examinations','fold':fi,'value':float(val)})
        ax.set_title(f'{string.ascii_lowercase[di]}  {DATASETS[ds][0]}',loc='left',fontweight='bold',pad=12)
        ax.set_xticks(range(3),[v[1] for v in conditions]);ax.set_xlim(-.35,2.35);ax.set_ylim(lo,hi);decorate(ax)
    axes[0].set_ylabel('Mean positive-label confidence')
    fig.text(.53,.053,'Matched visual evidence; points: five-fold mean; error bars: fold-to-fold SD',ha='center',fontsize=8,color='#555555')
    save(fig,axes,'temp6')


def write_sources(raw):
    keys=sorted(set(k for row in SOURCE for k in row))
    with (OUT/'figure_source_data.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(SOURCE)
    stats={}
    for ds in DATASETS:
        stats[ds]={'cases_unique':len(np.unique(raw[ds]['full']['case'])),
                   'checkpoint_case_evaluations':len(raw[ds]['full']['case'])}
        for mode in ['b_none','b_graph','full']:
            stats[ds][mode]={}
            for g in ['Single-positive','Co-positive']:
                f,n=fold_means(raw[ds][mode],'visual_prob',g)
                stats[ds][mode][g]={'mean':float(f.mean()),'sd':float(f.std(ddof=1)),'n_unique':n}
        d=raw[ds]['full']; swap=d['swapped_prob']; y=d['labels']; l=y.shape[1]
        # 单独记录交换伙伴为阴性/共阳性的结果，既不丢弃也不混淆共存证据。
        stats[ds]['swaps_by_partner']={}
        for partner_positive in [False,True]:
            correct_values=[];swapped_values=[]
            for shift in range(1,l):
                perm=(np.arange(l)+shift)%l
                valid=(y>0)&((y[:,perm]>0)==partner_positive)&(d['text_active'][:,None]>0)
                correct_values.extend(d['prob'][valid]);swapped_values.extend(swap[:,shift-1][valid])
            stats[ds]['swaps_by_partner']['co_positive' if partner_positive else 'absent']={
                'evaluations':len(correct_values),
                'correct_mean':float(np.mean(correct_values)) if correct_values else None,
                'swapped_mean':float(np.mean(swapped_values)) if swapped_values else None}
    (OUT/'summary.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False))


def main():
    raw=load();style()
    fig_overlap(raw);fig_deletion(raw);fig_reasoning(raw);fig_retrieval(raw)
    write_sources(raw)
    (OUT/'completed.json').write_text(json.dumps({'datasets':list(DATASETS),'folds':5,
        'figures':['temp3.png','temp4.png','temp5.png','temp6.png'],'source_data':'figure_source_data.csv'},indent=2))
    print('四张图已渲染，数据与PDF位于',OUT)


if __name__=='__main__':
    main()
