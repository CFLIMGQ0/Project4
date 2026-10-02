#!/usr/bin/env python3
"""核验五折逐条件结果，输出2×3曲面、标准差和可追溯Markdown数据。"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/r051_3d_fivefold_20260929'
sys.path.insert(0, '/home/Lim/.agents/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment
DATASETS = [('ct_rate', 'CT-RATE'), ('amos_mm', 'AMOS-MM'), ('mr_rate_1k', 'MR-RATE-1K')]
RATIOS, BLOCKS = list(range(0, 81, 10)), list(range(8, 0, -1))


def read(path):
    return json.loads(Path(path).read_text())


def audit():
    summary = read(OUT / 'summary.json')
    assert summary['completed_grids'] == 15 and summary['measured_paired_conditions'] == 975
    checked, original_ids = 0, {}
    for ds, _ in DATASETS:
        for fold in range(1, 6):
            folder = OUT / 'grids' / ds / f'fold_{fold}'
            obj = read(folder / 'grid.json')
            assert obj['evaluation_split'] == 'test' and len(obj['results']) == 65
            assert obj['protocol_sha256'] == summary['protocol_sha256']
            for name, source in obj['checkpoints'].items():
                assert hashlib.sha256((ROOT / source['path']).read_bytes()).hexdigest() == source['sha256']
            for key, result in tqdm(obj['results'].items(), desc=f'{ds} fold{fold} 独立复核', mininterval=5):
                path = ROOT / result['prediction_file']
                assert hashlib.sha256(path.read_bytes()).hexdigest() == result['prediction_sha256']
                with np.load(path, allow_pickle=False) as z:
                    assert z['case_indices'].tolist() == obj['case_indices']
                    y, known = z['labels'] > .5, z['known_mask']
                    for name in ['r051', 'original_pe']:
                        pred = z[name + '_probabilities'] >= z[name + '_thresholds']
                        assert np.isfinite(z[name + '_probabilities']).all()
                        tp = (pred & y & known).sum(0)
                        fp = (pred & ~y & known).sum(0)
                        fn = (~pred & y & known).sum(0)
                        f1 = float((2 * tp / np.maximum(2 * tp + fp + fn, 1)).mean())
                        assert abs(f1 - result[name]['macro_f1']) < 1e-12
                        acc = float(z[name + '_case_acc005'].mean())
                        assert abs(acc - result[name]['acc005']) < 1e-12 and 0 <= acc <= 1
                    # 独立核对Original PE的均匀槽位坐标指标，不把端点计入分母。
                    indices, counts = z['selected_raw_indices'], z['original_counts']
                    original_acc = []
                    for selected, count in zip(indices, counts):
                        selected = selected[selected >= 0]
                        assert selected[0] == 0 and selected[-1] == count - 1
                        truth = selected.astype(np.float64) / (count - 1)
                        error = np.abs(np.linspace(0, 1, len(selected)) - truth)[1:-1]
                        original_acc.append(float((error <= .05).mean()))
                    np.testing.assert_allclose(original_acc, z['original_pe_case_acc005'], atol=1e-12, rtol=0)
                assert abs(result['delta_f1'] - result['r051']['macro_f1'] + result['original_pe']['macro_f1']) < 1e-12
                assert abs(result['delta_acc005'] - result['r051']['acc005'] + result['original_pe']['acc005']) < 1e-12
                checked += 1
            original_ids[(ds, fold)] = obj['case_indices']
    for ds, _ in DATASETS:
        for left in range(1, 6):
            for right in range(left + 1, 6):
                a, b = set(original_ids[(ds, left)]), set(original_ids[(ds, right)])
                assert a == b if ds == 'amos_mm' else not a & b
    (OUT / 'metrics_audit.json').write_text(json.dumps(dict(status='passed', paired_conditions=checked,
        checks=['逐条件预测哈希', '阈值与已知标签掩码下独立重算F1', '逐病例位置指标均值',
                '独立重算Original PE坐标一致性', '所有条件固定病例集合', '折间测试病例划分', '检查点哈希']),
        ensure_ascii=False, indent=2) + '\n')
    return summary


def plot(summary):
    values = {(r['dataset'], r['ratio'], r['blocks']): r for r in summary['conditions']}
    assert len(values) == 195
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'font.size': 6, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.linewidth': .5, 'savefig.transparent': False})
    fig = plt.figure(figsize=(7, 4.65))
    grid = fig.add_gridspec(2, 3, left=.015, right=.91, bottom=.105, top=.91,
                            wspace=.08, hspace=.27)
    axes = []
    x, y = np.meshgrid(RATIOS, BLOCKS)
    for row, (field, label) in enumerate([('delta_f1', 'ΔF1 (pp)'), ('delta_acc005', 'ΔAcc@0.05 (pp)')]):
        surfaces = [np.asarray([[values[(ds, ratio, 1 if ratio == 0 else b)][field + '_mean'] * 100
                                for ratio in RATIOS] for b in BLOCKS]) for ds, _ in DATASETS]
        limit = max(1., float(np.ceil(max(abs(z).max() for z in surfaces))))
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        cmap = mpl.colormaps['RdBu']
        for col, ((ds, title), z) in enumerate(zip(DATASETS, surfaces)):
            assert z.shape == (8, 9) and np.isfinite(z).all() and np.all(z[:, 0] == z[0, 0])
            ax = fig.add_subplot(grid[row, col], projection='3d'); axes.append(ax)
            ax.plot_surface(x, y, z, facecolors=cmap(norm(z)), shade=False, edgecolor='#717171',
                            linewidth=.2, antialiased=True, rstride=1, cstride=1, alpha=.92)
            ax.scatter(x.ravel(), y.ravel(), z.ravel(), s=1.7, c=cmap(norm(z.ravel())), depthshade=False)
            ax.contourf(x, y, z, levels=np.linspace(-limit, limit, 21), zdir='z', offset=-limit,
                        cmap=cmap, norm=norm, alpha=.25)
            ax.set_xlim(0, 80); ax.set_ylim(8, 1); ax.set_zlim(-limit, limit)
            ax.set_xticks([0, 40, 80], labels=['0', '40', '80'])
            ax.set_yticks([8, 4, 1], labels=['B8', 'B4', 'B1'])
            ax.set_zticks([-limit, 0, limit])
            ax.tick_params(labelsize=5.5, pad=0, length=2)
            ax.set_xlabel('Deletion (%)', fontsize=6, labelpad=1)
            ax.set_ylabel('Blocks', fontsize=6, labelpad=1)
            ax.view_init(elev=25, azim=-58)
            ax.set_box_aspect((1, .9, .72), zoom=.9)
            ax.set_title(title, fontsize=7, pad=5)
            ax.text2D(.02, 1.04, chr(97 + row * 3 + col), transform=ax.transAxes,
                      fontsize=7, fontweight='bold')
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.set_facecolor((.97, .97, .97, 1.))
                axis._axinfo['grid'].update(color=(.75, .75, .75, .5), linewidth=.4)
        cax = fig.add_axes((.925, .605 if row == 0 else .20, .012, .22))
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[-limit, 0, limit])
        colorbar.ax.tick_params(labelsize=5.5, length=2, pad=2)
        colorbar.outline.set_linewidth(.5)
        colorbar.set_label(label, fontsize=6, labelpad=3)
    fig.text(.5, .976, 'ACPE − Original PE | Mean over five trained folds', ha='center', va='top', fontsize=7)
    fig.text(.5, .037, 'Smaller B: longer contiguous gaps at the same deletion ratio.', ha='center', fontsize=6)
    stem = OUT / 'r051_fivefold_3d_2x3'
    require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=list('abcdef'),
        json_out=str(stem) + '.alignment.json', overlay_svg=str(stem) + '.alignment.svg',
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    # 3D曲面仅显示五折配对差值的均值，所有条件的样本标准差独立完整导出。
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.png'), dpi=600)
    plt.close(fig)


def tables(summary):
    lines = ['# r051：五折测试的完整三维删除网格', '',
             '> 3个数据集 × 5折 × 65独立条件 = 975个配对条件；每个条件评估ACPE和Original PE。',
             '> 以下为真实五折评估，不是复制单折结果；0%删除仅实测一次，在图的B轴重复展示。', '',
             '## 评估协议', '',
             '- 复用已完成的r051五折权重，对照使用同一划分、种子、训练设置重新训练Original PE。',
             '- 先对完整原序列做B-block删除，再均匀采样至多64张，保留原始采集索引。',
             '- 删除比例0%—80%，步长10%；B1—B8。固定比例下B越小，连续缺失越集中。',
             '- 每折所有条件和两个模型共用同一可评估测试病例集合；序列过短者统一排除。',
             '- 阈值在干净验证集确定并固定，测试和删除曲面不参与调参。',
             '- CT-RATE、MR-RATE-1K使用五个互斥测试折；AMOS-MM五次训练评估同一个固定测试集。',
             '- r051保留8轮位置warmup；Original PE保留标准注入方式，未额外引入warmup机制。',
             '- Acc@0.05衡量归一化采集索引一致性；端点不计入分母，先逐病例计算再平均。',
             '- 均值和样本标准差（ddof=1）均按五折计算，Δ逐折配对后汇总。',
             '- 图中绘制均值，不叠加密集三维误差棒；标准差和全部逐折实测数据如下。', '',
             '## 五折汇总（单位：百分点）', '',
             '| 数据集 | 删除比例 | B | ACPE F1 | Original PE F1 | ΔF1 | ACPE Acc@0.05 | Original PE Acc@0.05 | ΔAcc@0.05 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['conditions']:
        vals = []
        for field in ['r051_macro_f1', 'original_pe_macro_f1', 'delta_f1', 'r051_acc005', 'original_pe_acc005', 'delta_acc005']:
            vals.append(f"{r[field + '_mean'] * 100:.4f} ± {r[field + '_std'] * 100:.4f}")
        lines.append(f"| {r['dataset']} | {r['ratio']}% | {r['blocks']} | " + ' | '.join(vals) + ' |')
    lines += ['', '## 全部逐折实测数据（单位：0—1比例）', '',
              '| 数据集 | 折 | 删除比例 | B | 病例数 | ACPE F1 | Original PE F1 | ΔF1 | ACPE Acc@0.05 | Original PE Acc@0.05 | ΔAcc@0.05 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    with (OUT / 'grid_folds.csv').open() as stream:
        for r in csv.DictReader(stream):
            fields = ['Dataset', 'Fold', 'DeletionRatio', 'Blocks', 'NumCases', 'R051_MacroF1',
                      'OriginalPE_MacroF1', 'Delta_MacroF1', 'R051_Acc005', 'OriginalPE_Acc005', 'Delta_Acc005']
            lines.append('| ' + ' | '.join(r[k] for k in fields) + ' |')
    (OUT / 'temp_fivefold.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    summary = audit()
    tables(summary)
    plot(summary)
