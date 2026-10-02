#!/usr/bin/env python3
"""整理 r051 单折验证网格，绘制三维曲面并替换 temp.md。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/r051_3d_20260928"
sys.path.insert(0,"/home/Lim/.agents/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment
DATASETS = [("ct_rate","CT-RATE"),("amos_mm","AMOS-MM"),("mr_rate_1k","MR-RATE-1K")]
RATIOS = list(range(0,81,10))
BLOCKS = list(range(8,0,-1))


def read_data(require_all=True):
    values={}
    for key,_ in DATASETS:
        path=OUT/key/"grid.json"
        if not path.exists():
            if require_all:raise RuntimeError(f"缺少 {key} 实测网格")
            continue
        obj=json.loads(path.read_text())
        assert obj['evaluation_split']=='validation' and obj['fold']==0
        assert obj['index_protocol']=='original_acquisition_indices'
        if len(obj['results'])!=65:
            if require_all:raise RuntimeError(f"{key} 仅完成{len(obj['results'])}/65")
            continue
        for ratio in RATIOS:
            for b in ([1] if ratio==0 else range(1,9)):
                r=obj['results'][f'{ratio:02d}_B{b}']
                assert r['status']=='COMPLETE' and r['num_cases']==obj['num_cases']
                for metric,delta in [('macro_f1','delta_f1'),('acc005','delta_acc005')]:
                    assert abs(r['r051'][metric]-r['original_pe'][metric]-r[delta])<1e-12
                    assert all(0<=r[name][metric]<=1 for name in ('r051','original_pe'))
        values[key]=obj
    return values


def surface(obj,metric):
    result=np.asarray([[obj['results'][f'{ratio:02d}_B{1 if ratio==0 else b}'][metric]
                        for ratio in RATIOS] for b in BLOCKS])
    assert result.shape==(8,9) and np.isfinite(result).all()
    assert np.all(result[:,0]==result[0,0])
    return result


def plot(values,preview=False):
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
                         'pdf.fonttype':42,'svg.fonttype':'none','font.size':7,
                         'axes.linewidth':.6,'savefig.transparent':False})
    fig=plt.figure(figsize=(10.2,5.8))
    grid=fig.add_gridspec(2,4,left=.045,right=.905,bottom=.13,top=.90,wspace=.12,hspace=.23)
    metrics=[('delta_f1','ΔMacro-F1 (pp)'),('delta_acc005','ΔAcc@0.05 (pp)')]
    axes=[];cmap=mpl.colormaps['RdBu']
    x,y=np.meshgrid(RATIOS,BLOCKS)
    for row,(metric,label) in enumerate(metrics):
        measured=[surface(o,metric)*100 for o in values.values()]
        limit=max(1.,np.ceil(max(float(np.abs(z).max()) for z in measured)))
        norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
        for col,(key,title) in enumerate(DATASETS+[(None,'Fourth dataset')]):
            ax=fig.add_subplot(grid[row,col],projection='3d');axes.append(ax)
            if key is None or key not in values:
                ax.set_axis_off()
                message='Not evaluated' if key is None else 'Evaluation in progress'
                ax.text2D(.5,.45,message,transform=ax.transAxes,ha='center',va='center',fontsize=8,color='#656565')
            else:
                z=surface(values[key],metric)*100
                ax.plot_surface(x,y,z,facecolors=cmap(norm(z)),shade=False,edgecolor='#484848',
                                linewidth=.18,antialiased=True,rstride=1,cstride=1)
                ax.scatter(x.ravel(),y.ravel(),z.ravel(),s=1,color='#303030',depthshade=False)
                ax.set_xlim(0,80);ax.set_ylim(8,1);ax.set_zlim(-limit,limit)
                ax.set_xticks([0,40,80],labels=['0','40','80'])
                ax.set_yticks([8,4,1],labels=['B8','B4','B1'])
                ax.set_zticks([-limit,0,limit],labels=[f'−{limit:g}','0',f'{limit:g}'])
                ax.tick_params(labelsize=6,pad=-1,length=2,width=.4)
                ax.view_init(elev=26,azim=-58);ax.set_box_aspect((1,.85,.7),zoom=.78)
                for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
                    axis.pane.fill=False;axis.line.set_linewidth(.4)
                ax.grid(False)
            if row==0:ax.set_title(title,fontsize=8.5,pad=10)
            ax.text2D(.0,1.07,chr(ord('a')+row*4+col),transform=ax.transAxes,
                      fontsize=9,fontweight='bold',ha='left',va='top')
        cax=fig.add_axes((.93,.59 if row==0 else .205,.012,.23))
        cbar=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,ticks=[-limit,0,limit])
        cbar.ax.tick_params(labelsize=7,length=2,pad=2);cbar.outline.set_linewidth(.5)
        fig.text(.012,.69 if row==0 else .285,label,rotation=90,rotation_mode='anchor',
                 fontsize=9,ha='center',va='center')
    fig.suptitle('r051 − Original PE | Single-fold validation',x=.5,y=.976,fontsize=10)
    fig.text(.5,.065,'Deletion ratio: 0–80%  ·  Contiguous deletion blocks: B1–B8',ha='center',fontsize=8)
    fig.text(.5,.029,'Original acquisition indices retained; Acc@0.05 measures acquisition-index agreement.',
             ha='center',fontsize=7)
    name='r051_deletion_2x4_preview' if preview else 'r051_deletion_2x4'
    stem=OUT/name
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=list('abcdefgh'),
        json_out=str(stem)+'.alignment.json',overlay_svg=str(stem)+'.alignment.svg',
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    for suffix in ('pdf','svg','png'):
        fig.savefig(stem.with_suffix('.'+suffix),dpi=350,bbox_inches='tight')
    plt.close(fig)
    print('三维图已导出',stem)


def write_tables(values):
    fields=['Dataset','DeletionRatio','Blocks','Fold','Seed','NumCases','R051_MacroF1',
            'OriginalPE_MacroF1','Delta_MacroF1','R051_Acc005','OriginalPE_Acc005','Delta_Acc005','Status']
    rows=[]
    for dataset,title in DATASETS:
        obj=values[dataset]
        for ratio in RATIOS:
            for b in ([1] if ratio==0 else range(1,9)):
                r=obj['results'][f'{ratio:02d}_B{b}']
                rows.append([dataset,ratio,b,0,42,r['num_cases'],r['r051']['macro_f1'],
                             r['original_pe']['macro_f1'],r['delta_f1'],r['r051']['acc005'],
                             r['original_pe']['acc005'],r['delta_acc005'],'COMPLETE'])
    assert len(rows)==195
    with (OUT/'grid.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(fields);writer.writerows(rows)
    lines=['# r051 三维图：单折验证集完整删片网格','',
           '> 本文件已替换为改进版 r051 的真实评估结果。三个数据集共 195 个独立条件全部完成。',
           '> 单折、单种子结果，无五折均值或标准差；不是原图的五折测试集结果。','',
           '## 图与数据','',
           '- [三维图 PDF](outputs/r051_3d_20260928/r051_deletion_2x4.pdf)',
           '- [三维图 PNG](outputs/r051_3d_20260928/r051_deletion_2x4.png)',
           '- [完整 CSV](outputs/r051_3d_20260928/grid.csv)',
           '- [覆盖前的 temp.md 备份](outputs/r051_3d_20260928/temp_before_r051.md)','',
           '## 评估协议','',
           '- r051 配置：position_warmup=8、descriptor=no_current；保留已训练权重及 checkpoint 对应的位置缩放。',
           '- 对照：同一研究协议、同一开发划分训练的 r033 Original PE；没有为对照额外训练 warm-up 版本。',
           '- 输入原始采集索引。对完整原始序列进行 B-block 删除，再均匀采样至多64张；图像掩码标记补齐。',
           '- 比例0%至80%，步长10%；B1至B8。0%只实测B1，绘图复制至其余B值，所以每个数据集65个独立条件。',
           '- 两模型、全部条件共用同一可评估病例集合；不使用训练病例或保留测试集。',
           '- 使用各模型先前保存的干净验证集分类阈值，固定阈值，不利用本次曲面重新调参；Macro-F1只计算已知标签。',
           '- Acc@0.05：逐病例将上下文坐标归一化到首尾跨度，统计内部切片与原始归一化采集索引的误差≤0.05比例，再平均病例。Original PE使用均匀槽位坐标。',
           '- 该指标是采集索引一致性，且r051输入可获取采集索引；不是物理距离或解剖位置恢复能力的证据。',
           '- Δ = r051 − Original PE。下表采用0–1比例；图上乘以100显示百分点，蓝色为正、红色为负。',
           '- 原 temp.md 使用五折测试集及重编号输入；之前自动研究是在至多64张缓存包上删片。本次两者均不同，不直接混合或比较其绝对分数。',
           '- 图的第四列无可用数据，明确留空；不填入 Merlin 或其他版本结果。','',
           '## 病例与检查点','',
           '| 数据集 | 原验证病例 | 全网格共同病例 | 排除病例 | r051 最佳epoch / 位置缩放 | Original PE 最佳epoch |',
           '| --- | ---: | ---: | ---: | --- | ---: |']
    for dataset,title in DATASETS:
        obj=values[dataset];elig=json.loads((OUT/dataset/'eligibility.json').read_text());c=obj['checkpoints']
        lines.append(f"| {title} | {len(elig['validation_ids'])} | {obj['num_cases']} | {len(elig['excluded'])} | {c['r051']['best_epoch']} / {c['r051']['position_scale']:.3f} | {c['original_pe']['best_epoch']} |")
    lines += ['', '排除规则仅依赖序列长度：病例必须支持全部65个删除条件。短序列无法构造足够的非空、非相邻删除块时，从全部条件中统一排除。逐病例原因在各数据集 eligibility.json。','',
              '## 全部195个实测条件','',
              '| 数据集 | 删除比例 | B | 病例数 | r051 F1 | Original PE F1 | ΔF1 | r051 Acc@0.05 | Original PE Acc@0.05 | ΔAcc@0.05 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    titles=dict(DATASETS)
    for row in rows:
        lines.append('| '+' | '.join([titles[row[0]],str(row[1])+'%',str(row[2]),str(row[5])]+[repr(x) for x in row[6:12]])+' |')
    lines += ['', '## 绘图矩阵','', '每张表行对应删除比例，列对应B8至B1；0%这一行的重复值只用于连接曲面，不是额外实验。']
    for dataset,title in DATASETS:
        for metric,label in [('delta_f1','ΔMacro-F1'),('delta_acc005','ΔAcc@0.05')]:
            lines += ['', f'### {title}：{label}', '', '| 删除比例 | '+' | '.join('B'+str(b) for b in BLOCKS)+' |', '| ---: | '+' | '.join(['---:']*8)+' |']
            matrix=surface(values[dataset],metric)
            for column,ratio in enumerate(RATIOS):
                lines.append('| '+str(ratio)+'% | '+' | '.join(repr(float(x)) for x in matrix[:,column])+' |')
    lines += ['', '## 可追溯原始文件','']
    for dataset,title in DATASETS:
        lines += [f'- {title}：[网格与权重指纹](outputs/r051_3d_20260928/{dataset}/grid.json)、[特征来源](outputs/r051_3d_20260928/{dataset}/feature_manifest.json)、[病例排除明细](outputs/r051_3d_20260928/{dataset}/eligibility.json)。每个条件的预测概率、标签掩码、病例ID和逐病例Acc存于该目录 predictions/。']
    lines += ['', '本轮仅进行冻结特征提取与推理；此前自动训练保持暂停。']
    content='\n'.join(lines)+'\n'
    (OUT/'temp_r051.md').write_text(content)
    target=ROOT/'temp.md';temp=ROOT/'temp.md.r051.tmp';temp.write_text(content);temp.replace(target)
    print('已覆盖 temp.md；195个实测条件及6个绘图矩阵已保存。')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--write-tables',action='store_true')
    args=parser.parse_args()
    data=read_data(require_all=not args.preview)
    if not data:raise RuntimeError('尚无完整数据集可绘图')
    plot(data,preview=args.preview)
    if args.write_tables:
        assert not args.preview
        assert (OUT/'temp_before_r051.md').exists(),'必须先备份旧文件'
        write_tables(data)


if __name__=='__main__':main()
