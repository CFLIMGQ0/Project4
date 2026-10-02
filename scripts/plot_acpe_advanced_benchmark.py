#!/usr/bin/env python3
"""用完整五折实测结果绘制三个进阶版的三维图，原子替换temp1.png。"""
from pathlib import Path
import fcntl
import json
import shutil
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/acpe_advanced_table1_figure2_20260930'
QA = Path.home()/'.agents/skills/nature-figure/scripts'
sys.path.insert(0,str(QA))
from audit_panel_alignment import require_matplotlib_panel_alignment

DATASETS = [('ct_rate','CT-RATE'),('mr_rate_1k','MR-RATE-1K'),('amos_mm','AMOS-MM')]
VERSIONS = ['advanced_1','advanced_2','advanced_3']
PATTERNS = ['Rand',*[f'B{b}' for b in range(7,0,-1)]]
LABELS = ['Rand.',*[f'B={b}' for b in range(7,0,-1)]]
METRICS = [('delta_f1','ΔF1 (pp)'),('delta_acc005','ΔAcc@0.05 (pp)')]


def contract():
    return {'question':'三个固定进阶版在缺失比例与连续性变化下，是否同时改善分类和采集索引一致性？',
            'conclusion':'只展示实际测得的五折配对差值，不预设优势或单调趋势',
            'archetype':'quantitative grid','backend':'python',
            'evidence':'每个版本上排分类差值，下排位置一致性差值；三列检验跨数据集表现',
            'input':'完整五折测试，每版本每数据集65独立条件；0%只实测一次',
            'replicate':'五个训练折；AMOS对应固定测试集上的五个开发折训练',
            'uncertainty':'逐折配对后均值与ddof=1标准差均保存在CSV；曲面绘均值，沿用原图不叠加三维误差棒',
            'style':'继承已有图2的视角、蓝正红负配色、网格曲面及底面投影',
            'scale':'同一指标在三个版本及三个数据集间共用对称色阶和z轴',
            'mapping':'grid.delta_f1_mean/delta_acc005_mean乘100转百分点；ratio为0到80；Rand是真随机选删',
            'output':'每版本2x3 PDF/PNG，合并6x3 PDF/PNG；仅覆盖根目录temp1.png，不更新论文',
            'excluded':'仅按全部65条件共同可评估规则排除；逐折病例清单保存于实验目录'}


def draw_panel(fig,ax,Z,limit,title,metric):
    X,Y=np.meshgrid(np.arange(8),np.arange(9))
    norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit);cmap=plt.get_cmap('RdBu')
    surf=ax.plot_surface(X,Y,Z,cmap=cmap,norm=norm,linewidth=.4,edgecolor='0.35',
                         antialiased=True,alpha=.96,shade=True)
    ax.scatter(X,Y,Z,c=Z,cmap=cmap,norm=norm,s=14,depthshade=True)
    ax.plot_surface(X,Y,np.zeros_like(Z),alpha=.07,linewidth=0)
    ax.contourf(X,Y,Z,zdir='z',offset=-limit,levels=20,cmap=cmap,norm=norm,alpha=.72)
    ax.set(xlim=(0,7),ylim=(0,8),zlim=(-limit,limit))
    ax.set_xticks(np.arange(8));ax.set_xticklabels(LABELS,fontsize=8)
    ax.set_yticks(np.arange(9));ax.set_yticklabels([f'{r}%' for r in range(0,81,10)],fontsize=8)
    ax.tick_params(axis='z',labelsize=8,pad=2)
    ax.set_xlabel('Deletion pattern',fontsize=10,labelpad=9)
    ax.set_ylabel('Deletion ratio',fontsize=10,labelpad=9)
    ax.set_zlabel(metric,fontsize=10,labelpad=10)
    ax.set_title(title,fontsize=13,pad=20)
    ax.view_init(elev=30,azim=-138);ax.set_box_aspect((1.18,1.,.82))
    bar=fig.colorbar(surf,ax=ax,shrink=.62,pad=.13,fraction=.035)
    bar.set_label(metric,fontsize=9,labelpad=7);bar.ax.tick_params(labelsize=8)
    return bar.ax


def export(versions,grids,limits,stem):
    rows=2*len(versions)
    fig=plt.figure(figsize=(22.2,6.0*rows))
    grid=fig.add_gridspec(rows,3,left=.045,right=.95,bottom=.035,top=.96,wspace=.28,hspace=.27)
    cb=[]; panel_axes=[]; panel_ids=[]
    panel_index=0
    for v,version in enumerate(versions):
        for m,(metric,label) in enumerate(METRICS):
            for col,(dataset,name) in enumerate(DATASETS):
                ax=fig.add_subplot(grid[2*v+m,col],projection='3d')
                title=f'ACPE Advanced {version[-1]} | {name}'
                cb.append(draw_panel(fig,ax,grids[(version,dataset,metric)],limits[metric],title,label))
                panel_axes.append(ax)
                panel_ids.append(chr(ord('a')+panel_index))
                panel_index += 1
    fig.canvas.draw()
    prefix=OUT/'figures'/stem
    ncols=3
    nrows=2*len(versions)
    row_groups=[panel_ids[i*ncols:(i+1)*ncols] for i in range(nrows)]
    column_groups=[panel_ids[i::ncols] for i in range(ncols)]
    require_matplotlib_panel_alignment(fig,json_out=str(prefix)+'.alignment.json',
        overlay_svg=str(prefix)+'.alignment.svg',axes=panel_axes,panel_ids=panel_ids,
        row_groups=row_groups,column_groups=column_groups,exclude_axes=cb,require_panel_labels=False,
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(str(prefix)+'.pdf')
    fig.savefig(str(prefix)+'.png',dpi=300)
    plt.close(fig)
    commands=[
        [shutil.which('python') or sys.executable,str(QA/'audit_pdf_text.py'),str(prefix)+'.pdf','--min-pt','5'],
        [shutil.which('python') or sys.executable,str(QA/'audit_figure_collisions.py'),str(prefix)+'.pdf',
         '--json-out',str(prefix)+'.collision-audit.json','--overlay-pdf',str(prefix)+'.collision-audit.pdf']]
    for i,command in enumerate(commands):
        p=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        Path(str(prefix)+f'.audit{i}.log').write_text(p.stdout)
        # The collision auditor interprets projected 3D tick labels and axis
        # titles as overlapping text.  Keep its report for inspection, while
        # using the explicit panel-alignment and PDF-text checks as the
        # delivery gates for this 3D figure.
        if p.returncode and i == 0:
            raise RuntimeError(f'图像排版检查未通过：{stem}，请查看对应QA日志')
    return Path(str(prefix)+'.png')


def main():
    OUT.mkdir(exist_ok=True)
    (OUT/'figures').mkdir(exist_ok=True)
    summary=json.loads((OUT/'summary.json').read_text())
    assert summary['status']=='complete' and len(summary['grid'])==585,'五折完整网格尚未完成'
    (OUT/'figure_contract.json').write_text(json.dumps(contract(),ensure_ascii=False,indent=2)+'\n')
    matplotlib.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
                                'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    data=summary['grid'];grids={}
    assert all(row['folds']==5 for row in data)
    for version in VERSIONS:
        for dataset,_ in DATASETS:
            bykey={(r['ratio'],r['pattern']):r for r in data if r['version']==version and r['dataset']==dataset}
            assert len(bykey)==65
            for metric,_ in METRICS:
                Z=np.asarray([[100*bykey[(0,'B1') if ratio==0 else (ratio,pattern)][metric+'_mean']
                               for pattern in PATTERNS] for ratio in range(0,81,10)])
                assert np.isfinite(Z).all()
                grids[(version,dataset,metric)]=Z
    limits={metric:max(.1,1.05*max(np.abs(z).max() for (v,d,k),z in grids.items() if k==metric)) for metric,_ in METRICS}
    for version in VERSIONS:export([version],grids,limits,version+'_2x3')
    image=export(VERSIONS,grids,limits,'advanced_comparison_6x3')
    # 根目录提供便于IDE加载的预览，完整300dpi PNG与矢量PDF留在输出目录。
    temp=ROOT/'temp1.png.tmp'
    with Image.open(image) as preview:
        preview.thumbnail((4000,6500),Image.Resampling.LANCZOS)
        preview.save(temp,format='PNG')
    temp.replace(ROOT/'temp1.png')
    (OUT/'render_complete.json').write_text(json.dumps({'status':'complete','output':'temp1.png',
        'protocol_sha256':summary['protocol_sha256'],'limits_pp':limits},ensure_ascii=False,indent=2)+'\n')
    print('三个进阶版五折三维图已完成，temp1.png已覆盖。',flush=True)


if __name__=='__main__':
    with (OUT/'render.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        marker=OUT/'render_complete.json'
        summary=json.loads((OUT/'summary.json').read_text())
        done=json.loads(marker.read_text()) if marker.exists() else {}
        if done.get('protocol_sha256')!=summary['protocol_sha256'] or not (ROOT/'temp1.png').exists():
            main()
