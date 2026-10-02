"""204 GPU0上的持续五折研究调度器：完整训练、配对排名、有限深度扩展。"""
import argparse
import fcntl
import itertools
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from .common import ROOT, OUT, PYTHON, DATASETS, read, write, identity, key
from .plan import DEPTH, MAX_CANDIDATES


def control_name(config):
    if config.get('structured_training'):
        return 'original_structured_training'
    if config.get('lqd_mode')=='off':
        return 'original_no_lqd'
    if config.get('image_weight'):
        return 'original_visual_aux'
    return 'original_pe'


def specs(candidates,plan):
    # 每个候选优先完成同一数据集五折，然后覆盖其它数据集。
    values=[]
    for candidate in candidates:
        for ds in DATASETS:
            for fold in range(1,6):
                values.append({'id':f"{candidate['id']}_{ds}_f{fold}", 'candidate':candidate['id'],
                               'dataset':ds,'fold':fold,'config':candidate['config'],
                               'protocol_sha256':plan['sha256']})
    return values


def aggregate(candidates,plan):
    all_results={};progress={};completed=[]
    for c in candidates:
        group={}
        for ds in DATASETS:
            for fold in range(1,6):
                p=OUT/'runs'/f"{c['id']}_{ds}_f{fold}"/'result.json'
                if p.exists():
                    result=read(p)
                    assert result['protocol_sha256']==plan['sha256']
                    assert result['config']==c['config'] and result['fold']==fold and result['dataset']==ds
                    assert result['evaluation_split']=='development_validation' and not result['test_evaluated']
                    group[ds,fold]=result
        progress[c['id']]=len(group)
        if len(group)==15:
            all_results[c['name']]=group
            completed.append(c)
    ranking=[]
    for c in completed:
        base_name=control_name(c['config'])
        if base_name not in all_results:continue
        group,base=all_results[c['name']],all_results[base_name]
        datasets={};clean_deltas=[];severe_deltas=[];stds=[]
        for ds in DATASETS:
            differences={}
            for condition in ('clean',)+tuple(key(*v) for v in plan['deletion_conditions']):
                vals=[group[ds,f]['conditions'][condition]['f1']-base[ds,f]['conditions'][condition]['f1'] for f in range(1,6)]
                absolute=[group[ds,f]['conditions'][condition]['f1'] for f in range(1,6)]
                differences[condition]={'delta_mean':statistics.mean(vals),'delta_std':statistics.stdev(vals),
                                        'fold_deltas':vals,'f1_mean':statistics.mean(absolute),'f1_std':statistics.stdev(absolute)}
            severe=[statistics.mean(differences[k]['fold_deltas'][i] for k in plan['severe_conditions']) for i in range(5)]
            clean=differences['clean']['delta_mean'];severe_mean=statistics.mean(severe)
            datasets[ds]={'conditions':differences,'clean_delta':clean,'severe_delta':severe_mean,
                           'severe_std':statistics.stdev(severe),'positive_folds':sum(v>0 for v in severe),
                           'severity_advantage_growth':differences[key(80,1)]['delta_mean']-differences[key(20,1)]['delta_mean'],
                           'continuity_advantage_growth':differences[key(80,1)]['delta_mean']-differences[key(80,8)]['delta_mean']}
            clean_deltas.append(clean);severe_deltas.append(severe_mean);stds.append(statistics.stdev(severe))
        admissible=(min(clean_deltas)>=-.005 and statistics.mean(severe_deltas)>0 and
                    sum(v['severe_delta']>0 and v['positive_folds']>=3 for v in datasets.values())>=2)
        score=.25*statistics.mean(clean_deltas)+.75*statistics.mean(severe_deltas)-.1*statistics.mean(stds)
        ranking.append({'id':c['id'],'name':c['name'],'config':c['config'],'baseline':base_name,
                        'score':score,'admissible':admissible,'datasets':datasets,'kind':c['kind']})
    ranking.sort(key=lambda x:x['score'],reverse=True)
    summary={'updated':time.time(),'protocol_sha256':plan['sha256'],'completed_folds':sum(progress.values()),
             'scheduled_folds':15*len(candidates),'completed_candidates':len(completed),
             'progress':progress,'ranking':ranking,'metric':'固定0.5阈值的开发验证F1，不是测试结果'}
    write(OUT/'summary.json',summary)
    lines=['# ACPE 五折自动研究','',
           f"已完成 {sum(progress.values())}/{15*len(candidates)} 个折任务；{len(completed)}/{len(candidates)} 个完整候选。",'',
           '当前只用204主机GPU0。每方案三个数据集各五折；完整15折后才进入排名。',
           '优先严重连续缺失：60%/80%删除，B=1/4。B越小，固定删除量下连续缺口越长。',
           '主指标为固定0.5阈值的开发验证F1；测试集不参与本轮。AMOS开发标签是自动标注。',
           'CT/MR保留原fold0，剩余患者重分五个开发折；AMOS沿用原五个开发验证折。',
           '本轮13条件先筛选；完整65条件与最终测试尚未执行。',
           '匹配对照：调整训练增强/辅助监督时，Original PE同步使用相同设置。',
           '','| 候选 | 对照 | CT严重缺失ΔF1 | AMOS严重缺失ΔF1 | MR严重缺失ΔF1 | 初筛条件 |',
           '|---|---|---:|---:|---:|---|']
    for r in ranking:
        values=[f"{r['datasets'][ds]['severe_delta']*100:+.2f}" for ds in DATASETS]
        lines.append('| '+ ' | '.join([r['name'],r['baseline'],*values,'通过' if r['admissible'] else '未通过'])+' |')
    if not ranking:lines.append('| 尚无完整可比较候选 | - | - | - | - | - |')
    lines+=['','所有差值为百分点。未通过者保留用于诊断，不当作正面结论。',
            '数据重叠和反复模型选择使开发成绩具有乐观偏差，五折不是独立外部验证。']
    (OUT/'results.md').write_text('\n'.join(lines)+'\n')
    return summary


def extend(candidates,summary,state):
    complete=summary['completed_candidates']
    depth_count=sum(c['kind']=='depth' for c in candidates)
    allowance=min(MAX_CANDIDATES-len(candidates),max(0,(complete-7)//3*3-depth_count),3)
    if allowance<=0:return
    ranked=[x for x in summary['ranking'] if x['config'].get('position') not in ('original_pe','anchored')
            and x['name']!='old_acpe'][:3]
    if not ranked:return
    seen={identity(c['config']) for c in candidates}
    # 优先沿严重缺失收益最好的已完成方案，检验删除训练和大间距置信度。
    changes=[{'structured_training':True},{'gap_confidence':True}]+DEPTH
    for change,parent in itertools.product(changes,ranked):
        cfg={**parent['config'],**change}
        # 不把多种训练干预叠加，确保存在严格匹配的Original PE控制组。
        if cfg.get('structured_training') and (cfg.get('lqd_mode')=='off' or cfg.get('image_weight')):continue
        if identity(cfg) in seen:continue
        number=len(candidates);name=f'depth_{number:03d}'
        candidate={'id':f'c{number:03d}_{name}','name':name,'config':cfg,'kind':'depth','parent':parent['id'],
                   'hypothesis':'完整五折后的单因素扩展：'+str(change),
                   'selection_evidence':{'parent_score':parent['score'],'parent_admissible':parent['admissible'],
                                         'completed_candidates':complete}}
        candidates.append(candidate);seen.add(identity(cfg));allowance-=1
        state.setdefault('decisions',[]).append({'time':time.time(),**candidate})
        if allowance<=0:break
    write(OUT/'candidates.json',candidates)


def free_memory():
    p=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],
                     text=True,capture_output=True,timeout=15,check=True)
    return {int(row.split(',')[0]):int(row.split(',')[1]) for row in p.stdout.splitlines()}


def launch(module,args,log):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(ROOT/'src'),
                                    OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    handle=log.open('a')
    p=subprocess.Popen([PYTHON,'-m',module,*args],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
    handle.close()
    return p


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--aggregate',action='store_true')
    args=parser.parse_args()
    plan=read(OUT/'protocol.json');candidates=read(OUT/'candidates.json')
    if args.aggregate:
        aggregate(candidates,plan);return
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(OUT/'controller_identity.json',{'pid':os.getpid(),'started':time.time(),'host':'204','gpu':0})
    state=read(OUT/'state.json') if (OUT/'state.json').exists() else {'attempts':{},'failures':{},'decisions':[]}
    active=None;active_id=None;prep_attempts=0;last_aggregate=0
    (OUT/'logs').mkdir(exist_ok=True)
    while True:
        policy=read(OUT/'control.json') if (OUT/'control.json').exists() else {'paused':False}
        if active is not None and active.poll() is not None:
            code=active.returncode
            if code!=0:
                state['failures'][active_id]={'returncode':code,'time':time.time()}
            elif active_id in state['failures']:
                del state['failures'][active_id]
            print(f'结束 {active_id} exit={code}',flush=True)
            active=None;active_id=None
        state['updated']=time.time();state['active']={'id':active_id,'pid':active.pid} if active else None
        state['paused']=policy.get('paused',False)
        write(OUT/'state.json',state)
        if time.time()-last_aggregate>30:
            summary=aggregate(candidates,plan)
            extend(candidates,summary,state)
            last_aggregate=time.time()
        if active is None and not state['paused']:
            ready=all((OUT/'features'/ds/'ready.json').exists() for ds in DATASETS)
            free=free_memory().get(0,0)
            disk=shutil.disk_usage(ROOT).free/2**30
            if not ready:
                if prep_attempts<3 and free>=3500 and disk>=12:
                    active=launch('acpe_fivefold.prepare',['--encode'],OUT/'logs/feature_preparation.log')
                    active_id='feature_preparation';prep_attempts+=1
                    print(f'启动删除视图特征补提取 PID={active.pid}',flush=True)
            elif free>=1800 and disk>=12:
                pending=[]
                for spec in specs(candidates,plan):
                    result=OUT/'runs'/spec['id']/'result.json'
                    if result.exists() or state['attempts'].get(spec['id'],0)>=3:continue
                    pending.append(spec)
                # 广度未完成时也允许完整五折证据触发少量优先深度任务。
                depth_ids={c['id'] for c in candidates if c['kind']=='depth'}
                pending.sort(key=lambda x:0 if x['candidate'] in depth_ids else 1)
                if pending:
                    spec=pending[0];path=OUT/'jobs'/(spec['id']+'.json')
                    if path.exists():assert read(path)==spec
                    else:write(path,spec)
                    state['attempts'][spec['id']]=state['attempts'].get(spec['id'],0)+1
                    active=launch('acpe_fivefold.train',['--job',str(path)],OUT/'logs'/(spec['id']+'.log'))
                    active_id=spec['id']
                    print(f'启动 {active_id} PID={active.pid}',flush=True)
                elif summary['completed_folds']==15*len(candidates) and len(candidates)>=MAX_CANDIDATES:
                    write(OUT/'completion.json',{'time':time.time(),'summary':summary,'test_evaluated':False})
                    break
        time.sleep(5)


if __name__=='__main__':main()
