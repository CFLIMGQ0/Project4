"""单 CUDA 进程调度入口；沿用研究计划，自动收尾原多进程任务。"""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time
import os

import controller as legacy


def host_output(host,out):
    return str(out) if host=='204' else legacy.REMOTE


def probe(host,out,state):
    old={j['id']:j['launcher_pid'] for j in state['jobs'].values()
         if j['status']=='running' and j['host']==host and j.get('backend')!='single_process'}
    tickets=[j['ticket'] for j in state['jobs'].values()
             if j['status']=='running' and j['host']==host and j.get('backend')=='single_process']
    base=host_output(host,out)
    argv=[legacy.PYTHON,base+'/workspace/src/auto_research/runtime_probe.py','--output',base,
          '--legacy',json.dumps(old),'--tickets',json.dumps(tickets)]
    return json.loads(legacy.command(host,shlex.join(argv),timeout=30))


def put(host,path,value):
    if host=='204':
        legacy.write(Path(path),value)
    else:
        program="import sys,pathlib,json; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(sys.stdin.read()); t.replace(p)"
        cmd=shlex.join([legacy.PYTHON,'-c',program,path])
        subprocess.run(legacy.SSH+[cmd],input=json.dumps(value,ensure_ascii=False),text=True,check=True,timeout=20)


def retrieve(out,job):
    folder=out/'runs'/job['id']
    if job['host']=='202':
        folder.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-az','-e',shlex.join(legacy.SSH[:-1]),
                         f"Lim@172.16.170.202:{legacy.REMOTE}/runs/{job['id']}/",str(folder)+'/'],
                       capture_output=True,text=True,check=True,timeout=90)
        if job.get('backend')=='single_process':
            name=f"{job['id']}.attempt{job['attempts']}.log"
            subprocess.run(['rsync','-az','-e',shlex.join(legacy.SSH[:-1]),
                            f"Lim@172.16.170.202:{legacy.REMOTE}/logs/{name}",str(out/'logs'/name)],
                           capture_output=True,text=True,timeout=30)
    return folder/'result.json'


def launch_worker(host,gpu,out,state):
    base=host_output(host,out);wid=f'{host}_gpu{gpu}'
    queue=base+'/gpu_workers/'+wid
    argv=[legacy.PYTHON,'-u',base+'/workspace/src/auto_research/gpu_worker.py',
          '--queue-dir',queue,'--host',host,'--gpu',str(gpu)]
    cmd='exec env '+f'CUDA_VISIBLE_DEVICES={gpu} LD_LIBRARY_PATH={legacy.ENV_LIB} OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 '+shlex.join(argv)
    log=(out/'logs'/(wid+'.worker.log')).open('a')
    process=subprocess.Popen(['bash','-c',cmd] if host=='204' else legacy.SSH+[cmd],stdout=log,stderr=subprocess.STDOUT)
    log.close()
    state.setdefault('workers',{})[wid]=dict(host=host,gpu=gpu,launcher_pid=process.pid,started=time.time(),queue_dir=queue)
    print(f'启动唯一 GPU worker {wid}，启动器 PID={process.pid}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    lock=(out/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.write(str(os.getpid()));lock.flush()
    path=out/'state.json';state=json.loads(path.read_text());state['execution_mode']='single_process_per_gpu'
    print('单进程调度接管；每卡旧任务全部完成后启动一个新 worker。',flush=True)
    while True:
        observations={}
        for host in ('204','202'):
            try:observations[host]=probe(host,out,state)
            except Exception as error:print(f'{host} 状态读取稍后重试：{error}',flush=True)
        for job in list(state['jobs'].values()):
            if job['status']!='running' or job['host'] not in observations:continue
            obs=observations[job['host']];event=None
            if job.get('backend')=='single_process':
                event=obs['done'].get(job['ticket'])
                worker=obs['workers'].get(job['worker_id'],{})
                if event is None:
                    if worker.get('alive'):
                        if job['ticket'] not in obs['ticket_states'] and 'submission' in job:
                            try:put(job['host'],job['submission']['path'],job['submission']['request'])
                            except Exception as error:print(f'重新提交票据稍后重试：{error}',flush=True)
                        continue
                    started=state.get('workers',{}).get(job['worker_id'],{}).get('started',0)
                    if time.time()-started < 90:continue
                    event=dict(status='failed',error='GPU worker 已退出')
            elif obs['legacy_alive'].get(job['id'],False):
                continue
            try:
                result=retrieve(out,job)
            except Exception as error:
                job['sync_error']=str(error);continue
            if result.exists():
                job.update(status='completed',result=json.loads(result.read_text()),finished=time.time())
                print('完成 '+job['id'],flush=True)
            else:
                error=event.get('error','无结果') if event else '旧任务退出且未产生结果'
                job.update(status='queued' if job['attempts']<4 else 'failed',failure_tail=error,finished=time.time())
                print('任务待处理 '+job['id']+' '+job['status'],flush=True)
        legacy.analyze(state)
        for _ in range(6):legacy.expand(state,state['target_rounds'])
        legacy.add_confirmation(state)
        policy=json.loads((out/'resource_policy.json').read_text());state['resource_policy']=policy
        state['gpu_inventory']={h:o['gpus'] for h,o in observations.items()}
        queued=[j for j in state['jobs'].values() if j['status']=='queued']
        queued.sort(key=lambda j:(not j.get('confirmation',False),j['attempts']))
        for host,obs in observations.items():
            if obs['disk_free_gib']<12:continue
            for gpu in sorted(obs['gpus'],key=lambda g:g['free'],reverse=True):
                index=gpu['index'];wid=f'{host}_gpu{index}'
                if index in policy.get('excluded_gpus',{}).get(host,[]):continue
                old=[j for j in state['jobs'].values() if j['status']=='running' and j['host']==host
                     and j['gpu']==index and j.get('backend')!='single_process']
                if old:continue
                worker=obs['workers'].get(wid,{})
                if not worker.get('alive'):
                    registered=state.get('workers',{}).get(wid,{})
                    if time.time()-registered.get('started',0)<90:continue
                    if queued and gpu['free']>1400:
                        launch_worker(host,index,out,state);legacy.write(path,state)
                    continue
                state.setdefault('workers',{}).setdefault(wid,dict(host=host,gpu=index,
                      launcher_pid=worker['pid'],started=worker['updated'],queue_dir=host_output(host,out)+'/gpu_workers/'+wid))['pid']=worker['pid']
                # 在 worker 常驻显存基础上留出正在装载的任务，不为每项创建进程。
                assigned=[j for j in state['jobs'].values() if j['status']=='running' and j.get('worker_id')==wid]
                outstanding=max(0,len(assigned)-len(worker['tasks']))
                slots=max(0,min(6,worker['additional_capacity']-outstanding))
                while queued and slots>0:
                    job=queued.pop(0);jid=job['id'];attempt=job['attempts']+1
                    ticket=f'{jid}.attempt{attempt}';base=host_output(host,out)
                    spec={k:job[k] for k in ('id','round','dataset','config','seed','fold')}
                    legacy.write(out/'jobs'/(jid+'.json'),spec)
                    if host=='202':
                        try:put(host,base+'/jobs/'+jid+'.json',spec)
                        except Exception as error:
                            print(f'远端任务配置提交稍后重试：{error}',flush=True)
                            break
                    request=dict(ticket=ticket,job_id=jid,root=str(legacy.ROOT) if host=='204' else '/home/Lim/Project4',
                                 job=base+'/jobs/'+jid+'.json',output=base+'/runs/'+jid,
                                 log=base+'/logs/'+ticket+'.log')
                    job.update(status='running',host=host,gpu=index,started=time.time(),attempts=attempt,
                               backend='single_process',worker_id=wid,ticket=ticket,
                               launcher_pid=state['workers'][wid]['launcher_pid'],worker_pid=worker['pid'],
                               submission=dict(path=base+'/gpu_workers/'+wid+'/inbox/'+ticket+'.json',request=request))
                    # 先登记任务，再原子提交票据。重启时同一票据不会再次训练。
                    legacy.write(path,state)
                    try:put(host,job['submission']['path'],request)
                    except Exception as error:print(f'票据提交稍后重试：{error}',flush=True)
                    slots-=1
                    print(f'入队内部实验 {jid} -> {wid} / PID {worker["pid"]}',flush=True)
        if not any(j['status']=='running' and j.get('backend')!='single_process' for j in state['jobs'].values()):
            policy['transition']='single_process_active';legacy.write(out/'resource_policy.json',policy)
        state['updated']=time.time();legacy.write(path,state);legacy.report(out,state)
        running=sum(j['status']=='running' for j in state['jobs'].values())
        waiting=sum(j['status']=='queued' for j in state['jobs'].values())
        complete=sum(r['status']=='completed' for r in state['rounds'])
        if not running and not waiting and complete>=state['target_rounds']:
            for wid,worker in state.get('workers',{}).items():
                put(worker['host'],worker['queue_dir']+'/control.json',dict(stop_when_idle=True))
            print('研究队列全部完成，释放 GPU worker。',flush=True);return
        time.sleep(5)


if __name__=='__main__':main()
