"""只读采样单进程运行器和旧任务，不初始化 CUDA。"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time


def process_matches(pid, fragment, marker):
    try:
        cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().replace('\0',' ')
        return fragment in cmd and marker in cmd
    except (FileNotFoundError,PermissionError):
        return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--legacy',default='{}')
    parser.add_argument('--tickets',default='[]')
    args=parser.parse_args()
    legacy=json.loads(args.legacy);tickets=json.loads(args.tickets)
    workers,done,ticket_states={},{},{}
    for folder in (args.output/'gpu_workers').glob('*'):
        state=folder/'worker_state.json'
        if state.exists():
            obj=json.loads(state.read_text());obj['alive']=process_matches(obj['pid'],'/gpu_worker.py',str(folder));workers[folder.name]=obj
        for ticket in tickets:
            for location in ('inbox','claimed','done'):
                if (folder/location/(ticket+'.json')).exists():ticket_states[ticket]=location
            path=folder/'done'/(ticket+'.json')
            if path.exists():done[ticket]=json.loads(path.read_text())
    legacy_alive={}
    for jid,pid in legacy.items():
        # 远端 launcher_pid 属于本机 SSH；用训练记录里的远端 PID 判断。
        path=args.output/'runs'/jid/'protocol.json'
        if path.exists():pid=json.loads(path.read_text()).get('pid',pid)
        legacy_alive[jid]=process_matches(pid,'/auto_research/train.py',jid)
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
    inventory=[]
    for line in gpu.stdout.strip().splitlines():
        i,free,used,util=(int(s.strip()) for s in line.split(','));inventory.append(dict(index=i,free=free,used=used,utilization=util))
    print(json.dumps(dict(updated=time.time(),workers=workers,done=done,ticket_states=ticket_states,gpus=inventory,
                          disk_free_gib=shutil.disk_usage(args.output).free/2**30,
                          legacy_alive=legacy_alive)))


if __name__=='__main__':main()
