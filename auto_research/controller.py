"""两主机共享 GPU 自适应排队；完成一轮需三数据集训练、评估、分析齐全。"""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from plan import BREADTH, DATASETS, DEPTH

PYTHON = "/home/Lim/conda/envs/myenv/bin/python"
ENV_LIB = "/home/Lim/conda/envs/myenv/lib"
SSH = ["ssh", "-i", "/home/Lim/.ssh/id_ed25519_project4_pool", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=30", "Lim@172.16.170.202"]
REMOTE = "/home/Lim/Project4/outputs/acpe_research_200_20260928"
ROOT = Path(__file__).resolve().parents[2]


class AdoptedProcess:
    """只接管已登记任务的监控，不重启也不终止训练。"""
    def __init__(self, pid, job_id):
        self.pid, self.job_id = pid, job_id

    def poll(self):
        try:
            commandline=Path(f"/proc/{self.pid}/cmdline").read_bytes().decode().replace("\0", " ")
            return None if self.job_id in commandline else 0
        except FileNotFoundError:
            return 0


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def key(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def command(host, cmd, timeout=20):
    argv = ["bash", "-c", cmd] if host == "204" else SSH + [cmd]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=True).stdout


def inventory(host):
    raw = command(host, "nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv,noheader,nounits")
    return [dict(index=int(x[0]), free=int(x[1]), used=int(x[2]), utilization=int(x[3]))
            for line in raw.strip().splitlines() if len(x := line.split(",")) == 4]


def add_round(state, name, config, hypothesis, kind="breadth", parent=None):
    number = len(state["rounds"]) + 1
    identity = f"r{number:03d}_{name}"
    state["rounds"].append(dict(id=identity, number=number, name=name, config=config, hypothesis=hypothesis,
                                kind=kind, parent=parent, status="queued", created=time.time()))
    for dataset in DATASETS:
        identity_job = identity + "_" + dataset
        state["jobs"][identity_job] = dict(id=identity_job, round=identity, dataset=dataset,
                                           config=config, seed=42, fold=0, status="queued", attempts=0)


def analyze(state):
    for research in state["rounds"]:
        if research["status"] == "completed":
            continue
        statuses=[state["jobs"][research["id"]+"_"+dataset]["status"] for dataset in DATASETS]
        research["status"]=("awaiting_reference" if all(s=="completed" for s in statuses)
                            else "running" if any(s in {"running","completed"} for s in statuses)
                            else "failed" if any(s=="failed" for s in statuses) else "queued")
    reference = state["rounds"][0]["id"]
    baseline = [state["jobs"][reference+"_"+dataset] for dataset in DATASETS]
    if not all(j["status"] == "completed" for j in baseline):
        return
    for research in state["rounds"]:
        jobs = [state["jobs"][research["id"]+"_"+d] for d in DATASETS]
        if research["status"] == "completed" or not all(j["status"] == "completed" for j in jobs):
            continue
        comparisons = {}
        for job, ref in zip(jobs, baseline):
            conditions = job["result"]["conditions"]
            base_conditions = ref["result"]["conditions"]
            comparisons[job["dataset"]] = {name: value["macro_f1"]-base_conditions[name]["macro_f1"]
                                            for name, value in conditions.items() if name in base_conditions}
        clean = [x["clean"] for x in comparisons.values()]
        robust = [sum(x[n] for n in ("d50_b1", "d80_b1", "d80_b8"))/3 for x in comparisons.values()]
        admissible = min(clean) >= -.01 and sum(x > 0 for x in robust) >= 2
        score = sum(robust)/3 - sum(max(0, -x-.01) for x in clean)/3
        research.update(status="completed", completed=time.time(), differences=comparisons,
                        robust_score=score, admissible=admissible,
                        decision=("候选：至少两数据集删除表现改善，继续换验证折/种子复核（开发集内）" if admissible and score > 0
                                  else "未满足跨数据集筛选条件；保留用于机制诊断，不写为有效方法"))
        if research["parent"]:
            parent = next(r for r in state["rounds"] if r["id"] == research["parent"])
            research["parent_score_change"] = score - parent.get("robust_score", 0)
    original = next((r for r in state["rounds"] if r["name"] == "original_pe"), None)
    if original and original["status"] == "completed":
        for research in state["rounds"]:
            if research["status"] != "completed":
                continue
            differences = {}
            for dataset in DATASETS:
                result = state["jobs"][research["id"]+"_"+dataset]["result"]["conditions"]
                comparison = state["jobs"][original["id"]+"_"+dataset]["result"]["conditions"]
                differences[dataset] = {name: value["macro_f1"]-comparison[name]["macro_f1"]
                                        for name,value in result.items() if name in comparison}
            research["original_pe_differences"] = differences


def expand(state, target):
    # 先有六个完整的三数据集证据，再动态展开深度；保持足够的待运行任务供六卡补位。
    pending_rounds = sum(r["status"] != "completed" for r in state["rounds"])
    completed = [r for r in state["rounds"][1:] if r["status"] == "completed"]
    if pending_rounds >= 36 or len(completed) < 6 or len(state["rounds"]) >= target:
        return
    existing = {key(r["config"]) for r in state["rounds"]}
    parents = sorted(completed, key=lambda r: r["robust_score"], reverse=True)
    # 保留前三名及不同损失/结构家族，负结果也可以触发诊断深度，但绝不称其提升。
    pool = parents[:3]
    for candidate in parents:
        if candidate["kind"] == "breadth" and candidate not in pool:
            pool.append(candidate)
        if len(pool) >= 8:
            break
    offset = state.get("depth_cursor", 0)
    changes = [(field, value, reason) for field, values, reason in DEPTH for value in values]
    for k in range(len(changes)):
        field, value, reason = changes[(offset+k) % len(changes)]
        for parent in pool:
            config = deepcopy(parent["config"])
            if config.get(field) == value:
                continue
            if field == "lqd_weight" and config.get("lqd_mode") == "off":
                continue
            if parent["name"] in {"no_pe", "original_pe", "anchored_control"}:
                continue
            if field == "placement" and config.get("route") == "absolute" and value.startswith("relative"):
                continue
            config[field] = value
            if key(config) in existing:
                continue
            statement = f"父方案 {parent['id']} 的三数据集验证结果已完成；{reason}。父方案状态：{parent['decision']}"
            add_round(state, f"depth_{field}_{key(config)[:5]}", config, statement, "depth", parent["id"])
            state["depth_cursor"] = (offset+k+1) % len(changes)
            return


def add_confirmation(state):
    # 确认实验仍不读取保留测试集；换验证折和随机种子，单独报告，不并入筛选均值。
    existing = {j["round"] for j in state["jobs"].values() if j.get("confirmation")}
    # 最多六个优先方案复核，给先完成与后出现的候选都保留机会。
    if len(existing - {state["rounds"][0]["id"]}) >= 6:
        return
    ranked = sorted([r for r in state["rounds"] if r.get("admissible") and r.get("robust_score", 0) > 0],
                    key=lambda r:r["robust_score"], reverse=True)
    for research in ranked[:3]:
        if research["id"] in existing:
            continue
        for chosen in (state["rounds"][0], research):
            for dataset in DATASETS:
                jid = chosen["id"] + "_" + dataset + "_confirm"
                if jid not in state["jobs"]:
                    state["jobs"][jid] = dict(id=jid, round=chosen["id"], dataset=dataset,
                                               config=chosen["config"], seed=137, fold=1, confirmation=True,
                                               status="queued", attempts=0)
        return


def report(out, state):
    complete = sum(r["status"] == "completed" for r in state["rounds"])
    jobs = list(state["jobs"].values())
    lines = ["# ACPE 系统 Auto Research", "",
             f"目标：{state['target_rounds']} 轮假设检验；已完成 {complete} 轮（其中共同参考 1 轮，完成时计入）。",
             f"已登记 {len(state['rounds'])} 轮；训练任务完成 {sum(j['status']=='completed' for j in jobs)}，运行 {sum(j['status']=='running' for j in jobs)}，待运行 {sum(j['status']=='queued' for j in jobs)}，失败 {sum(j['status']=='failed' for j in jobs)}。", "",
             "每轮必须完成三个数据集的完整训练、验证集删除评估和分析。折数、epoch 数不充作轮数。", "",
             "当前是缓存图像包上的验证集机制筛选；不是论文的原体积删片实验，不能直接填入论文表格。测试集不参与本轮选择。", "",
             "深度方案依据已完成结果生成；不保证发现提升。自动报告只给筛选证据，不作统计显著性或 SOTA 宣称。", "",
             "当前禁用 GPU：" + json.dumps(state.get("resource_policy", {}).get("excluded_gpus", {}), ensure_ascii=False) + "。", "",
             "|轮次|类别|假设|状态/判断|", "|---|---|---|---|"]
    for r in state["rounds"]:
        lines.append(f"|{r['id']}|{r['kind']}|{r['hypothesis']}|{r.get('decision',r['status'])}|")
    lines += ["", "## 运行任务", "", "|任务|主机/GPU|启动时间|", "|---|---|---|"]
    for j in jobs:
        if j["status"] == "running":
            lines.append(f"|{j['id']}|{j['host']}/{j['gpu']}|{time.strftime('%m-%d %H:%M:%S',time.localtime(j['started']))}|")
    lines += ["", "## 已完成方案相对共同参考的验证增量", "", "|方案|CT-RATE clean / deletion|AMOS-MM clean / deletion|MR-RATE-1K clean / deletion|", "|---|---|---|---|"]
    for r in sorted([r for r in state["rounds"] if r["status"] == "completed"], key=lambda r:r.get("robust_score",0), reverse=True):
        cells = []
        for d in DATASETS:
            diff = r["differences"][d]
            robust = sum(diff[n] for n in ("d50_b1","d80_b1","d80_b8"))/3
            cells.append(f"{diff['clean']:+.4f} / {robust:+.4f}")
        lines.append("|" + r["id"] + "|" + "|".join(cells) + "|")
    lines += ["", "Original PE 在相同新协议下完成后，逐数据集差值写入 state.json 的 original_pe_differences；当前表格参考是本轮完整 ACPE。", "",
              "换折/种子复核属于开发集内的稳定性检查，不是全新独立外部测试。所有阈值只在干净验证集选择。"]
    (out / "status.md").write_text("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-rounds", type=int, default=200)
    parser.add_argument("--resume-active", action="store_true")
    args=parser.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/"controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.write(str(os.getpid())); lock.flush()
    state_path=out/"state.json"
    active={}
    if state_path.exists():
        state=json.loads(state_path.read_text())
        # 重启先检查已完成结果；仍活跃的前任任务不可重复启动。
        for job in state["jobs"].values():
            if job["status"] == "running":
                if not args.resume_active:
                    raise RuntimeError("登记中存在活跃任务；请先核实原进程，禁止重复执行")
                handle=(out/"logs"/(job["id"]+f".attempt{job['attempts']}.log")).open("a")
                active[job["id"]]=(AdoptedProcess(job["launcher_pid"],job["id"]),handle)
    else:
        state=dict(target_rounds=args.target_rounds, rounds=[], jobs={}, created=time.time())
        for name,config,hypothesis in BREADTH:
            add_round(state,name,config,hypothesis)
    peak=float(state.get("memory_estimate_mib",1500.))
    oom_floor=float(state.get("oom_memory_floor_mib",0))
    (out/"jobs").mkdir(exist_ok=True)
    (out/"logs").mkdir(exist_ok=True)
    print("调度启动：遵守 resource_policy.json 的 GPU 禁用策略，按空闲显存动态调度",flush=True)
    while True:
        for job in state["jobs"].values():
            if job["status"] == "failed" and "无法构造" in job.get("failure_tail", "") and not job.get("eligibility_recovery"):
                job.update(status="queued",eligibility_recovery=True)
                print("复用完整训练，仅恢复统一样本子集评估："+job["id"],flush=True)
        for jid, (process,logfile) in list(active.items()):
            rc=process.poll()
            if rc is None:
                continue
            logfile.close()
            job=state["jobs"][jid]
            folder=out/"runs"/jid
            if job["host"] == "202":
                folder.mkdir(parents=True,exist_ok=True)
                rsync=["rsync","-az","-e",shlex.join(SSH[:-1]),f"Lim@172.16.170.202:{REMOTE}/runs/{jid}/",str(folder)+"/"]
                sync=subprocess.run(rsync,capture_output=True,text=True)
                if sync.returncode:
                    job["sync_error"]=sync.stderr[-1500:]
            result=folder/"result.json"
            if rc == 0 and result.exists():
                job["result"]=json.loads(result.read_text())
                job["status"]="completed"
                print("完成 "+jid,flush=True)
            else:
                logtext=(out/"logs"/(jid+f".attempt{job['attempts']}.log")).read_text(errors="replace")
                oom="out of memory" in logtext.lower()
                if oom:
                    oom_floor=max(peak*1.25,1800)
                    state["oom_memory_floor_mib"]=oom_floor
                retry=job["attempts"] < (3 if oom else 2)
                job["status"]="queued" if retry else "failed"
                job["failure_tail"]=logtext[-3000:]
                print(f"任务退出 {jid} rc={rc} OOM={oom} 状态={job['status']}",flush=True)
            job["finished"]=time.time()
            del active[jid]
        # CUDA context/库开销加在实际 PyTorch 预留峰值之上。保留 350MiB，
        # 不以首次保守估算长期限制已证明较小的任务并发。
        measured=[j["result"]["peak_reserved_mib"] for j in state["jobs"].values() if j.get("result")]
        if measured:
            peak=max(800.,max(measured)+350,oom_floor)
        analyze(state)
        # 新机制观察可通过显式 inbox 加入，保留来源和假设，不重写已有配置。
        inbox=out/"inbox.json"
        if inbox.exists():
            imported=set(state.get("imported_hypotheses",[]))
            for proposal in json.loads(inbox.read_text()):
                if proposal["name"] not in imported and len(state["rounds"]) < args.target_rounds:
                    add_round(state,proposal["name"],proposal["config"],proposal["hypothesis"],"evidence_driven")
                    imported.add(proposal["name"])
            state["imported_hypotheses"]=sorted(imported)
        for _ in range(6):
            expand(state,args.target_rounds)
        add_confirmation(state)
        queued=[j for j in state["jobs"].values() if j["status"] == "queued"]
        # 确认任务占优先队列，同时保留广度/深度并行的普通队列。
        def queue_priority(job):
            history=out/"runs"/job["id"]/"history.json"
            evaluation_pending=False
            if history.exists():
                evaluation_pending=len(json.loads(history.read_text())) == job["config"].get("epochs",30)
            return (not evaluation_pending, job["round"]!=state["rounds"][0]["id"],
                    not job.get("confirmation",False), j_order[job["id"]])
        j_order={jid:i for i,jid in enumerate(state["jobs"])}
        queued.sort(key=queue_priority)
        inventories={}
        policy_path=out/"resource_policy.json"
        policy=json.loads(policy_path.read_text()) if policy_path.exists() else {"excluded_gpus":{}}
        state["resource_policy"]=policy
        if queued:
            for host in ("204","202"):
                try:
                    disks=shutil.disk_usage(out).free/2**30 if host=="204" else float(command(host,"df -BG --output=avail /home/Lim/Project4 | tail -1").strip().rstrip("G"))
                    if disks < 12:
                        print(f"{host} 磁盘余量不足12GiB，暂停新任务",flush=True)
                        continue
                    mem=command(host,"awk '/MemAvailable/ {print $2}' /proc/meminfo")
                    available_ram=float(mem.strip())/1024
                    gpus=inventory(host)
                    inventories[host]=gpus
                    for gpu in sorted(gpus,key=lambda g:g["free"],reverse=True):
                        if gpu["index"] in policy.get("excluded_gpus",{}).get(host,[]):
                            continue
                        running=[j for j in state["jobs"].values() if j["status"]=="running" and j["host"]==host and j["gpu"]==gpu["index"]]
                        inflight=sum(peak for j in running if time.time()-j["started"] < 90)
                        free=gpu["free"]-inflight
                        # 256MiB 仅为外部进程波动预留；不设15/18GB固定上限。
                        while queued and free >= peak+256 and available_ram >= 2500:
                            job=queued.pop(0)
                            jid=job["id"]
                            spec={k:job[k] for k in ("id","round","dataset","config","seed","fold")}
                            path=out/"jobs"/(jid+".json")
                            write(path,spec)
                            if host=="202":
                                encoded=json.dumps(spec,ensure_ascii=False)
                                put="mkdir -p "+shlex.quote(REMOTE+"/jobs")+" && cat > "+shlex.quote(REMOTE+"/jobs/"+jid+".json")
                                subprocess.run(SSH+[put],input=encoded,text=True,check=True,timeout=20)
                            code_root=str(out/"workspace") if host=="204" else REMOTE+"/workspace"
                            data_root=str(ROOT) if host=="204" else "/home/Lim/Project4"
                            jobpath=str(path) if host=="204" else REMOTE+"/jobs/"+jid+".json"
                            resultdir=str(out/"runs"/jid) if host=="204" else REMOTE+"/runs/"+jid
                            argv=[PYTHON,"-u",code_root+"/src/auto_research/train.py","--root",data_root,"--job",jobpath,"--output",resultdir]
                            environment=f"CUDA_VISIBLE_DEVICES={gpu['index']} LD_LIBRARY_PATH={ENV_LIB} OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 "
                            cmd="exec env "+environment+shlex.join(argv)
                            job.update(status="running",host=host,gpu=gpu["index"],started=time.time(),attempts=job["attempts"]+1)
                            logfile=(out/"logs"/(jid+f".attempt{job['attempts']}.log")).open("w")
                            process=subprocess.Popen(["bash","-c",cmd] if host=="204" else SSH+[cmd],stdout=logfile,stderr=subprocess.STDOUT)
                            job["launcher_pid"]=process.pid
                            active[jid]=(process,logfile)
                            free-=peak
                            available_ram-=2500
                            write(state_path,state)
                            print(f"启动 {jid} host={host} gpu={gpu['index']} 预留={peak:.0f}MiB",flush=True)
                except Exception as error:
                    print(f"{host} 调度暂缓：{type(error).__name__}: {error}",flush=True)
        state["updated"]=time.time()
        state["gpu_inventory"]=inventories
        state["memory_estimate_mib"]=peak
        write(state_path,state)
        report(out,state)
        if not active and not queued:
            completed=sum(r["status"]=="completed" for r in state["rounds"])
            if completed >= args.target_rounds:
                print("已完成预定轮数，结束自动探索；仍需人工解释和正式协议确认。",flush=True)
                return
            if any(j["status"]=="failed" for j in state["jobs"].values()):
                print("存在失败任务，等待修复；不把失败轮计作完成。",flush=True)
        time.sleep(20)


if __name__=="__main__":
    main()
