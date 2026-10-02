"""本轮独立目录和不可变协议的公共工具。"""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get('PROJECT4_ROOT', Path(__file__).resolve().parents[2]))
OUT = ROOT / 'outputs/acpe_fivefold_research_20260929'
BASE = ROOT / 'outputs/acpe_research_200_20260928/workspace/src'
sys.path.insert(0, str(BASE))
DATASETS = {'ct_rate': 'ct_rate_680', 'amos_mm': 'amos_mm', 'mr_rate_1k': 'mr_rate_1k'}
CONDITIONS = ((0, 1),) + tuple((r, b) for r in (20, 40, 60, 80) for b in (1, 4, 8))
PYTHON = '/home/Lim/conda/envs/myenv/bin/python'
REMOTE = '/home/Lim/Project4'
SSH = ['ssh', '-i', '/home/Lim/.ssh/id_ed25519_project4_pool', '-o', 'BatchMode=yes',
       '-o', 'ConnectTimeout=8', 'Lim@172.16.170.202']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def key(ratio, blocks):
    return f'd{ratio:02d}_b{blocks}'


def folder(dataset):
    return ROOT / 'outputs' / DATASETS[dataset] / 'experiment'
