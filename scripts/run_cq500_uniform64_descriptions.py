#!/usr/bin/env python3
"""逐例均匀采样最多64张CT，使用本地Qwen生成单段研究用所见试稿。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
POLICY = "uniform_per_case_v1"


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def uniform_indices(total: int, requested: int) -> list[int]:
    """在首尾层面之间等间距选取；不足目标数时全部保留，不重复补帧。"""
    if total <= 0 or requested <= 0:
        return []
    count = min(total, requested)
    if count == total:
        return list(range(total))
    if count == 1:
        return [(total - 1) // 2]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def choose_frames(manifest: dict, requested: int) -> list[dict]:
    selected = uniform_indices(manifest["rendered_frames"], requested)
    result, offset = [], 0
    for series in manifest["series"]:
        length = series["frame_count"]
        for index in selected:
            if offset <= index < offset + length:
                local_index = index - offset
                result.append({
                    "global_index": index,
                    "series_index": series["series_index"],
                    "frame_index": local_index,
                    "cache": series["cache"],
                    "source": series["sources"][local_index],
                    "axial": series["axial"],
                    "standard_left_right": series["standard_left_right"],
                })
        offset += length
    assert offset == manifest["rendered_frames"]
    assert [r["global_index"] for r in result] == selected
    return result


def input_prompt(manifest: dict, selected: list[dict]) -> str:
    spans = []
    for series in manifest["series"]:
        locations = [i + 1 for i, row in enumerate(selected) if row["series_index"] == series["series_index"]]
        if not locations:
            continue
        orientation = (
            "横断位，每个窗内图像左侧对应患者右侧，图像右侧对应患者左侧"
            if series["axial"] and series["standard_left_right"]
            else "本段不能统一确定左右侧，不确定时只描述图像侧别"
        )
        spans.append(f"第{locations[0]}至{locations[-1]}帧为一个序列的采样层面，{orientation}。")
    prompt = (
        f"以下为同一患者检查中均匀选取的{len(selected)}张CT图像，以视频容器依次提供。"
        "帧顺序代表切片展示顺序，不代表病情随时间变化；采样层面之间可能存在间隔。"
        "每帧左、中、右三个窗依次为同一切片的脑窗、硬膜下窗、骨窗。"
        "不同序列可能重复显示相同部位，不应把重复显示理解为新增病变。"
        + "".join(spans)
        + "请综合全部提供的图像，直接写唯一一段约200字的中文影像所见。"
        "只依据可见位置、密度、形态及结构关系；对采样未覆盖或显示不清的内容保留不确定性。"
        "不输出标题、条目、病名、诊断结论、治疗建议，不添加无依据的尺寸或病史。"
    )
    if len(selected) < 10:
        prompt += "本例本地仅提供少量层面，请明确观察范围有限，文字可以短于200字。"
    if manifest["failed_files"]:
        prompt += "本例部分源图像无法解码，涉及显示范围时应保留观察限制。"
    return prompt


def build_protocol(args: argparse.Namespace) -> dict:
    # 不加载torch即可读取原始提示词，避免协调进程重复导入大模型依赖。
    import ast
    module = ast.parse((Path(__file__).parent / "generate_cq500_image_descriptions.py").read_text())
    values = {}
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "USER_PROMPT":
                values["user_prompt"] = ast.literal_eval(node.value)
            if isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT":
                values["system_prompt"] = values["user_prompt"] + ast.literal_eval(node.value.right)
    spec = {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "model_revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "sampling_policy": POLICY, "sample_frames": args.sample_frames,
        "quantization": "int8", "max_new_tokens": args.max_new_tokens,
        "input_mode": "ordered_three_window_video", "windows_per_frame": 3,
        "source_sequence_order": "manifest_series_order_then_spatial_position",
        "automatic_rewrites": 0, "generation_calls_per_case": 1,
        "input_prompt_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        **values,
    }
    digest = hashlib.sha256(json.dumps(spec, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return {
        **spec, "protocol_sha256": digest, "expected_cases": args.expected_cases,
        "model_path": str(args.model_path), "devices": args.devices,
        "ground_truth_read": False, "prediction_files_read": False,
        "synthetic": True, "review_status": "not_reviewed",
        "prior_visual_quality_gate": "failed",
        "mode": "user_requested_research_drafts_with_known_quality_limitations",
        "note": "保留既有质量核对失败记录；本次为用户要求继续生成的采样试稿，不标记为医生审核通过或原始临床报告。",
    }


def aggregate(output: Path, expected: int, protocol_sha256: str) -> dict:
    reports = []
    for path in sorted((output / "cases").glob("case_*.json")):
        row = json.loads(path.read_text())
        if row.get("protocol_sha256") != protocol_sha256:
            raise ValueError(f"生成协议不一致，禁止混合输出：{path}")
        if row.get("status") == "draft_generated" and row.get("text", "").strip():
            reports.append(row)
    ids = {r["scan_id"] for r in reports}
    if len(ids) != len(reports) or ids - set(range(expected)):
        raise ValueError("病例编号存在重复或超出范围")
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer)
    writer.writerow(["patient_id", "image_description"])
    writer.writerows((r["scan_id"], r["text"]) for r in reports)
    temporary = output / "descriptions_draft.csv.tmp"
    temporary.write_text(csv_buffer.getvalue(), encoding="utf-8-sig")
    temporary.replace(output / "descriptions_draft.csv")
    progress = {
        "expected_cases": expected, "completed_cases": len(reports),
        "sampled_frames": sum(r["sampled_frames"] for r in reports),
        "format_flagged_cases": sum(bool(r["quality_flags"]) for r in reports),
        "inference_seconds_total": sum(r["inference"]["seconds"] for r in reports),
        "missing_cases": sorted(set(range(expected)) - ids),
        "synthetic": True, "review_status": "not_reviewed",
        "protocol_sha256": protocol_sha256,
    }
    save_json(output / "progress.json", progress)
    return progress


def worker(args: argparse.Namespace) -> None:
    from generate_cq500_image_descriptions import Describer, block_label_files, quality_flags
    import numpy as np
    import torch
    from tqdm import tqdm

    sys.addaudithook(block_label_files)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    protocol = json.loads((args.output_dir / "generation_protocol.json").read_text())
    expected_digest = build_protocol(args)["protocol_sha256"]
    if protocol["protocol_sha256"] != expected_digest:
        raise ValueError("工作进程的生成配置与输出目录协议不一致")
    # 内存副本正在复制时可先导入依赖，文件长度全部一致后才加载权重。
    original = ROOT / "pre_weights/Qwen3-VL-8B-Instruct"
    if args.model_path.resolve() != original.resolve():
        expected_files = {p.name: p.stat().st_size for p in original.iterdir() if p.is_file()}
        wait_started = time.monotonic()
        print(f"工作进程{args.worker_index}等待模型内存副本完成", flush=True)
        while not all((args.model_path / name).is_file()
                      and (args.model_path / name).stat().st_size == size
                      for name, size in expected_files.items()):
            if time.monotonic() - wait_started > 1800:
                raise TimeoutError("模型内存副本30分钟内未完成")
            time.sleep(5)
    print(f"工作进程{args.worker_index}开始加载本地模型", flush=True)
    describer = Describer(args.model_path, "cuda:0", args.max_new_tokens, "int8")
    print(f"工作进程{args.worker_index}模型加载完成", flush=True)
    assigned = list(range(args.worker_index, args.expected_cases, len(args.devices)))
    for scan_id in tqdm(assigned, desc=f"均匀采样所见/分组{args.worker_index}"):
        destination = args.output_dir / "cases" / f"case_{scan_id:03d}.json"
        manifest = json.loads((args.views_dir / f"case_{scan_id:03d}" / "manifest.json").read_text())
        if destination.exists():
            previous = json.loads(destination.read_text())
            if (previous.get("status") == "draft_generated"
                    and previous.get("protocol_sha256") == protocol["protocol_sha256"]
                    and previous.get("source_fingerprint") == manifest["source_fingerprint"]):
                continue
            raise ValueError(f"已有其他协议或状态的病例，禁止覆盖：{destination}")
        start = time.monotonic()
        try:
            selected = choose_frames(manifest, args.sample_frames)
            if not selected:
                raise ValueError("本例没有可用于描述的图像")
            pictures = []
            for series in manifest["series"]:
                local = [r["frame_index"] for r in selected if r["series_index"] == series["series_index"]]
                if not local:
                    continue
                with np.load(args.views_dir / f"case_{scan_id:03d}" / series["cache"]) as archive:
                    frames = archive["frames"]
                    assert len(frames) == series["frame_count"]
                    pictures.extend(frames[local])
            frames = np.stack(pictures)
            assert len(frames) == len(selected) == min(args.sample_frames, manifest["rendered_frames"])
            prompt = input_prompt(manifest, selected)
            text, audit = describer.generate(prompt, frames)
            if not text:
                raise ValueError("模型返回空文本")
            audit["additional_model_frame_sampling"] = "none"
            audit["frame_sampling"] = POLICY
            save_json(destination, {
                "scan_id": scan_id, "status": "draft_generated", "text": text,
                "raw_text": audit.pop("raw_text", text), "character_count": len(text),
                "synthetic": True, "review_status": "not_reviewed",
                "prior_visual_quality_gate": "failed", "quality_flags": quality_flags(text),
                "ground_truth_read": False, "prediction_files_read": False,
                "protocol_sha256": protocol["protocol_sha256"],
                "source_fingerprint": manifest["source_fingerprint"],
                "source_frames": manifest["rendered_frames"],
                "sampled_frames": len(selected), "sampling_policy": POLICY,
                "selected_frames": selected, "input_prompt": prompt,
                "source_decode_failures": manifest["failed_files"],
                "inference": audit, "wall_seconds": round(time.monotonic() - start, 3),
                "automatic_rewrites": 0,
            })
            print(f"病例{scan_id:03d}已保存：{len(selected)}张，{len(text)}字，推理{audit['seconds']:.1f}秒", flush=True)
            del pictures, frames
        except Exception as exc:
            save_json(args.output_dir / "errors" / f"case_{scan_id:03d}.json", {
                "scan_id": scan_id, "error": str(exc), "traceback": traceback.format_exc(),
                "protocol_sha256": protocol["protocol_sha256"],
            })
            print(f"病例{scan_id:03d}生成失败，已记录：{exc}", flush=True)
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()


def supervise(args: argparse.Namespace) -> None:
    if not args.allow_unreviewed_drafts:
        raise ValueError("本模型存在已知阅片误述，仅可显式启用 --allow-unreviewed-drafts 生成研究试稿")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol(args)
    protocol_path = args.output_dir / "generation_protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        if old["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("现有输出目录采用不同协议，请使用新目录，不能覆盖原文")
    else:
        save_json(protocol_path, protocol)
    state_path = args.output_dir / "run_state.json"
    started = time.time()
    workers, log_files = [], []
    for index, device in enumerate(args.devices):
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": str(device), "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4"})
        command = [
            sys.executable, "-u", str(Path(__file__).resolve()), "--worker-index", str(index),
            "--model-path", str(args.model_path), "--views-dir", str(args.views_dir),
            "--output-dir", str(args.output_dir), "--sample-frames", str(args.sample_frames),
            "--max-new-tokens", str(args.max_new_tokens), "--expected-cases", str(args.expected_cases),
            "--devices", *map(str, args.devices),
        ]
        log = (args.output_dir / f"worker_{index}.log").open("a", encoding="utf-8")
        log_files.append(log)
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        workers.append(process)
    save_json(state_path, {"status": "running", "supervisor_pid": os.getpid(),
                          "worker_pids": [w.pid for w in workers], "started_unix": started,
                          "devices": args.devices, "expected_cases": args.expected_cases})
    try:
        while any(w.poll() is None for w in workers):
            progress = aggregate(args.output_dir, args.expected_cases, protocol["protocol_sha256"])
            print(f"已生成{progress['completed_cases']}/{args.expected_cases}例采样试稿", flush=True)
            time.sleep(20)
        progress = aggregate(args.output_dir, args.expected_cases, protocol["protocol_sha256"])
        codes = [w.returncode for w in workers]
        status = "draft_generation_complete" if progress["completed_cases"] == args.expected_cases and not any(codes) else "incomplete"
        save_json(state_path, {"status": status, "worker_exit_codes": codes,
                              "started_unix": started, "finished_unix": time.time(),
                              "wall_seconds": time.time() - started, **progress})
        print(f"运行结束：{status}；{progress['completed_cases']}/{args.expected_cases}例", flush=True)
        if status == "incomplete":
            raise SystemExit(1)
    finally:
        for log in log_files:
            log.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=ROOT / "pre_weights/Qwen3-VL-8B-Instruct")
    parser.add_argument("--views-dir", type=Path, default=ROOT / "outputs/cq500/image_descriptions/views")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/cq500/image_descriptions/uniform64")
    parser.add_argument("--sample-frames", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=420)
    parser.add_argument("--expected-cases", type=int, default=491)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 3])
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--allow-unreviewed-drafts", action="store_true")
    args = parser.parse_args()
    if args.sample_frames <= 0 or not args.devices or len(set(args.devices)) != len(args.devices):
        raise ValueError("采样数量必须为正，设备列表必须非空且不重复")
    if args.worker_index is not None:
        if not 0 <= args.worker_index < len(args.devices):
            raise ValueError("工作进程编号超出范围")
        worker(args)
    else:
        supervise(args)


if __name__ == "__main__":
    main()
