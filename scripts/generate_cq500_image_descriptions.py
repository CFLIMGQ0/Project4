#!/usr/bin/env python3
"""依据全部可解码图像生成逐例中文所见；输入完全隔离分类标签与预测结果。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from transformers.video_utils import VideoMetadata
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
USER_PROMPT = """你是一名经验丰富的放射科医生，擅长头颅 CT 序列阅片。请综合观察同一患者提供的全部 CT 图像，以专业、客观的医学语言，撰写一段约 200 字的“影像所见”。
描述应涵盖脑实质、脑室系统、中线结构及颅骨等可见情况；发现异常时，说明其位置、密度、形态、范围、在连续切片中的分布，以及对周围结构的影响。将各切片的信息整合为检查级描述，避免逐张罗列。
所有内容必须依据实际可见的影像信息，不参考预设分类标签，不虚构病史、症状、测量数值或影像细节。对显示不清或无法确认的情况保留不确定性，内容长度以真实观察为准。仅输出一段“影像所见”，不提供诊断结论、治疗建议或额外解释。"""
SYSTEM_PROMPT = USER_PROMPT + """
这是仅依据图像生成研究用所见草稿的任务。输出使用简体中文，目标约180至240个字符。
只描述灰度、位置、轮廓、对称性、结构关系和可见范围。病名、病因推测、分类结论、阳性/阴性判断、建议及治疗均不写。
不得将部分层面的表现推及未展示层面；骨窗显示不足时保留观察限制。
无可靠尺寸依据时仅用形态和相对范围描述，不写毫米、厘米或具体CT值。
常见生理性结构的高密度不能自动写成异常；明确异常不能套用“完全正常”模板。
图像由同一检查的顺序切片构成，三窗为同一切片的三种显示方式，不是三名患者。"""
FORBIDDEN = re.compile(
    r"\b(?:IPH|Mass\s*Effect|Midline\s*Shift|CQ500)\b|"
    r"出血|血肿|占位效应|中线移位|脑疝|梗死|骨折|肿瘤|"
    r"阳性|阴性|诊断为|考虑为|提示为|建议|治疗",
    re.I,
)


def block_label_files(event: str, arguments: tuple) -> None:
    """生成进程禁止打开真实标签、既有划分以及分类模型预测文件。"""
    if event != "open" or not arguments or not isinstance(arguments[0], (str, bytes)):
        return
    path = str(arguments[0]).lower()
    if any(word in path for word in ("reads.csv", "prediction_probabilities", "oof_predictions", "patient_folds.json")):
        raise PermissionError("标签盲化保护：禁止生成进程访问标签或预测文件")


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def clean(text: str) -> str:
    text = re.sub(r"^\s*(?:影像所见|所见)\s*[:：]?\s*", "", text.strip())
    return re.sub(r"\s+", "", text).replace("**", "")


def quality_flags(text: str) -> list[str]:
    flags = []
    if len(text) < 150:
        flags.append("short_text")
    if len(text) > 280:
        flags.append("long_text")
    if FORBIDDEN.search(text):
        flags.append("diagnostic_or_label_wording")
    if re.search(r"\d+(?:\.\d+)?\s*(?:mm|cm|HU|毫米|厘米)", text, re.I):
        flags.append("unsupported_numeric_measurement")
    if not text.endswith(("。", "！", "？")):
        flags.append("possibly_truncated")
    if len(re.findall(r"[\u4e00-\u9fff]", text)) < max(1, len(text)) * 0.8:
        flags.append("non_chinese_or_formatting")
    return flags


class Describer:
    def __init__(self, model_path: Path, device: str, max_new_tokens: int, quantization: str) -> None:
        logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        extra = {}
        if quantization == "int8":
            extra["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_skip_modules=["visual", "lm_head"],
            )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map={"": device},
            attn_implementation="sdpa", local_files_only=True,
            **extra,
        ).eval()
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

    @torch.inference_mode()
    def generate(self, prompt: str, frames: np.ndarray | None = None) -> tuple[str, dict]:
        content = [{"type": "text", "text": prompt}]
        if frames is not None:
            content.insert(0, {"type": "video"})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if frames is not None:
            # 每张切片都输入；禁用默认视频抽帧和额外缩放，保留224像素的独立三窗。
            count = len(frames)
            rgb = np.repeat(frames[:, None, :, :], 3, axis=1)
            metadata = VideoMetadata(
                total_num_frames=count, fps=1.0,
                width=frames.shape[2], height=frames.shape[1],
                duration=float(count), frames_indices=list(range(count)),
            )
            inputs = self.processor(
                text=[text], videos=[rgb], video_metadata=[metadata],
                do_sample_frames=False, do_resize=False,
                return_tensors="pt", padding=True,
            )
            grid = inputs["video_grid_thw"][0].tolist()
            assert grid[0] == (count + 1) // 2, (grid, count)
            assert grid[1] * 16 == frames.shape[1] and grid[2] * 16 == frames.shape[2]
        else:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            grid = None
        inputs = inputs.to(self.device)
        tokens = inputs["input_ids"].shape[-1]
        start = time.monotonic()
        output = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens,
            do_sample=False, repetition_penalty=1.05, use_cache=True,
        )
        new_tokens = output[0, tokens:]
        result = self.processor.decode(new_tokens, skip_special_tokens=True)
        audit = {
            "input_tokens": tokens, "output_tokens": len(new_tokens),
            "seconds": round(time.monotonic() - start, 3),
            "video_grid_thw": grid, "frame_sampling": "none",
            "raw_text": result,
        }
        del inputs, output, new_tokens
        return clean(result), audit


def describe_case(describer: Describer, case_dir: Path, result_dir: Path, chunk_size: int) -> dict:
    manifest = json.loads((case_dir / "manifest.json").read_text())
    scan_id = manifest["scan_id"]
    output = result_dir / f"case_{scan_id:03d}.json"
    if output.exists():
        previous = json.loads(output.read_text())
        if previous.get("status") == "generated" and previous.get("source_fingerprint") == manifest["source_fingerprint"]:
            return previous
    chunks = []
    chunk_dir = result_dir / "chunk_notes" / f"case_{scan_id:03d}"
    for series in manifest["series"]:
        frames = np.load(case_dir / series["cache"])["frames"]
        assert len(frames) == series["frame_count"]
        for start in range(0, len(frames), chunk_size):
            end = min(start + chunk_size, len(frames))
            note_path = chunk_dir / f"series_{series['series_index']:03d}_{start:05d}_{end:05d}.json"
            if note_path.exists():
                record = json.loads(note_path.read_text())
            else:
                orientation = (
                    "横断位，每个窗内图像左侧对应患者右侧，图像右侧对应患者左侧。"
                    if series["axial"] and series["standard_left_right"]
                    else "方向信息不足以统一指定左右；不确定时只描述可见结构及图像侧别。"
                )
                prompt = (
                    "以下是同一患者同一CT序列的一组连续切片，用视频容器依次呈现。视频秒数只代表切片顺序，"
                    "不代表病情随时间变化。每帧左、中、右三个窗依次为脑窗、硬膜下窗、骨窗。"
                    + orientation +
                    f"这一组共{end-start}张，是本序列第{start+1}至{end}张。"
                    "请观察全部切片，记录这一组实际可见的主要影像所见，约180至240字。"
                    "描述密度、形态、脑室与周围结构关系；仅对可见层面作判断，不臆测病史，不写病名或分类结论。"
                    "若仅显示颅底或头顶部，应说明本组观察范围，不评价未显示的脑室等结构。"
                )
                note, audit = describer.generate(prompt, frames[start:end])
                record = {
                    "series_index": series["series_index"], "start": start, "end": end,
                    "observed_frames": end - start, "text": note, "inference": audit,
                }
                save_json(note_path, record)
            chunks.append(record)
            print(f"病例{scan_id:03d}，序列{series['series_index']}，已观察{end}/{len(frames)}帧", flush=True)
    observed_frames = sum(x["observed_frames"] for x in chunks)
    assert observed_frames == manifest["rendered_frames"]
    if not chunks:
        result = {
            "scan_id": scan_id, "status": "no_decodable_images", "text": "",
            "review_status": "needs_review", "source_fingerprint": manifest["source_fingerprint"],
        }
        save_json(output, result)
        return result
    notes = "\n".join(
        f"片段{i+1}（序列{x['series_index']}，切片{x['start']+1}—{x['end']}）：{x['text']}"
        for i, x in enumerate(chunks)
    )
    limitation = ""
    if manifest["failed_files"]:
        limitation += "部分源图像无法解码，文字中需要简短交代局部观察受限。"
    if observed_frames < 10:
        limitation += "源数据仅提供少量层面，需明确其范围有限，不应写全脑及整个颅骨正常；可短于200字。"
    prompt = (
        "请将同一患者各序列、各连续片段的影像观察记录整合成唯一一段检查级影像所见。"
        "不同序列可能重复呈现相同部位；描述去重，不将重复发现解释为新增异常。"
        "保留明确、具体的可见发现；矛盾或未显示内容保留不确定性。"
        "不得新增记录中不存在的结构、左右侧、尺寸或病史。"
        "仅输出一段约180至240字中文所见，不输出标题、条目、病名、诊断或阳性/阴性分类词。"
        + limitation + "\n影像观察记录：\n" + notes
    )
    final_text, audit = describer.generate(prompt)
    flags = quality_flags(final_text)
    revisions = []
    for _ in range(2):
        repairable = [x for x in flags if x != "short_text" or observed_frames >= 10]
        if not repairable:
            break
        edit_prompt = (
            prompt + "\n待规范的草稿：" + final_text +
            "\n请仅依据上面的观察记录规范草稿：用可见位置、密度和结构关系替代病名或分类词；"
            "不添加测量值；目标约200字、一段完整中文，句末用句号。"
        )
        final_text, revision_audit = describer.generate(edit_prompt)
        revisions.append(revision_audit)
        flags = quality_flags(final_text)
    result = {
        "scan_id": scan_id, "status": "generated", "text": final_text,
        "character_count": len(final_text), "synthetic": True,
        "review_status": "not_reviewed", "quality_flags": flags,
        "ground_truth_read": False, "source_fingerprint": manifest["source_fingerprint"],
        "source_files": manifest["source_files"], "duplicate_files": manifest["duplicate_files"],
        "failed_files": len(manifest["failed_files"]), "observed_frames": observed_frames,
        "observed_series": len(manifest["series"]), "chunks": len(chunks),
        "summary_inference": audit, "format_revisions": revisions,
    }
    save_json(output, result)
    print(f"病例{scan_id:03d}完成：{len(final_text)}字；格式标记{flags}\n{final_text}", flush=True)
    return result


def export_results(result_dir: Path) -> None:
    results = [json.loads(p.read_text()) for p in sorted(result_dir.glob("case_*.json"))]
    with (result_dir / "descriptions.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "scan_id", "description", "character_count", "observed_frames", "observed_series",
            "synthetic", "review_status", "quality_flags",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "scan_id": r["scan_id"], "description": r["text"],
                "character_count": r.get("character_count", 0),
                "observed_frames": r.get("observed_frames", 0),
                "observed_series": r.get("observed_series", 0), "synthetic": True,
                "review_status": r["review_status"], "quality_flags": ";".join(r.get("quality_flags", [])),
            })
    save_json(result_dir / "progress.json", {
        "completed_cases": len(results),
        "observed_frames": sum(r.get("observed_frames", 0) for r in results),
        "flagged_cases": sum(bool(r.get("quality_flags")) for r in results),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=ROOT / "pre_weights/Qwen3-VL-8B-Instruct")
    parser.add_argument("--views-dir", type=Path, default=ROOT / "outputs/cq500/image_descriptions/views")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/cq500/image_descriptions/generated")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case-ids", type=int, nargs="+")
    parser.add_argument("--expected-cases", type=int, default=491)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=420)
    parser.add_argument("--quantization", choices=["bf16", "int8"], default="bf16")
    args = parser.parse_args()
    sys.addaudithook(block_label_files)
    gate_path = args.views_dir.parent / "quality_gate.json"
    if args.case_ids is None and gate_path.exists():
        gate = json.loads(gate_path.read_text())
        if gate.get("status") != "passed":
            raise RuntimeError("当前生成方案未通过视觉核对，已阻止全量生成；请先验证新的识图方案")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    save_json(args.output_dir / "generation_protocol.json", {
        "model_path": str(args.model_path), "user_prompt": USER_PROMPT,
        "system_prompt": SYSTEM_PROMPT,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "chunk_size": args.chunk_size, "max_new_tokens": args.max_new_tokens,
        "quantization": args.quantization,
        "frame_sampling": "none", "input_modalities": ["dicom_pixels"],
        "ground_truth_read": False, "prediction_files_read": False,
        "synthetic_reports": True, "human_review": "pending",
        "notes": "研究用图像派生文本，不是数据集原始临床报告。所有可解码的唯一SOP图像都进入模型；重复副本和解码失败留存清单。",
    })
    assert 0 <= args.shard_index < args.num_shards
    selected = args.case_ids if args.case_ids is not None else list(range(args.expected_cases))
    selected = [i for i in selected if i % args.num_shards == args.shard_index]
    describer = Describer(args.model_path, args.device, args.max_new_tokens, args.quantization)
    for scan_id in tqdm(selected, desc="生成检查级图像描述"):
        case_dir = args.views_dir / f"case_{scan_id:03d}"
        wait_started = time.monotonic()
        while not (case_dir / "manifest.json").exists():
            if time.monotonic() - wait_started > 3600:
                raise RuntimeError(f"病例{scan_id}的图像输入未准备好，请检查转换进度")
            time.sleep(5)
        describe_case(describer, case_dir, args.output_dir, args.chunk_size)
        export_results(args.output_dir)
    print(f"本次{len(selected)}例处理完成；输出：{args.output_dir / 'descriptions.csv'}", flush=True)


if __name__ == "__main__":
    main()
