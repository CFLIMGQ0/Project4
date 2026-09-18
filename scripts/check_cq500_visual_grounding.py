#!/usr/bin/env python3
"""用匿名静态像素检查视觉模型能否正确描述影像，避免扩展不可靠试稿。"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import pydicom

from generate_cq500_image_descriptions import Describer, SYSTEM_PROMPT, save_json
from prepare_cq500_report_views import render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--native-brain-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    describer = Describer(args.model_path, "cuda:0", 360, "int8")
    manifest = json.loads((args.case_dir / "manifest.json").read_text())
    arrays = [np.load(args.case_dir / series["cache"])["frames"] for series in manifest["series"]]
    frames = np.concatenate(arrays)
    side = frames.shape[1]
    probes = []
    if args.native_brain_only:
        pictures = []
        for series in manifest["series"]:
            for source in series["sources"]:
                ds = pydicom.dcmread(source["path"], force=True)
                pixels = ds.pixel_array
                if pixels.ndim == 3:
                    pixels = pixels[source["frame"]]
                pictures.append(Image.fromarray(render(pixels, ds, 448)[:, :448]).convert("RGB"))
        probes.append(("all_native_brain_images", pictures,
                       "这些是同一患者完整的连续头部CT脑窗切片，已按空间位置排序，一张图片对应一张切片。"
                       "每张图片的左侧是患者右侧，右侧是患者左侧。仅根据所有脑窗图像描述主要密度、形态、"
                       "脑室及中线结构情况，约200字。可见高密度和低密度区域均需如实描述；请勿把一侧异常说到另一侧。"
                       "本次只提供脑窗，因此不要评价骨质情况。只写可见所见，不写病名或推测病因。"))
    # 此处仅用于排查输入方式；最终描述任务仍要求全部影像覆盖。
    if len(frames) >= 22 and not args.native_brain_only:
        probes.append(("single_frame", [Image.fromarray(frames[20, :, :side]).convert("RGB")],
                       "请描述这一张头部CT脑窗的可见形态与灰度分布，尤其说明亮区、暗区的位置和形状。只写看到的内容，不推测病因或诊断。"))
    pictures = []
    for window in range(3):
        for start in range(0, len(frames), 12):
            subset = frames[start:start + 12]
            plate = Image.new("L", (side * 4, side * 3))
            for i, frame in enumerate(subset):
                plate.paste(Image.fromarray(frame[:, window * side:(window + 1) * side]),
                            ((i % 4) * side, (i // 4) * side))
            pictures.append(plate.convert("RGB"))
    if not args.native_brain_only:
        probes.append(("all_static_images", pictures,
                   "这些拼图包含同一患者全部连续CT切片，每幅拼图从左到右、从上到下为连续切片。"
                   "先给出全部脑窗拼图，其后为同样切片的硬膜下窗，最后为骨窗。"
                   "每个横断面内图像左侧对应患者右侧。请根据图像写一段约200字的影像所见。"
                   "只写实际可见的位置、灰度、形态、结构关系，不写病名、分类结论或治疗建议。"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, images, prompt in probes:
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
        text = describer.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = describer.processor(text=[text], images=images, do_resize=False, return_tensors="pt").to("cuda:0")
        start = time.monotonic()
        with torch.inference_mode():
            output = describer.model.generate(**inputs, max_new_tokens=360, do_sample=False, repetition_penalty=1.05)
        answer = describer.processor.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        record = {"probe": name, "image_count": len(images), "text": answer, "seconds": time.monotonic() - start,
                  "ground_truth_read": False, "review_status": "not_reviewed"}
        save_json(args.output_dir / f"{name}.json", record)
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
