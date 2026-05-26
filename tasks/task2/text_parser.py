from __future__ import annotations

import math
import re
from typing import Any

from tasks.common import normalize_text


REGION_NAMES = (
    "esophagus",
    "cardia_fundus",
    "gastric_body",
    "antrum_angle",
    "duodenum",
    "other",
)
REGION_INDEX = {name: index for index, name in enumerate(REGION_NAMES)}
REGION_KEYWORDS = {
    "esophagus": ["食管", "食道", "门齿", "齿状线"],
    "cardia_fundus": ["贲门", "胃底"],
    "gastric_body": ["胃体", "大弯", "小弯"],
    "antrum_angle": ["胃窦", "胃角", "幽门"],
    "duodenum": ["十二指肠", "球部", "降部"],
}
POSITIVE_LESION_KEYWORDS = [
    "隆起",
    "糜烂",
    "息肉",
    "萎缩",
    "充血",
    "水肿",
    "溃疡",
    "肿物",
    "狭窄",
    "出血",
    "发红",
    "褪色",
    "增生",
    "肿胀",
    "变薄",
    "蛇行",
    "结节",
    "病变",
    "smt",
    "新生物",
]
NEGATIVE_KEYWORDS = ["无异常", "正常", "光滑", "未见", "无殊", "齿状线清晰", "粘膜光滑"]
CLAUSE_SPLIT_PATTERN = re.compile(r"[。；;，,\n]+")
SPECIMEN_COUNT_PATTERN = re.compile(r"(?:\*|x|×)\s*(\d+)", re.IGNORECASE)


def _detect_region(text_norm: str) -> str:
    for region_name, keywords in REGION_KEYWORDS.items():
        if any(keyword in text_norm for keyword in keywords):
            return region_name
    return "other"


def _detect_polarity(text_norm: str) -> bool | None:
    has_positive = any(keyword in text_norm for keyword in POSITIVE_LESION_KEYWORDS)
    has_negative = any(keyword in text_norm for keyword in NEGATIVE_KEYWORDS)
    if has_positive and not has_negative:
        return True
    if has_negative and not has_positive:
        return False
    if has_positive:
        return True
    if has_negative:
        return False
    return None


def parse_watch_text(watch: str) -> list[dict[str, Any]]:
    watch_norm = normalize_text(watch)
    if not watch_norm:
        return []

    findings: list[dict[str, Any]] = []
    last_region_name = "other"
    for raw_clause in CLAUSE_SPLIT_PATTERN.split(watch_norm):
        clause = raw_clause.strip()
        if not clause:
            continue
        region_name = _detect_region(clause)
        has_polarity_hint = any(keyword in clause for keyword in POSITIVE_LESION_KEYWORDS + NEGATIVE_KEYWORDS)
        if region_name == "other" and has_polarity_hint and last_region_name != "other":
            region_name = last_region_name
        if region_name == "other" and not has_polarity_hint:
            continue
        last_region_name = region_name
        findings.append(
            {
                "region": region_name,
                "region_index": REGION_INDEX[region_name],
                "positive": _detect_polarity(clause),
                "text": clause,
            }
        )
    return findings


def parse_specimen(specimen: str) -> list[dict[str, Any]]:
    specimen_norm = normalize_text(specimen)
    if not specimen_norm:
        return []

    results: list[dict[str, Any]] = []
    for raw_clause in CLAUSE_SPLIT_PATTERN.split(specimen_norm):
        clause = raw_clause.strip()
        if not clause:
            continue
        region_name = _detect_region(clause)
        if region_name == "other":
            continue
        match = SPECIMEN_COUNT_PATTERN.search(clause)
        count = int(match.group(1)) if match else 1
        results.append(
            {
                "region": region_name,
                "region_index": REGION_INDEX[region_name],
                "count": count,
                "text": clause,
            }
        )
    return results


def _allocate_segment_sizes(total: int, weights: list[float]) -> list[int]:
    if total <= 0 or not weights:
        return []
    weight_sum = sum(max(weight, 1e-6) for weight in weights)
    raw_sizes = [total * max(weight, 1e-6) / weight_sum for weight in weights]
    sizes = [max(1, int(math.floor(size))) for size in raw_sizes]
    diff = total - sum(sizes)

    if diff > 0:
        order = sorted(range(len(raw_sizes)), key=lambda idx: raw_sizes[idx] - sizes[idx], reverse=True)
        for offset in range(diff):
            sizes[order[offset % len(order)]] += 1
    elif diff < 0:
        order = sorted(range(len(raw_sizes)), key=lambda idx: raw_sizes[idx] - sizes[idx])
        for idx in order:
            if diff == 0:
                break
            removable = min(sizes[idx] - 1, -diff)
            if removable > 0:
                sizes[idx] -= removable
                diff += removable

    if sum(sizes) != total:
        sizes[-1] += total - sum(sizes)
    return sizes


def generate_pseudo_labels(watch: str, specimen: str, num_images: int) -> dict[str, list[int] | list[float]]:
    if num_images <= 0:
        return {"region_labels": [], "relevance_scores": []}

    findings = parse_watch_text(watch)
    specimen_regions = parse_specimen(specimen)

    region_state: dict[str, dict[str, Any]] = {}
    for finding in findings:
        state = region_state.setdefault(
            str(finding["region"]),
            {"positive": False, "negative": False, "specimen": 0, "order": int(finding["region_index"])},
        )
        if finding["positive"] is True:
            state["positive"] = True
        elif finding["positive"] is False:
            state["negative"] = True
        state["order"] = min(int(state["order"]), int(finding["region_index"]))

    for specimen_item in specimen_regions:
        state = region_state.setdefault(
            str(specimen_item["region"]),
            {
                "positive": False,
                "negative": False,
                "specimen": 0,
                "order": int(specimen_item["region_index"]),
            },
        )
        state["specimen"] += int(specimen_item["count"])
        state["order"] = min(int(state["order"]), int(specimen_item["region_index"]))

    if not region_state:
        return {
            "region_labels": [REGION_INDEX["other"]] * num_images,
            "relevance_scores": [0.3] * num_images,
        }

    ordered_regions = sorted(region_state.items(), key=lambda item: (int(item[1]["order"]), str(item[0])))
    region_names = [item[0] for item in ordered_regions]

    weights: list[float] = []
    for _, state in ordered_regions:
        weight = 1.0
        if bool(state["positive"]):
            weight += 1.0
        if int(state["specimen"]) > 0:
            weight += 0.5
        weights.append(weight)

    segment_sizes = _allocate_segment_sizes(num_images, weights)
    region_labels: list[int] = []
    relevance_scores: list[float] = []

    for region_name, segment_size in zip(region_names, segment_sizes):
        state = region_state[region_name]
        if int(state["specimen"]) > 0:
            relevance = 1.0
        elif bool(state["positive"]):
            relevance = 0.8
        elif bool(state["negative"]):
            relevance = 0.1
        else:
            relevance = 0.3
        region_index = REGION_INDEX.get(region_name, REGION_INDEX["other"])
        region_labels.extend([region_index] * segment_size)
        relevance_scores.extend([relevance] * segment_size)

    if len(region_labels) < num_images:
        pad_num = num_images - len(region_labels)
        region_labels.extend([REGION_INDEX["other"]] * pad_num)
        relevance_scores.extend([0.3] * pad_num)
    elif len(region_labels) > num_images:
        region_labels = region_labels[:num_images]
        relevance_scores = relevance_scores[:num_images]

    return {
        "region_labels": region_labels,
        "relevance_scores": relevance_scores,
    }


__all__ = [
    "REGION_INDEX",
    "REGION_NAMES",
    "generate_pseudo_labels",
    "parse_specimen",
    "parse_watch_text",
]
