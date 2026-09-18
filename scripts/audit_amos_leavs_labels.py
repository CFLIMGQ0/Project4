#!/usr/bin/env python3
"""审计公开 LEAVS 标签的检查级分布；仅下载小型标签文件，不下载 CT。"""

import argparse
import csv
import hashlib
import itertools
import json
import re
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm


COMMIT = "da29ec6e65b39552bc3f433cdd2a1b67e4756537"
FILES = [
    "amos_test_annotations.csv",
    "parsing_results_llm_test_amos.csv",
    "parsing_results_llm_other_amos.csv",
    "README.md",
    "LICENSE",
    "src/majority_vote_amos.py",
    "src/convert_type_abnormality_to_abnormality.py",
    "src/create_raw_table.py",
    "src/convert_join_organs_suffixes.py",
]
ORGANS = {
    "liver": "肝脏", "gallbladder": "胆囊", "spleen": "脾脏",
    "right kidney": "右肾", "left kidney": "左肾", "pancreas": "胰腺",
    "stomach": "胃", "small bowel": "小肠", "large bowel": "大肠",
}
GROUPS = {
    "liver": ["liver"], "kidney": ["left kidney", "right kidney"],
    "gallbladder": ["gallbladder"], "spleen": ["spleen"],
    "bowel": ["small bowel", "large bowel"], "pancreas": ["pancreas"],
    "stomach": ["stomach"],
}
ZH = {**ORGANS, "kidney": "肾脏（左右合并）", "bowel": "肠道（大小肠合并）"}
TYPES = {
    "focal": ["focal"], "diffuse": ["diffuse"],
    "size": ["enlarged", "atrophy"],
    "postsurgical_absent": ["postsurgical", "absent"],
}
TYPE_ZH = {"focal": "局灶性异常", "diffuse": "弥漫性异常", "size": "增大或萎缩", "postsurgical_absent": "术后改变或器官缺如"}
PRIORITY = {0: 0, -2: 1, -3: 2, -1: 3, 1: 4}
STATE_FIELDS = {1: "positive", 0: "explicit_negative", -1: "possible", -3: "ambiguous", -2: "not_mentioned"}


def scan_id(value):
    matches = set(re.findall(r"amos_\d+", value))
    assert len(matches) == 1, value
    return matches.pop()


def union_state(values):
    """沿用官方异常聚合优先级，保留未提及和不确定状态。"""
    values = list(values)
    assert values and set(values) <= set(PRIORITY), values
    return max(values, key=PRIORITY.__getitem__)


def download_file(name, dest, proxy):
    url = f"https://raw.githubusercontent.com/rsummers11/LEAVS/{COMMIT}/{name}"
    target = dest / name
    if not target.exists():
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        with opener.open(urllib.request.Request(url, headers={"User-Agent": "AMEF-label-audit"}), timeout=45) as response:
            content = response.read()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(content)
    content = target.read_bytes()
    return {"file": name, "url": url, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def auto_labels(rows):
    cases = defaultdict(dict)
    original_splits = {}
    raw_states = defaultdict(Counter)
    for row in tqdm(rows, desc="整理自动标签", leave=False):
        if row["type_annotation"] != "labels":
            continue
        sid, organ = scan_id(row["subjectid_studyid"]), row["organ"]
        assert organ in ORGANS and organ not in cases[sid]
        original_splits[sid] = "official_train" if "imagesTr/" in row["subjectid_studyid"] else "official_validation"
        relevant = set(itertools.chain.from_iterable(TYPES.values())) | {"normal", "anatomy", "device", "quality", "adjacent"}
        for field in relevant:
            raw_states[f"{organ}|{field}"][str(int(row[field]))] += 1
        typed = {kind: union_state(int(row[field]) for field in fields) for kind, fields in TYPES.items()}
        typed["any"] = union_state(typed.values())
        typed["normal"] = int(row["normal"])
        cases[sid][organ] = typed
    assert all(set(organs) == set(ORGANS) for organs in cases.values())
    return dict(cases), original_splits, dict(raw_states)


def human_labels(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["type_annotation"] == "labels":
            grouped[scan_id(row["subjectid_studyid"])].append(row)
    cases = {}
    for sid, raters in grouped.items():
        assert len(raters) == len({r["labeler"] for r in raters}) == 3
        organs = {}
        for organ in ORGANS:
            typed = {}
            votes = {}
            for kind in TYPES:
                source = "enlarged_atrophy" if kind == "size" else kind
                values = [int(float(row[f"{organ}_{source}"])) for row in raters]
                assert set(values) <= {0, 1}
                typed[kind] = int(sum(values) >= 2)
                votes[kind] = {row["labeler"]: value for row, value in zip(raters, values)}
            votes["any"] = {row["labeler"]: int(any(votes[kind][row["labeler"]] for kind in TYPES)) for row in raters}
            typed["any"] = int(sum(votes["any"].values()) >= 2)
            typed["_votes"] = votes
            organs[organ] = typed
        cases[sid] = organs
    # 对照原论文表 1 的人工多数票结果，避免误读标注者或严重程度字段。
    expected = {("liver", "focal"): 91, ("liver", "diffuse"): 31, ("spleen", "size"): 21,
                ("right kidney", "focal"): 62, ("left kidney", "focal"): 57}
    for (organ, kind), n in expected.items():
        assert sum(case[organ][kind] for case in cases.values()) == n, (organ, kind)
    return cases


def group_value(organs, members, kind):
    if "_votes" in organs[members[0]]:
        # 官方 create_raw_table.py 先合并异常类型和器官，再对标注者作多数票。
        raters = set(organs[members[0]]["_votes"][kind])
        assert all(set(organs[o]["_votes"][kind]) == raters for o in members)
        return int(sum(any(organs[o]["_votes"][kind][r] for o in members) for r in raters) >= 2)
    return union_state(organs[organ][kind] for organ in members)


def matrices(cases):
    organ_matrix, typed_matrix, evidence_matrix = {}, {}, {}
    for sid, organs in cases.items():
        organ_matrix[sid], typed_matrix[sid], evidence_matrix[sid] = {}, {}, {}
        for group, members in GROUPS.items():
            value = group_value(organs, members, "any")
            organ_matrix[sid][group] = value
            # 严格证据口径：明确阳性；不确定；明确正常/全部明确阴性；缺少结论。
            if value == 1:
                evidence = "positive"
            elif value in {-1, -3}:
                evidence = "uncertain"
            elif "_votes" in organs[members[0]] or all(organs[o]["any"] == 0 or organs[o].get("normal") == 1 for o in members):
                evidence = "negative_supported"
            else:
                evidence = "not_stated"
            evidence_matrix[sid][group] = evidence
            for kind in TYPES:
                typed_matrix[sid][f"{group}|{kind}"] = group_value(organs, members, kind)
    return organ_matrix, typed_matrix, evidence_matrix


def stats(matrix):
    fields = list(next(iter(matrix.values())))
    output = {}
    for field in fields:
        counts = Counter(row[field] for row in matrix.values())
        output[field] = {name: counts[value] for value, name in STATE_FIELDS.items()}
        output[field]["n"] = len(matrix)
        output[field]["positive_percent"] = counts[1] * 100 / len(matrix)
    return output


def combinations_stats(matrix, fields):
    counts = Counter("".join("1" if row[field] == 1 else "0" for field in fields) for row in matrix.values())
    patterns = {"".join(bits): counts["".join(bits)] for bits in itertools.product("01", repeat=len(fields))}
    histogram = {str(n): sum(count for bits, count in patterns.items() if bits.count("1") == n) for n in range(len(fields) + 1)}
    result = {"labels": list(fields), "n": len(matrix), "patterns_positive_vs_other": patterns,
              "positive_count_histogram": histogram,
              "at_least_two_definite_positive": sum(n for bits, n in counts.items() if bits.count("1") >= 2),
              "all_positive": counts["1" * len(fields)],
              "any_uncertain_case": sum(any(row[f] in {-1, -3} for f in fields) for row in matrix.values()),
              "all_states_explicit_binary": sum(all(row[f] in {0, 1} for f in fields) for row in matrix.values())}
    assert sum(patterns.values()) == len(matrix)
    return result


def save_csv(path, rows):
    rows = list(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=Path("/xmlg/Lim/Project4/datasets/amos_mm_metadata/leavs") / COMMIT)
    parser.add_argument("--output-dir", type=Path, default=Path("/xmlg/Lim/Project4/outputs/amos_mm/leavs_label_audit"))
    parser.add_argument("--proxy", default="http://127.0.0.1:21171")
    args = parser.parse_args()
    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download_file, name, args.metadata_dir, args.proxy) for name in FILES]
        sources = [future.result() for future in tqdm(as_completed(futures), total=len(futures), desc="下载标签与规则文件")]
    human = human_labels(load_csv(args.metadata_dir / "amos_test_annotations.csv"))
    auto_test, test_splits, test_states = auto_labels(load_csv(args.metadata_dir / "parsing_results_llm_test_amos.csv"))
    auto_dev, dev_splits, dev_states = auto_labels(load_csv(args.metadata_dir / "parsing_results_llm_other_amos.csv"))
    assert len(human) == len(auto_test) == 200 and len(auto_dev) == 1487
    assert set(human) == set(auto_test) and not set(auto_dev).intersection(auto_test)
    cohorts = {"auto_all_1687": {**auto_dev, **auto_test}, "auto_development_1487": auto_dev,
               "auto_test_200": auto_test, "human_test_200": human,
               "hybrid_1687": {**auto_dev, **human}}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_commit": COMMIT,
                "unit": "CT检查编号，尚未验证患者唯一性", "sources": sources,
                "original_splits": dict(Counter({**dev_splits, **test_splits}.values())),
                "semantics": {"positive": "仅状态1为明确阳性", "0": "明确提及阴性", "-1": "可能/不确定", "-3": "语言模糊或解剖范围不明确", "-2": "未提及"},
                "organ_any_definition": "大小异常、局灶、弥漫、术后改变或器官缺如的并集；不计图像质量、设备和解剖变异",
                "combination_warning": "组合字符串0代表没有明确阳性，包含不确定和未提及，不等同临床全阴性。",
                "human_vote": "每例3人；先对每位标注者汇总目标异常类型/器官，再至少2票阳性，遵循官方create_raw_table.py的处理顺序。",
                "selection_scope": "候选排序只依据1487例开发集标签；尚未选择或冻结正式训练任务。",
                "raw_auto_state_counts": {key: dict(Counter(dev_states.get(key, {})) + Counter(test_states.get(key, {}))) for key in dev_states},
                "cohorts": {}}
    flat_stats, triple_rows = [], []
    candidate_typed = [
        ("liver|focal", "kidney|focal", "gallbladder|diffuse"),
        ("liver|diffuse", "spleen|size", "gallbladder|diffuse"),
        ("liver|focal", "kidney|focal", "bowel|focal"),
    ]
    all_matrices = {}
    for cohort, cases in tqdm(cohorts.items(), desc="统计检查级分布"):
        organ, typed, evidence = matrices(cases)
        all_matrices[cohort] = (organ, typed, evidence)
        organ_stats, typed_stats = stats(organ), stats(typed)
        for field, values in organ_stats.items():
            flat_stats.append({"cohort": cohort, "level": "organ", "label": field, **values})
        for field, values in typed_stats.items():
            flat_stats.append({"cohort": cohort, "level": "organ_finding", "label": field, **values})
        triples = [combinations_stats(organ, fields) for fields in itertools.combinations(GROUPS, 3)]
        for result in triples:
            triple_rows.append({"cohort": cohort, "labels": "|".join(result["labels"]), "n": result["n"],
                                "at_least_two_positive": result["at_least_two_definite_positive"], "all_three_positive": result["all_positive"],
                                **result["patterns_positive_vs_other"]})
        selected_fields = ["liver", "kidney", "gallbladder"]
        complete = {
            sid: {field: int(evidence[sid][field] == "positive") for field in selected_fields}
            for sid in organ
            if all(evidence[sid][field] in {"positive", "negative_supported"} for field in selected_fields)
        }
        complete_stats = combinations_stats(complete, selected_fields)
        complete_stats["marginal_positive"] = {field: sum(row[field] for row in complete.values()) for field in selected_fields}
        complete_stats["excluded_due_to_uncertain_or_not_stated"] = len(organ) - len(complete)
        manifest["cohorts"][cohort] = {
            "n": len(cases), "organ_stats": organ_stats, "organ_finding_stats": typed_stats,
            "seven_organ_distribution": combinations_stats(organ, list(GROUPS)),
            "organ_triples": triples,
            "typed_candidates": [combinations_stats(typed, fields) for fields in candidate_typed],
            "evidence_states": {field: dict(Counter(row[field] for row in evidence.values())) for field in GROUPS},
            "pairs": [{"labels": list(fields), "both_positive": sum(all(row[f] == 1 for f in fields) for row in organ.values())} for fields in itertools.combinations(GROUPS, 2)],
            "complete_evidence_trio_proposal": complete_stats,
        }
        save_csv(args.output_dir / f"{cohort}_organ_states.csv", ({"scan_id": sid, **row} for sid, row in sorted(organ.items())))
        save_csv(args.output_dir / f"{cohort}_finding_states.csv", ({"scan_id": sid, **row} for sid, row in sorted(typed.items())))
        save_csv(args.output_dir / f"{cohort}_evidence_states.csv", ({"scan_id": sid, **row} for sid, row in sorted(evidence.items())))
        if cohort == "hybrid_1687":
            save_csv(args.output_dir / "完整三标签候选.csv", ({"scan_id": sid,
                     "label_source": "human_majority_test" if sid in human else "automatic_development",
                     **row} for sid, row in sorted(complete.items())))
    # 所有差异均在同一200例上比较；人工标签不混入全量自动标签统计。
    human_matrix, _, _ = all_matrices["human_test_200"]
    auto_matrix, _, _ = all_matrices["auto_test_200"]
    assert sum(row[field] for row in human_matrix.values() for field in GROUPS if field != "stomach") == 381
    agreement = {}
    for field in GROUPS:
        tp = sum(human_matrix[sid][field] == 1 and auto_matrix[sid][field] == 1 for sid in human_matrix)
        fp = sum(human_matrix[sid][field] == 0 and auto_matrix[sid][field] == 1 for sid in human_matrix)
        fn = sum(human_matrix[sid][field] == 1 and auto_matrix[sid][field] != 1 for sid in human_matrix)
        agreement[field] = {"tp": tp, "fp": fp, "fn": fn, "tn": 200 - tp - fp - fn,
                            "strict_positive_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}
    manifest["human_auto_agreement_200"] = agreement
    save_csv(args.output_dir / "label_distribution.csv", flat_stats)
    save_csv(args.output_dir / "organ_triple_distribution.csv", triple_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    lines = ["# AMOS-MM / LEAVS 标签分布审计", "", "## 统计口径", "",
             "- 统计单位：1,687个唯一CT检查编号；尚未验证是否为1,687名独立患者。",
             "- 自动标签覆盖1,687例；其中200例另有每例3人的人工标注；遵循官方顺序，先汇总目标异常类型/器官，再对标注者取多数票。",
             "- 明确阳性仅计状态1；-1、-3保留为不确定，-2保留为未提及，不直接当作临床阴性。",
             "- 器官异常含大小异常、局灶、弥漫、术后改变或器官缺如；左右肾合并，大小肠合并。",
             "- 组合中的0仅表示未标为明确阳性。质量、设备、解剖变异不纳入器官异常定义。",
             "- 这里只审计标签，没有下载CT、核对报告匹配或核查病灶跨切片范围。", "",
             "## 器官分布", "", "| 器官 | 全量自动标签阳性/1687 | 开发集自动阳性/1487 | 人工测试阳性/200 |", "|---|---:|---:|---:|"]
    for field in GROUPS:
        values = [manifest["cohorts"][cohort]["organ_stats"][field]["positive"] for cohort in ("auto_all_1687", "auto_development_1487", "human_test_200")]
        lines.append(f"| {ZH[field]} | {values[0]} | {values[1]} | {values[2]} |")
    lines += ["", "## 肝脏＋肾脏＋胆囊异常的共存分布", "",
              "| 口径 | 检查数 | 零项明确阳性 | 一项明确阳性 | 恰好两项明确阳性 | 三项明确阳性 | 至少两项明确阳性 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for cohort, title in [("auto_all_1687", "全量自动标签"), ("auto_development_1487", "开发集自动标签"),
                          ("human_test_200", "人工多数票测试标签")]:
        triple = next(row for row in manifest["cohorts"][cohort]["organ_triples"] if row["labels"] == ["liver", "kidney", "gallbladder"])
        hist = triple["positive_count_histogram"]
        lines.append(f"| {title} | {triple['n']} | {hist['0']} | {hist['1']} | {hist['2']} | {hist['3']} | {triple['at_least_two_definite_positive']} |")
    complete = manifest["cohorts"]["hybrid_1687"]["complete_evidence_trio_proposal"]
    lines += ["", "## 无需直接把未提及当阴性的完整三标签候选（本次提出的筛选口径）", "",
              "- 自动标签：任一相关异常明确阳性则为1；无不确定项且器官被明确标为正常，或所有相关异常明确阴性，则为0；其余暂缺标签。",
              "- 双肾：任一肾阳性即阳性；两侧均有阴性依据才归阴性。200例人工多数票另作为完整测试参考。",
              f"- 满足三项标签都有依据的候选共{complete['n']}例，包括{manifest['cohorts']['auto_development_1487']['complete_evidence_trio_proposal']['n']}例自动标签开发检查和200例人工测试检查。",
              f"- 这部分零至三项阳性的分布为：{complete['positive_count_histogram']}；至少两项阳性{complete['at_least_two_definite_positive']}例。",
              "- 这是本次标签审计生成的候选子集，不是官方发布的新数据集；正式规模仍需CT/报告匹配和患者去重复核。",
              "- 对未纳入的开发病例，也可以复核报告补齐标签或采用未知标签损失掩码，不必永久排除。"]
    lines += ["", "## 按开发集共阳性数排序的器官三标签组合（探索性）", "", "| 三标签 | 开发集至少两项明确阳性/1487 | 开发集三项明确阳性 |", "|---|---:|---:|"]
    ranked = sorted(manifest["cohorts"]["auto_development_1487"]["organ_triples"], key=lambda row: row["at_least_two_definite_positive"], reverse=True)
    for result in ranked[:10]:
        lines.append(f"| {'＋'.join(ZH[f] for f in result['labels'])} | {result['at_least_two_definite_positive']} | {result['all_positive']} |")
    lines += ["", "## 细分异常分布（全量自动标签）", "", "| 器官 | 局灶 | 弥漫 | 增大/萎缩 | 术后/缺如 |", "|---|---:|---:|---:|---:|"]
    for field in GROUPS:
        values = [manifest["cohorts"]["auto_all_1687"]["organ_finding_stats"][f"{field}|{kind}"]["positive"] for kind in TYPES]
        lines.append(f"| {ZH[field]} | " + " | ".join(map(str, values)) + " |")
    lines += ["", "## 任务准备建议", "",
              "1. 先按开发集的阳性覆盖、跨器官共存及医学意义确定标签，不依据模型测试表现筛选。",
              "2. 比较宽泛器官异常与具体器官-异常类型两种方案；这些是异常标签，不能直接改称特定疾病诊断。",
              "3. 200例人工报告标注宜保留为独立测试，1,487例用于开发；统一各模型划分，核查患者重复。",
              "4. 最终监督标签须明确未提及/不确定的处理，并核查自动标签；统计中的非阳性不能无说明全部转换成阴性。",
              "5. 保留单阳性、共阳性和有依据的阴性病例；影像采样不依赖目标标签。",
              "6. 原报告作为输入时，预先固定目标答案表述的处理规则；病灶连续切片范围须另外核查影像。", "",
              "## 可复核资料", "", f"- 标签来源提交：{COMMIT}",
              "- 来源：https://github.com/rsummers11/LEAVS", "- summary.json记录来源URL、字节数、SHA256和五个统计队列。",
              "- label_distribution.csv包含每个标签的五种状态。",
              "- organ_triple_distribution.csv包含全部35个三器官组合与八种阳性模式。", ""]
    (args.output_dir / "报告.md").write_text("\n".join(lines), encoding="utf-8")
    print("统计完成：", args.output_dir)
    print("器官分布：", json.dumps(manifest["cohorts"]["auto_all_1687"]["organ_stats"], ensure_ascii=False))
    print("开发集候选：", json.dumps(ranked[:5], ensure_ascii=False))


if __name__ == "__main__":
    main()
