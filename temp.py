from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================
# 事业倾向心理测试（终端版）
# ------------------------------------------------------------
# 核心框架：Holland RIASEC 职业兴趣模型
# ------------------------------------------------------------
# 这份程序不是“娱乐向职业测试”，而是参考职业心理学中非常经典的
# Holland RIASEC（霍兰德职业兴趣）框架来设计。
#
# 六个维度：
# R = Realistic      现实型 / 实干型
# I = Investigative  研究型 / 分析型
# A = Artistic       艺术型 / 创意型
# S = Social         社会型 / 助人型
# E = Enterprising   企业型 / 影响型
# C = Conventional   常规型 / 秩序型
#
# 这类模型的重点不是“你能不能成功”，而是：
# 1. 你更喜欢什么样的工作活动
# 2. 你在哪类工作环境里更容易有动力
# 3. 什么样的任务更符合你的天然偏好
#
# 重要说明：
# - 本程序是“研究框架启发版”的终端测评工具，不是官方 O*NET 原版量表。
# - 为了便于终端交互，这里使用 24 题、4 级作答（A/B/C/D）的简化版设计。
# - 24题比20题略多一点，但能保证 6 个维度每个维度至少 4 题，结果更稳定。
# - 结果适合用于自我了解、职业方向思考、岗位偏好梳理，不是招聘筛选工具。
#
# 使用建议：
# - 按“你平时大多数时候的真实偏好”作答，而不是“你觉得正确的答案”。
# - 如果你正在经历强压力、临时失业、刚转行等状态，结果可能会被放大或扭曲。
# ============================================================


@dataclass(frozen=True)
class Question:
    qid: int
    text: str
    dimension: str  # R / I / A / S / E / C


DIMENSION_NAMES: Dict[str, str] = {
    "R": "现实型 / 实干型（Realistic）",
    "I": "研究型 / 分析型（Investigative）",
    "A": "艺术型 / 创意型（Artistic）",
    "S": "社会型 / 助人型（Social）",
    "E": "企业型 / 影响型（Enterprising）",
    "C": "常规型 / 秩序型（Conventional）",
}


DIMENSION_SHORT: Dict[str, str] = {
    "R": "偏好动手、操作、器械、现场、解决具体现实问题",
    "I": "偏好分析、研究、推理、探索原因、处理复杂问题",
    "A": "偏好表达、创意、变化、美感、非标准化产出",
    "S": "偏好帮助、支持、沟通、教学、陪伴与服务",
    "E": "偏好说服、推动、主导、影响资源和结果",
    "C": "偏好规则、秩序、流程、数据、准确和稳定执行",
}


ANSWER_MAP: Dict[str, int] = {
    "A": 1,  # 非常不喜欢
    "B": 2,  # 比较不喜欢
    "C": 3,  # 比较喜欢
    "D": 4,  # 非常喜欢
}


ANSWER_LABELS: Dict[str, str] = {
    "A": "非常不喜欢",
    "B": "比较不喜欢",
    "C": "比较喜欢",
    "D": "非常喜欢",
}


QUESTIONS: List[Question] = [
    # -------------------------------------------------
    # R 现实型 / 实干型（4题）
    # -------------------------------------------------
    Question(1, "自己动手修理设备、工具或硬件，让它重新正常运转。", "R"),
    Question(2, "面对具体问题时，我更愿意直接动手试，而不是先讲很多理论。", "R"),
    Question(3, "我对机械、设备、工程、现场操作、户外执行类工作有兴趣。", "R"),
    Question(4, "把抽象想法落成能真正使用的东西，这件事让我有成就感。", "R"),

    # -------------------------------------------------
    # I 研究型 / 分析型（4题）
    # -------------------------------------------------
    Question(5, "我喜欢分析复杂问题，并追究它背后的原因和规律。", "I"),
    Question(6, "相比马上行动，我有时更愿意先查资料、建逻辑、想清楚。", "I"),
    Question(7, "我会被研究、数据、技术、科学、系统分析类内容吸引。", "I"),
    Question(8, "解决难题时，我会因为‘弄明白了’这件事本身而感到满足。", "I"),

    # -------------------------------------------------
    # A 艺术型 / 创意型（4题）
    # -------------------------------------------------
    Question(9, "我喜欢用文字、视觉、声音、创意或独特表达来呈现想法。", "A"),
    Question(10, "我不太喜欢过于僵硬、标准答案太多的工作方式。", "A"),
    Question(11, "我会被内容创作、设计、品牌、故事、审美、灵感类工作吸引。", "A"),
    Question(12, "如果一份工作能让我自由发挥、提出新点子，我会更有动力。", "A"),

    # -------------------------------------------------
    # S 社会型 / 助人型（4题）
    # -------------------------------------------------
    Question(13, "帮助别人解决问题、成长或变得更好，会让我有价值感。", "S"),
    Question(14, "我通常能注意到别人的感受，并愿意做解释、安抚或支持。", "S"),
    Question(15, "我对教学、咨询、培训、服务、医疗、陪伴类工作更容易有兴趣。", "S"),
    Question(16, "比起只面对机器或系统，我通常更喜欢和人产生实际互动。", "S"),

    # -------------------------------------------------
    # E 企业型 / 影响型（4题）
    # -------------------------------------------------
    Question(17, "我喜欢推动事情发生，而不是只是参与别人已经定好的安排。", "E"),
    Question(18, "我对谈判、说服、带队、争取资源、拿结果这类事情有兴趣。", "E"),
    Question(19, "在团队里，我不太抗拒站到前面发起、组织或拍板。", "E"),
    Question(20, "如果一个目标值得冲，我会愿意承担压力去推动它落地。", "E"),

    # -------------------------------------------------
    # C 常规型 / 秩序型（4题）
    # -------------------------------------------------
    Question(21, "我对整理流程、归档信息、核对细节、维护秩序并不排斥。", "C"),
    Question(22, "当规则明确、步骤清晰时，我通常更容易发挥稳定。", "C"),
    Question(23, "我做事会比较在意准确性、完整性和少出错。", "C"),
    Question(24, "如果一个系统很混乱，我会想把它整理得更清楚、更可执行。", "C"),
]


DIMENSION_PROFILES: Dict[str, Dict[str, str]] = {
    "R": {
        "title": "现实型 / 实干型",
        "core": "你更容易被具体、直接、可操作、能看到结果的工作吸引。",
        "strength": "你的优势往往是动手、执行、现场应对、把事情真正做出来。",
        "risk": "如果工作太空泛、太多概念、长期看不到成果，你容易失去耐心。",
        "fit": "更适合偏工程、制造、设备、运维、实施、现场、产品落地、实操类环境。",
    },
    "I": {
        "title": "研究型 / 分析型",
        "core": "你更容易被难题、原理、逻辑、模型、数据和深度理解吸引。",
        "strength": "你的优势往往是分析、诊断、学习复杂系统、独立思考和找到本质。",
        "risk": "如果工作只要求重复执行、很少思考空间，你容易觉得没有挑战。",
        "fit": "更适合偏研究、算法、数据、咨询、策略、产品分析、技术探索、问题诊断类环境。",
    },
    "A": {
        "title": "艺术型 / 创意型",
        "core": "你更容易被表达、自主、灵感、审美、故事感和创新吸引。",
        "strength": "你的优势往往是提出新点子、表达风格、内容创造、概念转化和差异化。",
        "risk": "如果环境过于僵硬、流程过死、没有表达空间，你容易被压抑。",
        "fit": "更适合偏设计、内容、品牌、策划、创意、媒体、用户体验、创新表达类环境。",
    },
    "S": {
        "title": "社会型 / 助人型",
        "core": "你更容易从帮助、支持、沟通、解释、陪伴和培养他人中获得价值感。",
        "strength": "你的优势往往是共情、沟通、协作、服务意识、教学与关系建立。",
        "risk": "如果工作长期缺少人际意义，只剩下冷冰冰的流程，你可能会失去动力。",
        "fit": "更适合偏教育、培训、客户成功、咨询、人力、医疗、公益、服务与协作类环境。",
    },
    "E": {
        "title": "企业型 / 影响型",
        "core": "你更容易被目标、竞争、影响力、资源整合、主导和结果驱动吸引。",
        "strength": "你的优势往往是推动、组织、说服、拿结果、敢承担外部压力。",
        "risk": "如果环境里没有空间让你发起和推进，你会觉得束手束脚。",
        "fit": "更适合偏业务、销售、管理、运营、创业、商务拓展、市场增长、项目推动类环境。",
    },
    "C": {
        "title": "常规型 / 秩序型",
        "core": "你更容易被清晰规则、稳定流程、准确执行和秩序感吸引。",
        "strength": "你的优势往往是细致、可靠、流程化、规范化和把复杂事情整理清楚。",
        "risk": "如果环境过于混乱、边界极弱、朝令夕改，你可能会感到消耗。",
        "fit": "更适合偏财务、行政、法务支持、数据整理、质量管理、运营流程、项目协调类环境。",
    },
}


COMBO_ARCHETYPES: Dict[str, Dict[str, str]] = {
    "RI": {
        "title": "技术攻坚型",
        "summary": "你偏向把复杂问题研究清楚，再把它真正做出来。",
        "fit": "更容易适配技术研发、工程实现、系统设计、数据产品、硬件/软件结合类方向。",
    },
    "IR": {
        "title": "技术攻坚型",
        "summary": "你偏向把复杂问题研究清楚，再把它真正做出来。",
        "fit": "更容易适配技术研发、工程实现、系统设计、数据产品、硬件/软件结合类方向。",
    },
    "IA": {
        "title": "洞察创意型",
        "summary": "你既重视深度思考，也重视表达和创新，不太喜欢纯重复执行。",
        "fit": "更容易适配产品策略、用户研究、创意策划、内容策略、创新咨询类方向。",
    },
    "AI": {
        "title": "洞察创意型",
        "summary": "你既重视深度思考，也重视表达和创新，不太喜欢纯重复执行。",
        "fit": "更容易适配产品策略、用户研究、创意策划、内容策略、创新咨询类方向。",
    },
    "AS": {
        "title": "表达助人型",
        "summary": "你擅长通过表达、理解和沟通影响别人，工作意义感往往来自人与内容的连接。",
        "fit": "更容易适配教育内容、培训、品牌传播、咨询辅导、社区运营类方向。",
    },
    "SA": {
        "title": "表达助人型",
        "summary": "你擅长通过表达、理解和沟通影响别人，工作意义感往往来自人与内容的连接。",
        "fit": "更容易适配教育内容、培训、品牌传播、咨询辅导、社区运营类方向。",
    },
    "SE": {
        "title": "带人推进型",
        "summary": "你既关注人，也愿意把事情往前推，通常不满足于只做幕后支持。",
        "fit": "更容易适配团队管理、客户管理、业务运营、项目管理、组织协调类方向。",
    },
    "ES": {
        "title": "带人推进型",
        "summary": "你既关注人，也愿意把事情往前推，通常不满足于只做幕后支持。",
        "fit": "更容易适配团队管理、客户管理、业务运营、项目管理、组织协调类方向。",
    },
    "EC": {
        "title": "业务运营型",
        "summary": "你既看结果，也能处理规则、流程和落地，这让你更接近能打仗的执行管理者。",
        "fit": "更容易适配运营管理、商业分析、流程优化、销售运营、项目推进类方向。",
    },
    "CE": {
        "title": "业务运营型",
        "summary": "你既看结果，也能处理规则、流程和落地，这让你更接近能打仗的执行管理者。",
        "fit": "更容易适配运营管理、商业分析、流程优化、销售运营、项目推进类方向。",
    },
    "CR": {
        "title": "稳健执行型",
        "summary": "你既看重秩序和准确，也能接受具体任务和实际执行。",
        "fit": "更容易适配质量、实施、交付、供应链、流程执行、技术支持类方向。",
    },
    "RC": {
        "title": "稳健执行型",
        "summary": "你既看重秩序和准确，也能接受具体任务和实际执行。",
        "fit": "更容易适配质量、实施、交付、供应链、流程执行、技术支持类方向。",
    },
}


def print_line(char: str = "=", width: int = 78) -> None:
    print(char * width)



def print_intro() -> None:
    print_line()
    print("事业倾向心理测试（RIASEC 24题版）")
    print_line()
    print("说明：")
    print("1. 这是一份基于霍兰德 RIASEC 职业兴趣模型设计的事业方向测试。")
    print("2. 它测的是你更偏好的工作活动和工作环境，不是能力高低评判。")
    print("3. 每题请按‘你真实喜不喜欢做这种事’作答，不要按社会期待作答。")
    print("4. 结果更适合作为职业方向和岗位偏好的参考，不是唯一答案。")
    print_line("-")
    print("作答方式：")
    for key, label in ANSWER_LABELS.items():
        print(f"{key}. {label}")
    print_line()



def ask_nickname() -> str:
    name = input("请输入你的昵称（可直接回车跳过）：").strip()
    return name if name else "你"



def ask_question(question: Question) -> str:
    print("\n")
    print_line("-")
    print(f"第 {question.qid:02d} 题 / 共 {len(QUESTIONS)} 题")
    print(question.text)
    print("A. 非常不喜欢")
    print("B. 比较不喜欢")
    print("C. 比较喜欢")
    print("D. 非常喜欢")

    while True:
        ans = input("请输入 A / B / C / D：").strip().upper()
        if ans in ANSWER_MAP:
            return ans
        print("输入无效，请输入 A、B、C 或 D。")



def mean(values: List[int]) -> float:
    return sum(values) / len(values) if values else 0.0



def to_percent(avg_1_to_4: float) -> float:
    # 1 -> 0%, 4 -> 100%
    return ((avg_1_to_4 - 1.0) / 3.0) * 100.0



def score_bar(score: float, total_blocks: int = 24) -> str:
    pct = to_percent(score)
    filled = round((pct / 100.0) * total_blocks)
    return "█" * filled + "░" * (total_blocks - filled)



def level_label(score: float) -> str:
    if score < 1.90:
        return "低"
    if score < 2.85:
        return "中"
    return "高"



def sort_dimensions(scores: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)



def get_top_code(scores: Dict[str, float], top_n: int = 3) -> str:
    ordered = sort_dimensions(scores)
    return "".join(code for code, _ in ordered[:top_n])



def choose_combo_archetype(scores: Dict[str, float]) -> Dict[str, str]:
    ordered = sort_dimensions(scores)
    top_two = ordered[0][0] + ordered[1][0]
    if top_two in COMBO_ARCHETYPES:
        return COMBO_ARCHETYPES[top_two]
    return {
        "title": "综合适配型",
        "summary": "你的职业兴趣并不只集中在一种单一路径上，更像是一个需要看组合配置的人。",
        "fit": "更适合先看前两到前三维的组合，而不是只用一个字母定义自己。",
    }



def confidence_hint(scores: Dict[str, float]) -> str:
    ordered = sort_dimensions(scores)
    gap_top1_top2 = ordered[0][1] - ordered[1][1]
    gap_top2_top3 = ordered[1][1] - ordered[2][1]

    if gap_top1_top2 >= 0.60:
        return "你的第一兴趣维度比较突出，说明你的事业偏好方向相对更清晰。"
    if gap_top1_top2 < 0.25 and gap_top2_top3 < 0.25:
        return "你的前几项分数比较接近，说明你可能适合‘复合型岗位’，而不是极端单一岗位。"
    return "你的结果有主次，但不是特别极端，比较像现实里常见的混合型职业偏好。"



def build_dimension_section(scores: Dict[str, float]) -> str:
    lines: List[str] = ["【六维职业兴趣结果】"]
    for dim, score in sort_dimensions(scores):
        lines.append(f"{dim} - {DIMENSION_NAMES[dim]}：{score:.2f} / 4.00（{level_label(score)}）")
        lines.append(f"{score_bar(score)}  {to_percent(score):.1f} / 100")
        lines.append(f"偏好说明：{DIMENSION_SHORT[dim]}")
        lines.append("")
    return "\n".join(lines).rstrip()



def build_core_profile(scores: Dict[str, float]) -> str:
    ordered = sort_dimensions(scores)
    top1, score1 = ordered[0]
    top2, score2 = ordered[1]
    top3, score3 = ordered[2]
    code = get_top_code(scores, 3)
    combo = choose_combo_archetype(scores)

    return (
        "【你的核心事业画像】\n"
        f"前三兴趣代码：{code}\n"
        f"第一维度：{DIMENSION_NAMES[top1]}\n"
        f"第二维度：{DIMENSION_NAMES[top2]}\n"
        f"第三维度：{DIMENSION_NAMES[top3]}\n\n"
        f"综合类型：{combo['title']}\n"
        f"一句话概括：{combo['summary']}\n"
        f"更适合的方向：{combo['fit']}\n\n"
        f"结果解读提示：{confidence_hint(scores)}"
    )



def build_top_dimension_report(scores: Dict[str, float]) -> str:
    top_dim = sort_dimensions(scores)[0][0]
    profile = DIMENSION_PROFILES[top_dim]
    return (
        "【你的第一职业兴趣维度】\n"
        f"{profile['title']}\n"
        f"核心特征：{profile['core']}\n"
        f"优势：{profile['strength']}\n"
        f"风险：{profile['risk']}\n"
        f"适配环境：{profile['fit']}"
    )



def build_work_style_section(scores: Dict[str, float]) -> str:
    ordered = sort_dimensions(scores)
    top_codes = [code for code, _ in ordered[:2]]
    low_codes = [code for code, _ in ordered[-2:]]

    lines: List[str] = ["【放到现实工作里，你更可能是什么样】"]

    if "R" in top_codes:
        lines.append("- 你更可能偏向‘先做起来’，对空转讨论的耐心可能有限。")
    if "I" in top_codes:
        lines.append("- 你更可能会先想清楚逻辑、原因和结构，再决定如何行动。")
    if "A" in top_codes:
        lines.append("- 你更可能需要表达空间和创造空间，太模板化的工作会削弱你的动力。")
    if "S" in top_codes:
        lines.append("- 你更可能在能帮助别人、解释复杂内容、建立连接的岗位里更有价值感。")
    if "E" in top_codes:
        lines.append("- 你更可能喜欢推进目标、争取资源、把团队或项目往前带。")
    if "C" in top_codes:
        lines.append("- 你更可能在有秩序、规则和明确标准的环境里表现得更稳。")

    if "A" in low_codes:
        lines.append("- 你可能不是特别依赖创意表达来获得工作满足感。")
    if "S" in low_codes:
        lines.append("- 你可能不希望工作长期高度依赖情绪劳动和持续对人投入。")
    if "E" in low_codes:
        lines.append("- 你可能不太喜欢长期高压博弈、强说服或以竞争为主的岗位。")
    if "C" in low_codes:
        lines.append("- 你可能不太适合长期纯流程、纯重复、规则极强而变化极少的岗位。")
    if "R" in low_codes:
        lines.append("- 你可能不太偏好长期重体力、重现场、重工具操作的工作。")
    if "I" in low_codes:
        lines.append("- 你可能不太喜欢长期独自深挖复杂问题、只和抽象信息打交道。")

    return "\n".join(lines)



def build_strength_risk_section(scores: Dict[str, float]) -> str:
    ordered = sort_dimensions(scores)
    top = [code for code, _ in ordered[:3]]
    low = [code for code, _ in ordered[-2:]]

    strengths: List[str] = []
    risks: List[str] = []

    if "R" in top:
        strengths.append("把想法变成实际动作或成果")
    if "I" in top:
        strengths.append("分析复杂问题、做判断、找本质")
    if "A" in top:
        strengths.append("创意表达、差异化思考、提出新方案")
    if "S" in top:
        strengths.append("理解他人、沟通协调、建立合作")
    if "E" in top:
        strengths.append("推动目标、整合资源、向结果负责")
    if "C" in top:
        strengths.append("稳定执行、控细节、搭流程和规则")

    if "R" in low:
        risks.append("可能对纯现场、纯实操、重工具型任务动力不足")
    if "I" in low:
        risks.append("可能在需要长期深度钻研和抽象分析的岗位上耐心不足")
    if "A" in low:
        risks.append("可能对高自由度、缺乏标准的创意环境不一定有持续兴趣")
    if "S" in low:
        risks.append("可能在高密度助人、服务、情绪劳动岗位中消耗较大")
    if "E" in low:
        risks.append("可能不喜欢长期以销售、谈判、竞争和高压推进为主的环境")
    if "C" in low:
        risks.append("可能对过多流程、表格、审批、细碎规则产生厌烦")

    lines = ["【你在事业上的优势与潜在风险】"]
    lines.append("优势倾向：")
    for s in strengths[:5]:
        lines.append(f"- {s}")

    lines.append("")
    lines.append("潜在风险：")
    for r in risks[:5]:
        lines.append(f"- {r}")

    return "\n".join(lines)



def build_environment_section(scores: Dict[str, float]) -> str:
    ordered = sort_dimensions(scores)
    top2 = [code for code, _ in ordered[:2]]
    lines: List[str] = ["【你更适合什么样的工作环境】"]

    if "R" in top2 and "I" in top2:
        lines.append("- 适合：技术问题明确、可以研究也可以动手落地的环境。")
    if "I" in top2 and "A" in top2:
        lines.append("- 适合：有思考空间、允许提出新框架或新表达方式的环境。")
    if "A" in top2 and "S" in top2:
        lines.append("- 适合：既能表达想法，又能对人产生实际影响的环境。")
    if "S" in top2 and "E" in top2:
        lines.append("- 适合：既要沟通协作，又要推进目标和带动团队的环境。")
    if "E" in top2 and "C" in top2:
        lines.append("- 适合：既要拿结果，又离不开流程、节奏和执行管理的环境。")
    if "C" in top2 and "R" in top2:
        lines.append("- 适合：标准清楚、落地明确、可控性高的执行与运营环境。")

    if not any(True for _ in lines[1:]):
        lines.append("- 适合：需要根据你的前三维组合来选，而不是只看单一路径。")

    lines.append("- 更重要的不是岗位名字，而是日常工作内容是否匹配你的兴趣结构。")
    return "\n".join(lines)



def summarize_answers(records: List[Dict[str, object]]) -> Dict[str, List[str]]:
    high_hits: Dict[str, List[str]] = {k: [] for k in DIMENSION_NAMES}
    low_hits: Dict[str, List[str]] = {k: [] for k in DIMENSION_NAMES}

    for item in records:
        dim = str(item["dimension"])
        scored = int(item["scored"])
        text = str(item["text"])
        if scored >= 3:
            high_hits[dim].append(text)
        if scored <= 2:
            low_hits[dim].append(text)

    return {"high": high_hits, "low": low_hits}



def build_pattern_section(records: List[Dict[str, object]], scores: Dict[str, float]) -> str:
    patterns = summarize_answers(records)
    top_dim = sort_dimensions(scores)[0][0]
    low_dim = sort_dimensions(scores)[-1][0]

    lines: List[str] = ["【你的作答模式里最明显的点】"]
    lines.append(f"最能代表你高兴趣方向的是：{DIMENSION_NAMES[top_dim]}")
    for item in patterns["high"][top_dim][:3]:
        lines.append(f"- {item}")

    lines.append("")
    lines.append(f"目前相对没那么吸引你的是：{DIMENSION_NAMES[low_dim]}")
    for item in patterns["low"][low_dim][:3]:
        lines.append(f"- {item}")

    return "\n".join(lines)



def build_growth_suggestions(scores: Dict[str, float]) -> str:
    ordered = sort_dimensions(scores)
    top1 = ordered[0][0]
    low1 = ordered[-1][0]

    suggestions: List[str] = ["【更实用的事业建议】"]

    suggestions.append("- 选方向时，先看你每天要做什么，再看岗位名称好不好听。")
    suggestions.append("- 与其追求‘最热门’，不如优先寻找和你兴趣结构更匹配的任务组合。")

    if top1 == "R":
        suggestions.append("- 你适合多争取可以看见实际成果的任务，不要长期只做抽象讨论。")
    elif top1 == "I":
        suggestions.append("- 你适合多争取分析、诊断、研究和复杂问题拆解型任务。")
    elif top1 == "A":
        suggestions.append("- 你适合多争取表达、策划、创意和有自主空间的任务。")
    elif top1 == "S":
        suggestions.append("- 你适合多争取需要沟通、支持、培训、服务和关系协调的任务。")
    elif top1 == "E":
        suggestions.append("- 你适合多争取推进项目、拿结果、整合资源和带队的机会。")
    elif top1 == "C":
        suggestions.append("- 你适合多争取流程优化、质量控制、运营支撑和规则化工作。")

    if low1 == "R":
        suggestions.append("- 如果工作长期要求你重现场、重工具、重体力，你可能会越来越没劲。")
    elif low1 == "I":
        suggestions.append("- 如果工作长期只剩下深度研究和抽象分析，你可能容易觉得枯燥或太慢。")
    elif low1 == "A":
        suggestions.append("- 如果工作核心是高强度内容创作或审美表达，你未必会长期享受。")
    elif low1 == "S":
        suggestions.append("- 如果工作长期需要高密度服务他人和情绪劳动，你要留意消耗。")
    elif low1 == "E":
        suggestions.append("- 如果岗位核心是持续谈判、销售、竞争和高压推进，你可能不太舒服。")
    elif low1 == "C":
        suggestions.append("- 如果岗位非常重流程、重文档、重规则、重重复细节，你可能会觉得受限。")

    return "\n".join(suggestions)



def build_footer_note() -> str:
    return (
        "【怎么看待这份结果更合理】\n"
        "- RIASEC 测的是职业兴趣倾向，不是智商、能力高低或未来成就。\n"
        "- 你能做好的事情，和你喜欢长期做的事情，不一定完全相同。\n"
        "- 真正更接近现实的用法，是把结果和你的能力、经历、资源、市场机会一起看。\n"
        "- 所以这份报告最适合回答的是：‘什么样的工作内容，我更可能做得久、做得有劲。’"
    )



def build_report(nickname: str, scores: Dict[str, float], records: List[Dict[str, object]]) -> str:
    code = get_top_code(scores, 3)
    parts = [
        "【测试结果总览】\n"
        f"对象：{nickname}\n"
        f"前三职业兴趣代码：{code}\n"
        f"R 现实型：{to_percent(scores['R']):.1f} / 100\n"
        f"I 研究型：{to_percent(scores['I']):.1f} / 100\n"
        f"A 艺术型：{to_percent(scores['A']):.1f} / 100\n"
        f"S 社会型：{to_percent(scores['S']):.1f} / 100\n"
        f"E 企业型：{to_percent(scores['E']):.1f} / 100\n"
        f"C 常规型：{to_percent(scores['C']):.1f} / 100",
        "",
        build_core_profile(scores),
        "",
        build_dimension_section(scores),
        "",
        build_top_dimension_report(scores),
        "",
        build_work_style_section(scores),
        "",
        build_environment_section(scores),
        "",
        build_strength_risk_section(scores),
        "",
        build_pattern_section(records, scores),
        "",
        build_growth_suggestions(scores),
        "",
        build_footer_note(),
    ]
    return "\n".join(parts)



def save_report_to_txt(nickname: str, report: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch for ch in nickname if ch.isalnum() or ch in "_-") or "user"
    path = Path(f"career_riasec_report_{safe_name}_{timestamp}.txt")
    path.write_text(report, encoding="utf-8")
    return path



def save_raw_json(nickname: str, scores: Dict[str, float], records: List[Dict[str, object]]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch for ch in nickname if ch.isalnum() or ch in "_-") or "user"
    path = Path(f"career_riasec_raw_{safe_name}_{timestamp}.json")
    payload = {
        "nickname": nickname,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scores_avg_1_to_4": scores,
        "scores_percent": {k: round(to_percent(v), 2) for k, v in scores.items()},
        "levels": {k: level_label(v) for k, v in scores.items()},
        "top_code": get_top_code(scores, 3),
        "combo_archetype": choose_combo_archetype(scores),
        "answers": records,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path



def ask_yes_no(prompt: str, default_yes: bool = False) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    while True:
        raw = input(f"{prompt} {suffix} ").strip().lower()
        if raw == "":
            return default_yes
        if raw in {"y", "yes", "1", "是"}:
            return True
        if raw in {"n", "no", "0", "否"}:
            return False
        print("请输入 y / n。")



def run_test() -> None:
    print_intro()
    nickname = ask_nickname()

    grouped_scores: Dict[str, List[int]] = {k: [] for k in DIMENSION_NAMES}
    records: List[Dict[str, object]] = []

    for q in QUESTIONS:
        ans = ask_question(q)
        scored = ANSWER_MAP[ans]
        grouped_scores[q.dimension].append(scored)
        records.append(
            {
                "qid": q.qid,
                "text": q.text,
                "dimension": q.dimension,
                "dimension_name": DIMENSION_NAMES[q.dimension],
                "answer": ans,
                "answer_label": ANSWER_LABELS[ans],
                "scored": scored,
            }
        )

    scores: Dict[str, float] = {k: mean(v) for k, v in grouped_scores.items()}
    report = build_report(nickname, scores, records)

    print("\n")
    print_line()
    print(report)
    print_line()

    if ask_yes_no("是否把结果保存为 txt 文本报告？", default_yes=True):
        txt_path = save_report_to_txt(nickname, report)
        print(f"已保存报告：{txt_path.resolve()}")

    if ask_yes_no("是否把原始作答和分数保存为 json 文件？", default_yes=False):
        json_path = save_raw_json(nickname, scores, records)
        print(f"已保存原始数据：{json_path.resolve()}")

    print("\n测试结束。希望这份结果能帮助你更清楚：你更适合在什么样的事业环境里发力。")


if __name__ == "__main__":
    run_test()
