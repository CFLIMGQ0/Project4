from __future__ import annotations

import re
import unicodedata
from typing import Any

from tasks.task2.selection import TASK2_LABEL_RULES


# 唯一分类标准：直接引用 TASK2 生成三标签时实际使用的规则，避免两份词典发生漂移。
CATEGORY_STANDARD_TERMS_BY_LABEL: dict[str, tuple[str, ...]] = {
    label_name: tuple(terms) for label_name, terms in TASK2_LABEL_RULES.items()
}

# 只补充标准词的异体字或同义直写形式。这些表达同样会直接暴露答案，但当前标签脚本未逐项列出。
MASK_EXTENSION_TERMS_BY_LABEL: dict[str, tuple[str, ...]] = {
    "label_esophageal_smt": (
        "食管粘膜下隆起",
        "食管黏膜下肿瘤",
        "食管粘膜下肿瘤",
        "食管粘膜下肿物",
        "黏膜下肿瘤",
        "粘膜下肿瘤",
        "黏膜下肿物",
        "粘膜下肿物",
        "smt",
    ),
    "label_esophageal_mucosal_or_tumor": (
        "食管粘膜病变(待病理)",
        "食管粘膜病变(性质待定)",
        "食管隆起型病变",
        "食管早期癌",
        "食管早癌",
        "食管癌",
    ),
    "label_gastritis": (
        "慢性萎缩性胃炎",
        "慢性非萎缩性胃炎",
    ),
}


ANSWER_TERMS_BY_LABEL: dict[str, tuple[str, ...]] = {
    label_name: tuple(
        dict.fromkeys(
            (
                *CATEGORY_STANDARD_TERMS_BY_LABEL[label_name],
                *MASK_EXTENSION_TERMS_BY_LABEL.get(label_name, ()),
            )
        )
    )
    for label_name in CATEGORY_STANDARD_TERMS_BY_LABEL
}

GASTRITIS_GRADE_CODES = ("c1", "c2", "c3", "o1", "o2", "o3")
STANDALONE_LATIN_TERMS = ("smt", "sescc")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).replace("\u3000", " ").lower()
    return re.sub(r"\s+", " ", text).strip()


def _build_answer_pattern() -> re.Pattern[str]:
    lexical_terms = sorted(
        {
            normalize_text(term)
            for terms in ANSWER_TERMS_BY_LABEL.values()
            for term in terms
            if term not in GASTRITIS_GRADE_CODES and term not in STANDALONE_LATIN_TERMS
        },
        key=len,
        reverse=True,
    )
    lexical_pattern = "|".join(re.escape(term) for term in lexical_terms)
    # 报告中存在“0.3cmSMT”这类省略空格的写法；允许 SMT 紧跟测量单位 cm。
    smt_pattern = r"(?:(?<![a-z0-9])|(?<=cm))smt(?![a-z0-9])"
    sescc_pattern = r"(?<![a-z0-9])sescc(?![a-z0-9])"
    latin_pattern = rf"(?:{smt_pattern}|{sescc_pattern})"
    # C1、C-1、C 1、C–1 等均视为同一分级；边界避免匹配 abc1、c10 等无关字符串。
    grade_pattern = r"(?<![a-z0-9])[co]\s*[-–—]?\s*[123](?![a-z0-9])"
    return re.compile(f"(?:{lexical_pattern}|{latin_pattern}|{grade_pattern})", flags=re.IGNORECASE)


ANSWER_PATTERN = _build_answer_pattern()


def mask_answer_terms(text: Any, mask_token: str = "MASKTARGET") -> tuple[str, list[str]]:
    """在分词前用同一占位符遮蔽分类标准词及其直接书写变体。"""

    normalized = normalize_text(text)
    hits = [match.group(0) for match in ANSWER_PATTERN.finditer(normalized)]
    masked = ANSWER_PATTERN.sub(f" {mask_token.lower()} ", normalized)
    return re.sub(r"\s+", " ", masked).strip(), hits


def contains_answer_term(text: Any) -> bool:
    return ANSWER_PATTERN.search(normalize_text(text)) is not None


__all__ = [
    "ANSWER_PATTERN",
    "ANSWER_TERMS_BY_LABEL",
    "CATEGORY_STANDARD_TERMS_BY_LABEL",
    "GASTRITIS_GRADE_CODES",
    "MASK_EXTENSION_TERMS_BY_LABEL",
    "STANDALONE_LATIN_TERMS",
    "contains_answer_term",
    "mask_answer_terms",
    "normalize_text",
]
