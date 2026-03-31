#!/usr/bin/env python3
from __future__ import annotations

"""胃镜 reportTitle 四类预分层的带 * 小类边界剔除分析（纯标准库实现）。"""

import argparse
import csv
import math
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None

GROUP_MAP = {
    "常规白光胃镜": ["无痛胃镜检查报告", "胃镜检查报告", "一诊疗无痛胃镜报告", "职工体检胃镜(无痛)报告"],
    "手术胃镜": ["胃镜手术(住院)报告", "胃镜下切除手术报告", "胃镜下其他手术报告", "急诊胃镜下取异物报告", "胃镜下静脉曲张手术报告", "急诊胃镜报告"],
    "染色胃镜": ["放大染色胃镜精查报告", "无痛胃镜(含色素内镜)报告", "国际部无痛胃镜检查（含色素内镜）报告", "国际部胃镜检查（含色素内镜）报告"],
    "超声胃镜": ["超声胃镜检查报告", "无痛超声胃镜报告", "超声胃镜下手术报告"],
}
STAR_TITLES = ["胃镜下静脉曲张手术报告", "急诊胃镜报告", "超声胃镜下手术报告"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="胃镜 reportTitle 带*小类边界剔除分析")
    p.add_argument("--config", type=Path, default=Path("configs/path.yaml"))
    p.add_argument("--valid-dicts-report-csv", type=Path, default=None)
    p.add_argument("--centroid-cosine-csv", type=Path, default=None)
    p.add_argument("--fid-matrix-csv", type=Path, default=None)
    p.add_argument("--mmd-matrix-csv", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--similarity-dir",
        type=Path,
        default=None,
        help="相似度矩阵目录（目录下应包含 centroid_cosine_similarity.csv / fid_matrix.csv / mmd_matrix.csv）",
    )
    p.add_argument("--w-cos", type=float, default=1.0)
    p.add_argument("--w-fid", type=float, default=1.0)
    p.add_argument("--w-mmd", type=float, default=1.0)
    return p.parse_args()


def load_yaml_path(cfg: Path) -> dict[str, Any]:
    if yaml is None or (not cfg.is_file()):
        return {}
    with cfg.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("paths", {}) if isinstance(data.get("paths", {}), dict) else {}


def resolve(base: Path, p: str | Path) -> Path:
    x = Path(p).expanduser()
    return x if x.is_absolute() else (base / x).resolve()


def pick_inputs(args: argparse.Namespace) -> dict[str, Path]:
    cfg = args.config.expanduser().resolve()
    cfg_root = cfg.parent.parent
    paths = load_yaml_path(cfg)

    output_root = args.output_dir.expanduser().resolve() if args.output_dir else resolve(cfg_root, paths.get("output_dir", "./outputs"))
    # 默认优先指向 check_similarity/gastric（用户当前任务只分析胃镜）
    sim_dir = (
        args.similarity_dir.expanduser().resolve()
        if args.similarity_dir
        else (output_root / str(paths.get("check_similarity_dir_name", "check_similarity")) / "gastric")
    )

    return {
        "report": args.valid_dicts_report_csv.expanduser().resolve() if args.valid_dicts_report_csv else resolve(cfg_root, paths.get("valid_dicts_report_csv", "valid_dicts_report.csv")),
        "cos": args.centroid_cosine_csv.expanduser().resolve() if args.centroid_cosine_csv else sim_dir / "centroid_cosine_similarity.csv",
        "fid": args.fid_matrix_csv.expanduser().resolve() if args.fid_matrix_csv else sim_dir / "fid_matrix.csv",
        "mmd": args.mmd_matrix_csv.expanduser().resolve() if args.mmd_matrix_csv else sim_dir / "mmd_matrix.csv",
        "out_dir": output_root / "delete_reportTitle",
    }


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def finite(vals: list[float]) -> list[float]:
    return [v for v in vals if isinstance(v, float) and (not math.isnan(v)) and math.isfinite(v)]


def mean(vals: list[float]) -> float:
    v = finite(vals)
    return sum(v) / len(v) if v else float("nan")


def load_report_counts(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "reportTitle" not in (reader.fieldnames or []):
            raise KeyError("valid_dicts_report.csv 缺少 reportTitle 字段")
        counts: dict[str, int] = {}
        rows = list(reader)
        it = tqdm(rows, desc="过滤胃镜记录", unit="行") if tqdm else rows
        for r in it:
            t = (r.get("reportTitle") or "").strip()
            if ("胃" in t) and ("肠" not in t):
                counts[t] = counts.get(t, 0) + 1
        return counts


def load_square_matrix(path: Path, name: str) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"{name} 为空")
    header = [c.strip() for c in rows[0]]
    col_labels = header[1:]

    raw: dict[str, dict[str, list[float]]] = {}
    data_rows = rows[1:]
    it = tqdm(data_rows, desc=f"读取{name}", unit="行") if tqdm else data_rows
    for row in it:
        if not row:
            continue
        rlab = (row[0] or "").strip()
        if not rlab:
            continue
        vals = row[1:]
        for c, v in zip(col_labels, vals):
            clab = c.strip()
            if not clab:
                continue
            raw.setdefault(rlab, {}).setdefault(clab, []).append(to_float(v))

    labels = sorted(set(raw.keys()) & {k for d in raw.values() for k in d.keys()})
    if not labels:
        raise ValueError(f"{name} 行列标签无法对齐")

    mat: dict[str, dict[str, float]] = {a: {} for a in labels}
    for a in labels:
        for b in labels:
            v1 = mean(raw.get(a, {}).get(b, []))
            v2 = mean(raw.get(b, {}).get(a, []))
            if math.isnan(v1) and math.isnan(v2):
                v = float("nan")
            elif math.isnan(v1):
                v = v2
            elif math.isnan(v2):
                v = v1
            else:
                v = (v1 + v2) / 2.0
            mat[a][b] = v
    return mat


def matrix_minmax_similarity(mat: dict[str, dict[str, float]], higher_better: bool) -> dict[str, dict[str, float]]:
    vals: list[float] = []
    for a in mat:
        for b in mat[a]:
            v = mat[a][b]
            if not math.isnan(v) and math.isfinite(v):
                vals.append(v)
    if not vals:
        return {a: {b: float("nan") for b in mat[a]} for a in mat}

    lo, hi = min(vals), max(vals)
    out: dict[str, dict[str, float]] = {a: {} for a in mat}
    for a in mat:
        for b in mat[a]:
            v = mat[a][b]
            if math.isnan(v) or (not math.isfinite(v)):
                out[a][b] = float("nan")
            elif math.isclose(hi, lo):
                out[a][b] = 1.0
            elif higher_better:
                out[a][b] = (v - lo) / (hi - lo)
            else:
                out[a][b] = (hi - v) / (hi - lo)
    return out


def consensus(cos: dict[str, dict[str, float]], fid: dict[str, dict[str, float]], mmd: dict[str, dict[str, float]], w: tuple[float, float, float]) -> dict[str, dict[str, float]]:
    labels = sorted(set(cos.keys()) & set(fid.keys()) & set(mmd.keys()))
    out: dict[str, dict[str, float]] = {a: {} for a in labels}
    for a in labels:
        for b in labels:
            vals = [cos[a].get(b, float("nan")), fid[a].get(b, float("nan")), mmd[a].get(b, float("nan"))]
            ww = [w[0], w[1], w[2]]
            num, den = 0.0, 0.0
            for vv, wi in zip(vals, ww):
                if (not math.isnan(vv)) and math.isfinite(vv) and wi > 0:
                    num += vv * wi
                    den += wi
            out[a][b] = (num / den) if den > 0 else float("nan")
    return out


def affinity(mat: dict[str, dict[str, float]], src: str, targets: list[str]) -> float:
    vals = [mat.get(src, {}).get(t, float("nan")) for t in targets if t != src]
    return mean(vals)


def pair_mean(mat: dict[str, dict[str, float]], labels: list[str]) -> float:
    vals: list[float] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            vals.append(mat.get(labels[i], {}).get(labels[j], float("nan")))
    return mean(vals)


def fmt(x: float) -> str:
    return "NA" if (math.isnan(x) or (not math.isfinite(x))) else f"{x:.4f}"


def recommend(title: str, margin: float, sil: float, gcos: float, gfid: float, gmmd: float, best_alt_group: str) -> tuple[str, str]:
    score = 0
    if not math.isnan(margin):
        score += 2 if margin < -0.02 else 1 if margin < 0.02 else 0
    if not math.isnan(sil):
        score += 2 if sil < 0 else 1 if sil < 0.1 else 0
    score += 1 if (not math.isnan(gcos) and gcos > 0.01) else 0
    score += 1 if (not math.isnan(gfid) and gfid > 0.01) else 0
    score += 1 if (not math.isnan(gmmd) and gmmd > 0.01) else 0

    rec = "建议删除" if score >= 5 else "可删可留" if score >= 3 else "不建议删除"
    if title == "超声胃镜下手术报告":
        if rec == "建议删除" and score < 6:
            rec = "可删可留"
        if best_alt_group == "手术胃镜" and (not math.isnan(margin)) and margin > -0.01:
            rec = "不建议删除"

    text = (
        f"margin={fmt(margin)}（>0 更像所属大类），silhouette={fmt(sil)}（越高越像类内点）；"
        f"删除后纯度变化：cosine {fmt(gcos)}（越大越好），FID {fmt(gfid)}（>0 表示均值下降），MMD {fmt(gmmd)}（>0 表示均值下降）。"
    )
    if title == "超声胃镜下手术报告":
        text += "该类具备“超声+手术”双属性，需考虑真实 hybrid subtype 的研究价值。"
    return rec, text


def main() -> None:
    args = parse_args()
    if min(args.w_cos, args.w_fid, args.w_mmd) < 0 or math.isclose(args.w_cos + args.w_fid + args.w_mmd, 0.0):
        raise ValueError("权重需非负且至少一个大于0")

    ip = pick_inputs(args)
    out_dir = ip["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in ["report", "cos", "fid", "mmd"]:
        if not ip[k].is_file():
            raise FileNotFoundError(f"输入文件不存在：{ip[k]}")

    counts = load_report_counts(ip["report"])
    cos_raw = load_square_matrix(ip["cos"], "centroid_cosine_similarity")
    fid_raw = load_square_matrix(ip["fid"], "fid_matrix")
    mmd_raw = load_square_matrix(ip["mmd"], "mmd_matrix")

    labels = sorted(set(cos_raw) & set(fid_raw) & set(mmd_raw) & set(counts.keys()))
    if not labels:
        raise ValueError("三矩阵与胃镜 reportTitle 无交集")

    # 截断到交集
    cos_raw = {a: {b: cos_raw[a][b] for b in labels} for a in labels}
    fid_raw = {a: {b: fid_raw[a][b] for b in labels} for a in labels}
    mmd_raw = {a: {b: mmd_raw[a][b] for b in labels} for a in labels}

    cos_sim = matrix_minmax_similarity(cos_raw, True)
    fid_sim = matrix_minmax_similarity(fid_raw, False)
    mmd_sim = matrix_minmax_similarity(mmd_raw, False)
    cons = consensus(cos_sim, fid_sim, mmd_sim, (args.w_cos, args.w_fid, args.w_mmd))

    t2g = {t: g for g, ts in GROUP_MAP.items() for t in ts}
    stars = [t for t in STAR_TITLES if t in labels]
    if not stars:
        raise ValueError("三个带*小类未出现在矩阵交集中")

    rows: list[dict[str, Any]] = []
    it = tqdm(stars, desc="分析带*小类", unit="类") if tqdm else stars
    for t in it:
        g = t2g.get(t, "未知")
        own_set = [x for x in GROUP_MAP.get(g, []) if x in labels and x != t]
        own = affinity(cons, t, own_set)

        best_g = "NA"
        best_aff = float("nan")
        for gg, titles in GROUP_MAP.items():
            if gg == g:
                continue
            alt_set = [x for x in titles if x in labels]
            aa = affinity(cons, t, alt_set)
            if math.isnan(best_aff) or ((not math.isnan(aa)) and aa > best_aff):
                best_aff, best_g = aa, gg

        margin = own - best_aff if (not math.isnan(own) and not math.isnan(best_aff)) else float("nan")
        denom = max(abs(own), abs(best_aff), 1e-12) if (not math.isnan(own) and not math.isnan(best_aff)) else float("nan")
        sil = ((own - best_aff) / denom) if (not math.isnan(denom)) else float("nan")

        group_titles = [x for x in GROUP_MAP.get(g, []) if x in labels]
        wo_titles = [x for x in group_titles if x != t]

        cos_b, cos_a = pair_mean(cos_raw, group_titles), pair_mean(cos_raw, wo_titles)
        fid_b, fid_a = pair_mean(fid_raw, group_titles), pair_mean(fid_raw, wo_titles)
        mmd_b, mmd_a = pair_mean(mmd_raw, group_titles), pair_mean(mmd_raw, wo_titles)

        gcos = cos_a - cos_b if (not math.isnan(cos_a) and not math.isnan(cos_b)) else float("nan")
        gfid = fid_b - fid_a if (not math.isnan(fid_a) and not math.isnan(fid_b)) else float("nan")
        gmmd = mmd_b - mmd_a if (not math.isnan(mmd_a) and not math.isnan(mmd_b)) else float("nan")

        rec, exp = recommend(t, margin, sil, gcos, gfid, gmmd, best_g)
        rows.append({
            "reportTitle": t,
            "assigned_group": g,
            "sample_count": counts.get(t, 0),
            "own_affinity": own,
            "best_alt_group": best_g,
            "best_alt_affinity": best_aff,
            "margin": margin,
            "silhouette_like_score": sil,
            "purity_gain_cosine": gcos,
            "purity_gain_fid": gfid,
            "purity_gain_mmd": gmmd,
            "final_recommendation": rec,
            "explanation_cn": exp,
        })

    out_csv = out_dir / "star_report_title_analysis.csv"
    fields = [
        "reportTitle", "assigned_group", "sample_count", "own_affinity", "best_alt_group", "best_alt_affinity", "margin",
        "silhouette_like_score", "purity_gain_cosine", "purity_gain_fid", "purity_gain_mmd", "final_recommendation", "explanation_cn",
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    out_md = out_dir / "star_report_title_analysis.md"
    robustness = [
        "矩阵首列作为行标签并执行 strip 清洗。",
        "行列标签不一致时，仅保留交集标签。",
        "重复标签通过均值聚合，缺失值在融合时按可用视角重分配权重。",
        "矩阵做对称化处理 A=(A+A^T)/2。",
    ]
    md_lines = [
        "# 胃镜 reportTitle 四类预分层的边界样本剔除分析", "", "## 方法说明",
        "- 仅保留 reportTitle 含“胃”且不含“肠”的记录。",
        "- 三视角统一方向后归一化并加权融合为共识相似度。", "",
        "## 指标定义",
        "- own_affinity：与所属大类其余小类的平均共识相似度。",
        "- best_alt_affinity：与其他大类中最相近大类的平均共识相似度。",
        "- margin=own-best_alt。",
        "- silhouette_like_score=(own-alt)/max(|own|,|alt|)。",
        "- purity_gain_cosine=删除后类内平均 cosine 提升量。",
        "- purity_gain_fid=删除后类内平均 FID 下降量（>0 改善）。",
        "- purity_gain_mmd=删除后类内平均 MMD 下降量（>0 改善）。", "",
        "## 鲁棒处理策略",
    ] + [f"- {x}" for x in robustness] + ["", "## 三个带 * 小类逐项分析"]

    for r in rows:
        md_lines.extend([
            f"### {r['reportTitle']}",
            f"- 所属大类：{r['assigned_group']}；样本数 n={r['sample_count']}",
            f"- own_affinity={fmt(r['own_affinity'])}；best_alt_group={r['best_alt_group']}；best_alt_affinity={fmt(r['best_alt_affinity'])}",
            f"- margin={fmt(r['margin'])}；silhouette_like_score={fmt(r['silhouette_like_score'])}",
            f"- 纯度增益：cosine={fmt(r['purity_gain_cosine'])}，FID={fmt(r['purity_gain_fid'])}，MMD={fmt(r['purity_gain_mmd'])}",
            f"- 建议：**{r['final_recommendation']}**",
            f"- 解释：{r['explanation_cn']}", "",
        ])

    md_lines.extend(["## 最终结论", *[f"- {r['reportTitle']}：**{r['final_recommendation']}**" for r in rows], "",
                     "## hybrid subtype 讨论",
                     "“超声胃镜下手术报告”天然具备“超声+手术”双属性，若其与超声主类的亲和度并未明显劣化，建议保留为 hybrid subtype，供后续多模态研究使用。"])
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print("\n=== 指标意义 ===")
    print("own_affinity：目标小类与所属大类的贴合度，越大越好。")
    print("best_alt_affinity：目标小类与最近异类大类贴合度。")
    print("margin：两者差值；>0 表示更像本类，<0 表示边界风险。")
    print("silhouette_like_score：接近1更像类内点，接近-1更像边界/错分点。")
    print("purity_gain_cosine/FID/MMD：删除该类后所属大类纯度提升幅度（>0改善）。")

    print("\n=== 分析结果（终端摘要）===")
    for r in rows:
        print(f"- {r['reportTitle']} | 建议={r['final_recommendation']} | n={r['sample_count']} | margin={fmt(r['margin'])} | silhouette={fmt(r['silhouette_like_score'])} | Δcos={fmt(r['purity_gain_cosine'])} Δfid={fmt(r['purity_gain_fid'])} Δmmd={fmt(r['purity_gain_mmd'])}")

    print("\n输出文件：")
    print(out_csv)
    print(out_md)


if __name__ == "__main__":
    main()
