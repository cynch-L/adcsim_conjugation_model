"""Alarm & diagnostic layer — translating the model's intermediate quantities into "where the problem is and which knob to turn".

Design principles (three hard constraints; lose any one and this module is useless)
----------------------------------------------------------------------------
1. Every alarm must **attribute to a specific process step**. An alarm without attribution is useless.
2. Every alarm must **point to an actionable step** (adjust molar equivalents / adjust pH / extend time / change the site /
   add feed / inspect the code). Alarms that point to uncontrollable factors are not written.
3. Every alarm must carry an **evidence chain**: report each intermediate quantity's value + reference range,
   and give a three-state verdict for each candidate cause (supported / excluded / undetermined).
   ★ 'Excluded' is this module's most valuable output — it tells you which directions are not worth trying.

Confidence tiers
----------------------------------------------------------------------------
Tier A: physical necessity. A violation is an error requiring no calibration. Currently only two checks qualify:
      · mass conservation (opened bonds cannot exceed reductant molar equivalents)
      · the mathematical relation between conjugation probability and odd-numbered DAR
Tier B: model-robust. Depends on the model, but conclusions are stable across reasonable parameter ranges.
Tier C: industry-experience initial value. Thresholds must be calibrated against your own batch HIC data before they count as specification limits.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# Tier-C initial values. After calibration, replace wholesale with your own process specification limits.
DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "odd_dar_pct_warn": 5.0,
    "odd_dar_pct_crit": 10.0,
    "mean_dar_tolerance": 0.5,
    "high_dar_ge6_pct": 25.0,
    "low_dar_le1_pct": 10.0,
    "dar_std": 2.0,
    "p_conj_warn": 0.98,
    "p_conj_crit": 0.95,
    "bond_yield_ratio_warn": 0.85,
    "bond_yield_ratio_crit": 0.70,
    "min_bond_open_ratio": 0.30,
    "min_teff_s": 300.0,
    "teff_spread_max": 10.0,
    "min_thiolate_pct": 1.0,
    "ph_window": [6.5, 8.5],
    "kinetic_completion_min": 3.0,
    # Was "payload_feed_min": 2.0 — never read by anything, while the real
    # threshold (0.7) was hardcoded inline. Renamed to what it actually is:
    # the minimum payload *margin* (charge equivalents − thiols opened).
    "payload_margin_min": 0.7,
}

OK, WARN, CRIT, INFO = "ok", "warn", "crit", "info"
# INFO means "something to say but threshold not triggered"; not counted in
# triggered, only appears in the detailed list.
# Downstream modules (site_env / seqqc / payload_qc) must reuse the field names
# from _mk() when building their alarm dicts: step / item / value / unit /
# level / reference / meaning / action / confidence —— otherwise fields are
# silently dropped when multi-layer reports are concatenated.
SUPPORT, EXCLUDE, UNKNOWN = "supported", "excluded", "undetermined"


def _level(value, warn=None, crit=None, higher_is_worse=True):
    """Return (level, displayed threshold).

    warn / crit are both 'trigger thresholds', with crit more severe than warn:
      higher is worse (higher_is_worse=True):  warn < crit, e.g. odd% warn=5 crit=10
      lower is worse (higher_is_worse=False):  warn > crit, e.g. conjugation probability warn=0.98 crit=0.95
    """
    if warn is None:
        return OK, None
    if higher_is_worse:
        if crit is not None and value > crit:
            return CRIT, crit
        return (WARN, warn) if value > warn else (OK, warn)
    else:
        if crit is not None and value < crit:
            return CRIT, crit
        return (WARN, warn) if value < warn else (OK, warn)


# ----------------------------------------------------------------------------
# single check item
# ----------------------------------------------------------------------------
def _mk(step, name, value, unit, level, ref, meaning, action, conf="C",
        causes=None):
    d = dict(step=step, item=name, value=value, unit=unit, level=level,
             reference=ref, meaning=meaning, action=action, confidence=conf)
    if causes:
        d["causes"] = causes
    return d


# Public entry point: other modules (site_env / seqqc) build their alarms here
# to keep field names consistent. Pass the canonical link name for the step
# argument — no numbering, no metaphor.
make_alarm = _mk


def check_all(state: Dict[str, Any], thresholds: Optional[Dict[str, Any]] = None
              ) -> List[Dict[str, Any]]:
    """Run all checks. state is documented in INPUT_CONTRACT at the end of the module."""
    T = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))
    out: List[Dict[str, Any]] = []

    dist = state.get("dist") or []
    n = len(dist)
    pct = lambda arr: sum(arr) * 100.0 if arr else 0.0       # noqa: E731
    odd = pct(dist[1::2]) if n > 1 else 0.0
    high = pct(dist[6:]) if n > 6 else 0.0
    low = pct(dist[0:2]) if n > 1 else 0.0
    mean = state.get("mean_dar")
    if mean is None and n:
        mean = sum(i * p for i, p in enumerate(dist))
    std = state.get("dar_std")
    if std is None and n and mean is not None:
        var = sum((i - mean) ** 2 * p for i, p in enumerate(dist))
        std = math.sqrt(max(var, 0.0))

    eq = state.get("tcep_eq")
    opened = state.get("bonds_opened")
    per_bond = state.get("per_bond_open") or []
    p_conj = state.get("p_conj")
    payload_feed = state.get("payload_feed_ratio")

    # ---- Tier A: mass conservation ------------------------------------------------
    if eq is not None and opened is not None:
        violated = opened > eq + 1e-6
        out.append(_mk(
            "disulfide reduction", "Mass conservation (opened bonds ≤ reductant molar equivalents)",
            round(opened, 3), f"bonds (cap {eq})",
            CRIT if violated else OK, f"≤ {eq}",
            "Opened bonds exceed the reductant molar amount = TCEP-disulfide 1:1 conservation is violated",
            "**This is a model bug, not a process problem** — check the code; do not tune the process",
            conf="A"))

    # ---- disulfide reduction: reductant utilization -------------------------------------------
    if eq and opened is not None:
        ratio = opened / eq
        lv, ref = _level(ratio, T["bond_yield_ratio_warn"],
                         T["bond_yield_ratio_crit"], higher_is_worse=False)
        causes = None
        if lv != OK:
            causes = [
                dict(name="TCEP degraded / oxidized by air", metric="—", value="—",
                     ref="—", verdict=UNKNOWN),
                dict(name="TCEP hitting intrachain disulfides", metric="intrachain bond accessibility",
                     value=state.get("intrachain_sasa"), ref="far below interchain bonds",
                     verdict=UNKNOWN),
                dict(name="insufficient reduction time", metric="reduction completion",
                     value=state.get("reduction_completion"),
                     ref="≥ 90%", verdict=UNKNOWN),
            ]
        out.append(_mk(
            "disulfide reduction", "Reductant utilization (opened bonds ÷ equivalents)", round(ratio, 3), "—", lv,
            f"≥ {ref}", "Reductant wasted: not going to interchain bonds",
            "Check TCEP freshness / whether intrachain bonds consumed it / raise equivalents or extend time",
            causes=causes))

    # ---- disulfide reduction: hardest-to-open bond -----------------------------------------
    if per_bond:
        mn = min(per_bond)
        lv, ref = _level(mn, T["min_bond_open_ratio"], higher_is_worse=False)
        out.append(_mk(
            "disulfide reduction", "Opening ratio of the hardest-to-open bond", round(mn, 3), "—", lv, f"≥ {ref}",
            "This site barely opens, which lowers the achievable DAR ceiling and widens the distribution",
            "Wrong site chosen, or equivalents too low", ))

    # ---- site accessibility -------------------------------------------------
    teffs = state.get("t_eff_s") or []
    if teffs:
        mn_t = min(teffs)
        lv, ref = _level(mn_t, T["min_teff_s"], higher_is_worse=False)
        out.append(_mk(
            "site accessibility", "Effective exposure time of the most buried site", round(mn_t, 1), "s", lv, f"≥ {ref}",
            "This site is essentially buried inside the protein",
            "This site is unsuitable as a conjugation site; or pick a mutant with a more open conformation"))
        if max(teffs) > 0:
            spread = max(teffs) / max(mn_t, 1e-9)
            lv, ref = _level(spread, T["teff_spread_max"])
            out.append(_mk(
                "site accessibility", "Site exposure time, max / min", round(spread, 2), "fold", lv,
                f"≤ {ref}", "Site heterogeneity is extreme, so the product will inevitably be inhomogeneous",
                "Unrelated to process parameters; it is a property of the antibody itself / site selection"))

    # ---- thiol activation ---------------------------------------------------
    ft = state.get("thiolate_pct")
    if ft is not None:
        lv, ref = _level(ft, T["min_thiolate_pct"], higher_is_worse=False)
        out.append(_mk(
            "thiol activation", "Thiolate fraction", round(ft, 3), "%", lv, f"≥ {ref}",
            "Too few activated thiols; the reaction cannot get going", "Raise pH (or check whether pKa is abnormally high)"))
    ph = state.get("ph")
    if ph is not None:
        lo, hi = T["ph_window"]
        bad = not (lo <= ph <= hi)
        out.append(_mk(
            "thiol activation", "pH within process window", ph, "—", CRIT if bad else OK,
            f"{lo}–{hi}", "Too low and the reaction stalls; too high and payload hydrolysis and side reactions accelerate",
            "Bring pH back into the window"))

    # ---- covalent conjugation ---------------------------------------------------
    if p_conj is not None:
        lv, ref = _level(p_conj, T["p_conj_warn"], T["p_conj_crit"],
                         higher_is_worse=False)
        causes = None
        if lv != OK:
            causes = [
                dict(name="kinetics not finished", metric="rate × conc. × exposure time",
                     value=state.get("kinetic_completion"), ref="≥ 3",
                     verdict=UNKNOWN),
                dict(name="payload hydrolyzed inactive", metric="hydrolysis half-life",
                     value=state.get("hydrolysis_t_half_h"),
                     ref="≫ conjugation time", verdict=UNKNOWN),
                dict(name="maleimide steric hindrance", metric="site pocket size",
                     value=state.get("pocket_size_A"), ref="≥ maleimide ~5 Å",
                     verdict=UNKNOWN),
            ]
        out.append(_mk(
            "covalent conjugation", "Conjugation probability", round(p_conj, 4), "—", lv, f"≥ {ref}",
            "Mathematically guarantees a large fraction of odd-numbered DAR (bond opened but only one payload attached)",
            "Check payload freshness / pH / hydrolysis / site sterics",
            conf="A" if lv != OK else "B", causes=causes))

    kc = state.get("kinetic_completion")
    if kc is not None:
        lv, ref = _level(kc, T["kinetic_completion_min"], higher_is_worse=False)
        out.append(_mk(
            "covalent conjugation", "Kinetic completion (rate × concentration × exposure time)", round(kc, 2), "—", lv,
            f"≥ {ref}", "Reaction did not finish; conversion <95%", "Increase feed ratio / extend time"))

    if payload_feed is not None and opened is not None:
        thiols = 2.0 * opened
        margin = payload_feed - thiols
        lv = CRIT if margin < 0 else (WARN if margin < T["payload_margin_min"] else OK)
        out.append(_mk(
            "covalent conjugation", "payload margin (charge equivalents − thiols opened)", round(margin, 2),
            f"equiv (need {thiols:.2f})", lv, f"≥ {T['payload_margin_min']}",
            "payload is fed sub-stoichiometrically; when the margin is too thin it becomes the new limiting reagent",
            "Raise the payload charge in proportion to equivalents; this is a feed-ratio design issue, not a conjugation-probability issue"))

    # ---- combinatorial statistics: distribution shape -----------------------------------------------
    if n:
        lv, ref = _level(odd, T["odd_dar_pct_warn"], T["odd_dar_pct_crit"])
        causes = None
        if lv != OK:
            causes = [
                dict(name="conjugation step did not saturate", metric="conjugation probability", value=p_conj,
                     ref="≥ 0.98",
                     verdict=SUPPORT if (p_conj or 1) < 0.98 else EXCLUDE),
                dict(name="re-oxidized after reduction", metric="—", value="—", ref="—",
                     verdict=UNKNOWN),
                dict(name="disulfide rearrangement / mismatch", metric="—", value="—", ref="—",
                     verdict=UNKNOWN),
                dict(name="reduction step went wrong", metric="opened bonds ÷ equivalents",
                     value=(opened / eq) if (eq and opened is not None) else None,
                     ref="≥ 0.85", verdict=EXCLUDE),
            ]
        out.append(_mk(
            "combinatorial statistics", "Odd-numbered DAR fraction", round(odd, 2), "%", lv, f"≤ {ref}",
            "A bond opened but only one payload attached — can only be the conjugation step, unrelated to reduction",
            "This is the fastest attribution: first check payload freshness and feed",
            conf="A", causes=causes))

        lv, ref = _level(high, T["high_dar_ge6_pct"])
        causes = None
        if lv != OK:
            causes = [
                dict(name="too many reduction equivalents fed", metric="opened bonds ÷ equivalents",
                     value=(opened / eq) if (eq and opened is not None) else None,
                     ref="≤ 0.85", verdict=UNKNOWN),
                dict(name="conjugation too complete", metric="odd-numbered DAR fraction", value=round(odd, 2),
                     ref="≥ 1%", verdict=EXCLUDE if odd > 1 else UNKNOWN),
                dict(name="site accessibility out of control", metric="site exposure-time ratio",
                     value=round(max(teffs) / max(min(teffs), 1e-9), 2) if teffs else None,
                     ref="≤ 10×", verdict=UNKNOWN),
            ]
        out.append(_mk(
            "combinatorial statistics", "High DAR (≥6) fraction", round(high, 2), "%", lv, f"≤ {ref}",
            "High DAR correlates with aggregation propensity and toxicity risk",
            "Directly lower the equivalents — this is the only effective knob; no other adjustment needed", causes=causes))

        lv, ref = _level(low, T["low_dar_le1_pct"])
        causes = None
        if lv != OK:
            causes = [
                dict(name="reduction never started", metric="opened bonds", value=opened,
                     ref=f"≥ {0.85 * eq:.2f}" if eq else "—",
                     verdict=SUPPORT if (eq and opened and opened < 0.85 * eq)
                     else UNKNOWN),
                dict(name="conjugation step failed", metric="conjugation probability", value=p_conj,
                     ref="≥ 0.98",
                     verdict=EXCLUDE if (p_conj or 0) >= 0.98 else SUPPORT),
                dict(name="site itself is buried", metric="most-buried site exposure time",
                     value=min(teffs) if teffs else None, ref="≥ 300 s",
                     verdict=EXCLUDE if (teffs and min(teffs) >= 300) else UNKNOWN),
                dict(name="wrong pH", metric="thiolate fraction", value=ft, ref="≥ 1%",
                     verdict=EXCLUDE if (ft is not None and ft >= 1) else UNKNOWN),
            ]
        out.append(_mk(
            "combinatorial statistics", "Low DAR (≤1) fraction", round(low, 2), "%", lv, f"≤ {ref}",
            "Too much naked antibody, which competes with the drug for the target",
            "Read the diagnostic table: low opened bonds → check reduction; enough opened bonds → check conjugation", causes=causes))

        if std is not None:
            lv, ref = _level(std, T["dar_std"])
            out.append(_mk(
                "combinatorial statistics", "DAR distribution standard deviation", round(std, 3), "—", lv, f"≤ {ref}",
                "Product is too heterogeneous", "Unrelated to process parameters; caused by excessive site-to-site variation"))

    target = state.get("target_dar")
    if target is not None and mean is not None:
        dev = abs(mean - target)
        lv = CRIT if dev > 2 * T["mean_dar_tolerance"] else (
            WARN if dev > T["mean_dar_tolerance"] else OK)
        out.append(_mk(
            "combinatorial statistics", "Mean DAR deviation from target", round(mean - target, 3),
            f"(target {target})", lv, f"±{T['mean_dar_tolerance']}",
            "Overall deviation from specification", "Trace down: first look at opened bonds, then decide whether to check reduction or conjugation"))

    return out


def summarize_alarms(alarms: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_crit = sum(1 for a in alarms if a["level"] == CRIT)
    n_warn = sum(1 for a in alarms if a["level"] == WARN)
    return dict(n_crit=n_crit, n_warn=n_warn,
                worst=CRIT if n_crit else (WARN if n_warn else OK),
                # INFO does not count as triggered —— otherwise a single "note" would flag the whole batch as problematic
                triggered=[a for a in alarms if a["level"] in (WARN, CRIT)])


# ============================================================================
# Reverse triage: from observed symptoms back to candidate causes
# ----------------------------------------------------------------------------
# check_all() above runs FORWARD: the model computes every intermediate and
# then reports which step looks wrong. That is the right tool when you have a
# model run in front of you.
#
# The bench question is usually the other way round. A batch came out at DAR 2
# instead of DAR 4, you have three or four readings, and you want to know which
# step failed. This block answers that: each cause leaves a distinct pattern of
# symptoms, so an observed pattern is matched against those patterns.
#
# It deliberately does not name a single cause. It lists the causes still
# consistent with what was measured, drops the ones the measurements
# contradict, and then names the one extra reading that would separate whatever
# is left. The "excluded" list is the most useful part: it says which
# directions are not worth spending a batch on.
# ============================================================================

# Each lamp is one reading you can actually take at the bench. `normal` is the
# value you would see on a batch that is going according to plan; None means the
# lamp is an experiment rather than a routine reading, so "normal" is undefined.
FAILURE_LAMPS = [
    dict(id="free_thiol", name="游离巯基数",
         how="Ellman 试剂测还原后的抗体，数有几个巯基露出来",
         states=("low", "normal"), normal="normal",
         meaning="低 = 二硫键根本没开够，或者开了又氧化回去"),
    dict(id="dar", name="实测平均 DAR",
         how="HIC 或完整分子量质谱",
         states=("low", "normal"), normal="normal",
         meaning="低 = 最终挂上去的药比目标少"),
    dict(id="free_payload", name="反应结束时还剩多少游离药",
         how="反应液离心/超滤后测上清里的药浓度",
         states=("low", "normal", "high"), normal="normal",
         meaning="高 = 药还在但没挂上去；低 = 药没了（挂了、沉了、或降解了）"),
    dict(id="turbidity", name="反应液浊度/颗粒",
         how="浊度计或 DLS 看粒径分布",
         states=("normal", "high"), normal="normal",
         meaning="高 = 药自己在溶液里抱成团，能反应的浓度被压低"),
    dict(id="extend_time", name="反应时间翻倍后 DAR 涨不涨",
         how="同一批料重跑一次，时间加倍，再测 DAR",
         states=("yes", "no"), normal=None,
         meaning="涨 = 只是慢，时间不够；不涨 = 慢不是原因"),
    dict(id="high_dar", name="质谱里有没有高 DAR 物种",
         how="完整分子量质谱看 DAR6+ 的峰存不存在",
         states=("present", "absent"), normal=None,
         meaning="有 = 位点挂得动，只是没挂满；没有 = 某个位点压根挂不上去"),
]

# The pattern each cause leaves on the six lamps. A lamp left out of a
# signature means that cause does not constrain it either way.
FAILURE_SIGNATURES = [
    dict(id="reduction_short", name="TCEP 加少了，二硫键没开够",
         step="二硫键还原",
         signature=dict(free_thiol="low", dar="low", free_payload="high",
                        turbidity="normal", extend_time="no", high_dar="absent"),
         mechanism="已经打开的链间二硫键条数 vs TCEP 当量", tier="B",
         fix="加 TCEP 当量，或延长还原时间 / 提高还原温度"),
    dict(id="payload_clustering", name="payload 在共溶剂里自己抱团",
         step="共价连接（可用的药浓度）",
         signature=dict(free_thiol="normal", dar="low", free_payload="high",
                        turbidity="high", extend_time="no", high_dar="absent"),
         mechanism="工作浓度 ÷ 该共溶剂比例下的聚集临界浓度", tier="C",
         fix="提高共溶剂比例、降低投料浓度、加表面活性剂，或换更亲水的连接子"),
    dict(id="slow_kinetics", name="这个 pH 下反应太慢，时间不够",
         step="巯基活化 + 共价连接",
         signature=dict(free_thiol="normal", dar="low", free_payload="high",
                        turbidity="normal", extend_time="yes", high_dar="absent"),
         mechanism="硫醇阴离子分数 × 表观速率常数 × 有效暴露时间", tier="B",
         fix="提高 pH（不超过抗体稳定窗口）、延长时间、或提高投料浓度"),
    dict(id="steric_crowding", name="分子太大，铰链那几个位置塞不下",
         step="位点暴露度 + 几何位阻",
         signature=dict(free_thiol="normal", dar="low", free_payload="high",
                        turbidity="normal", extend_time="no", high_dar="absent"),
         mechanism="放置在真实 MD 帧上的重原子最小距离", tier="B",
         fix="换更紧凑的连接子、放弃最难的那几个位点、或降低目标 DAR"),
    dict(id="competition", name="两种 payload 抢同一批位点，一个挤掉另一个",
         step="组合统计（双载荷）",
         signature=dict(free_thiol="normal", dar="normal", free_payload="normal",
                        turbidity="normal", extend_time="no", high_dar="present"),
         mechanism="两种药的投料比 + 各自的体积惩罚，看谁占多少位",
         tier="C",
         fix="调投料比，或改成分支连接子把比例焊死在分子里"),
    dict(id="instability", name="payload 自己降解，或连接子自己断",
         step="模型外（稳定性，不是偶联）",
         signature=dict(free_thiol="normal", dar="low", free_payload="low",
                        turbidity="normal", extend_time="no", high_dar="absent"),
         mechanism=None, tier="C",
         fix="查药的储存与反应稳定性；本图谱只能把它和其他原因分开，不能定位它"),
]

# Causes the six lamps cannot separate, stated so the table is not mistaken for
# something it is not.
TRIAGE_OUTSIDE = [
    "抗体本身的氧化、糖型杂、聚体：不进图谱，它是输入条件不是偶联结果，要靠批次质谱当成输入项",
    "游离巯基又氧化回去：和「TCEP 加少了」留下一样的灯，要靠 Ellman 隔时间取样才分得开",
    "点击化学、酶偶联、糖基偶联：六个环节对不上，这套图谱不适用",
]


def match_failure_signature(observed):
    """Match observed lamp states against the cause signatures.

    `observed` maps lamp id -> measured state, e.g.
        {"free_thiol": "normal", "dar": "low", "free_payload": "high"}
    Lamps you have not measured are simply left out; the function then says
    which one to measure next.

    Returns ranked candidates, the causes the measurements rule out, and the
    single next reading with the best separating power.
    """
    lamp_ids = [l["id"] for l in FAILURE_LAMPS]
    observed = {k: v for k, v in (observed or {}).items() if v and k in lamp_ids}
    bad = [k for k in (observed or {}) if k not in lamp_ids]
    cands, excluded = [], []
    for cause in FAILURE_SIGNATURES:
        sig = cause["signature"]
        clashes = [(k, observed[k], sig[k])
                   for k in observed if k in sig and sig[k] != observed[k]]
        agreed = [k for k in observed if k in sig and sig[k] == observed[k]]
        row = dict(id=cause["id"], name=cause["name"], step=cause["step"],
                   tier=cause["tier"], mechanism=cause["mechanism"],
                   fix=cause["fix"], agreed=len(agreed),
                   lamps_explained=len(sig))
        if clashes:
            row["contradicted_by"] = [
                f"{k}: 测到 {got}，该原因要求 {exp}" for k, got, exp in clashes]
            excluded.append(row)
        else:
            row["supported_by"] = agreed
            cands.append(row)
    # rank: how much of the observed pattern this cause accounts for
    cands.sort(key=lambda r: -r["agreed"])
    # next reading: the unmeasured lamp that splits the remaining candidates
    unmeasured = [k for k in lamp_ids if k not in observed]
    next_lamp, best = None, None
    for k in unmeasured:
        groups = {}
        for c in cands:
            st = c["id"] and next(
                (s["signature"][k] for s in FAILURE_SIGNATURES
                 if s["id"] == c["id"] and k in s["signature"]), None)
            if st is None:
                continue
            groups.setdefault(st, []).append(c["id"])
        if len(groups) < 2:
            continue
        smallest = min(len(v) for v in groups.values())
        score = (len(groups), smallest)
        if best is None or score > best:
            best, next_lamp = score, k
    lamp_by_id = {l["id"]: l for l in FAILURE_LAMPS}
    nl = lamp_by_id.get(next_lamp) if next_lamp else None
    return dict(
        observed=observed, unknown_lamps=bad,
        candidates=cands, excluded=excluded,
        next_lamp=(dict(id=nl["id"], name=nl["name"], how=nl["how"],
                        splits=dict((k, v) for k, v in sorted(
                            (s["signature"][next_lamp], s["name"])
                            for s in FAILURE_SIGNATURES
                            if next_lamp in s["signature"])))
                   if nl else None),
        unmeasured=unmeasured, outside_model=TRIAGE_OUTSIDE,
        note=("Consistency, not proof. A candidate is only 'still possible'; "
              "the ranking says how much of what you measured it accounts for."),
    )


INPUT_CONTRACT = """
Fields that may be supplied in the state dict (omit any to skip its check;
do not pretend to have data you lack):
  dist                  list[float]  DAR 0..8 probabilities (sum to 1)
  mean_dar / dar_std    float        optional; derived on the fly from dist
  tcep_eq               float        reductant molar equivalents
  bonds_opened          float        number of opened bonds
  per_bond_open         list[float]  opening probability per bond
  reduction_completion  float        reduction completion (0-1)
  t_eff_s               list[float]  effective exposure time per site (seconds)
  thiolate_pct          float        thiolate percentage
  ph                    float        conjugation pH
  p_conj                float        conjugation probability
  kinetic_completion    float        k2 × [P] × t_eff
  payload_feed_ratio    float        payload charge equivalents
  target_dar            float        target DAR
"""
