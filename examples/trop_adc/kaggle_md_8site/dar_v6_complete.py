#!/usr/bin/env python
"""
dar_v6_complete.py — complete DAR model with review-found loopholes fixed
=============================================================================
What was fixed relative to v5
-----------------------------------------------------------------------------
1. thiol activation truly linked into the kinetic chain (it was disconnected in v5)
   v5: p_c = YIELD * (1 - exp(-K2 * [P] * t_eff))        <- no thiolate term
   v6: p_c = YIELD * (1 - exp(-k2_apparent * AUC))
      where k2_apparent = k2_ref * f_thiolate(pH) / f_thiolate(pH_ref).
      The thiolate fraction enters through this pH RATIO, not as a separate
      factor: k2_ref is already an apparent rate measured at pH_ref, so
      multiplying by f_thio again would double-count (see k2_apparent).

2. maleimide hydrolysis competition (v5 treated [P] as constant, inevitably overestimating at long times)
   [P](t) = [P]0 * exp(-k_hyd * t), k_hyd varies with pH

3. uncertainty propagation (v5 gave only a point estimate, no error bars)
   Monte Carlo: pKa fluctuation x replica/frame sampling -> confidence interval of mean DAR

4. multipoint calibration + leave-one-out validation framework (v5 was single-point inverse solve, essentially curve fitting)

★ A critical point that must be handled correctly: pKa cannot be a static value
-----------------------------------------------------------------------------
In the static pKa given by PROPKA for the reduced state, C-Cys214 = 20.76 (because that conformer is 82% buried).
Direct substitution:  f_thio = 1/(1+10^(20.76-7)) ≈ 1e-14  ->  p_c ≈ 0  ->  C214 judged dead
This contradicts the MD measurement: C214 has 11% of frames with SASA > 5 A^2, i.e. "opens occasionally" rather than "always buried".

Root cause: **site accessibility and thiol activation are not independent**.
   The reaction only happens during the "exposed fraction of time" (site accessibility already filters out buried conformers),
   so the pKa that determines the reaction rate must be the **exposed-conformer pKa**, not the conformer-averaged or buried-conformer pKa.
   Using the buried-conformer pKa amounts to penalizing "burial" twice (once at site accessibility, once again at thiol activation).

Therefore v6 uses the "exposure-conditioned pKa":
  - default takes the typical value for exposed cysteines, PKA_EFF = 8.5
  - use PKA_SD = 1.0 to represent conformational fluctuation (corresponding to review comment #4, ±1 unit)
  - sensitivity scan covers 7.0–11.0 to check whether the conclusion is robust

Data honesty annotation
-----------------------------------------------------------------------------
- C-Cys214 / A-Cys223 / B-Cys223 / A-Cys229: v5 MD measurement (2 ns x 2 replicas)
- remaining 4 sites (A232/B232/B229/D214): no MD, extrapolated by mirror/conservative
"""
import json, os, math, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results_hinge")
OUT = os.path.join(RES, "dar_v6_complete.json")
CFG_PATH = os.environ.get(
    "ADCSIM_CONFIG", os.path.join(HERE, "adc_model_config.yaml"))

# ---- All parameters come from config; no magic numbers left in code ----------------
# Fallback values are used only when the config file is missing; the normal path is always reading yaml.
_FALLBACK = {
    "process": dict(ph=7.0, mab_uM=10.0, reduction_time_h=2.0,
                    conjugation_time_h=2.0, payload_feed_ratio=6.0, tcep_eq=2.5,
                    reduction_temp_C=None, conjugation_temp_C=22.0,
                    ea_tcep_kJ_per_mol=53.0, ea_k2_kJ_per_mol=None),
    "chemistry": dict(k2_ref=300.0, k2_ph_ref=7.0, k_hyd=0.0, p_conj=0.995),
    "disulfide_reduction": dict(model="shared_pool", k_open=None, eta=1.0, n_bond=4),
    "site_accessibility": dict(sasa_lo=5.0, sasa_span=15.0,
                               mirror_policy="conservative"),
    "thiol_activation": dict(pka_eff=8.5, pka_sd=1.0, pka_scan=[7.0, 11.0]),
    "anchors": dict(tcep_eq=[2.00, 2.15, 2.50, 3.00],
                    mean_dar=[3.77, 3.97, 4.55, 5.38], source="user-provided",
                    reduction_temp_C=None),
    "dual_payload": dict(enabled=False, payload_A=None, payload_B=None,
                         feed_eq_A=None, feed_eq_B=None, k_ratio_A_over_B=1.0,
                         class_factor=None),
    "thresholds": dict(odd_dar_pct_warn=5.0, odd_dar_pct_crit=10.0,
                       high_dar_ge6_pct=25.0, low_dar_le1_pct=10.0, dar_std=2.0,
                       mean_dar_tolerance=0.5, p_conj_warn=0.98, p_conj_crit=0.95,
                       bond_yield_ratio_warn=0.85, bond_yield_ratio_crit=0.70,
                       min_bond_open_ratio=0.30, min_teff_s=300.0,
                       min_thiolate_pct=1.0, kinetic_completion_min=3.0,
                       ph_window=[6.5, 8.5]),
    "stats": dict(n_mc=4000, seed=20260918),
    "solutions": dict(reductant_solvent=dict(reagent="TCEP", stock_mM=20.0)),
}


# Repo root — used only to record config provenance portably. HERE/ROOT sit at
# <repo>/examples/trop_adc(/kaggle_md_8site), so two more hops reach <repo>.
_REPO_ROOT = os.path.dirname(os.path.dirname(ROOT))


def _portable_config_source(path):
    """Where the config came from, recorded without machine-specific paths.

    A config inside the repository becomes a repo-root-relative path; one
    outside it is reduced to its file name. Either way no absolute path is
    written into the output json. Computation reads CFG values, never this key.
    """
    if path == "fallback":
        return "fallback"
    abs_path = os.path.abspath(path)
    try:
        rel = os.path.relpath(abs_path, _REPO_ROOT)
    except ValueError:                          # different drive, on Windows
        rel = None
    if rel and not rel.startswith(os.pardir) and not os.path.isabs(rel):
        return rel
    return os.path.basename(abs_path)


def load_cfg(path=CFG_PATH):
    """Read yaml config; on missing or unreadable, fall back to defaults and warn on stderr."""
    if os.path.exists(path):
        try:
            import yaml
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
            # Union, not just the sections _FALLBACK knows about: a section that
            # exists only in the yaml used to be dropped silently, which is how
            # payload_qc (and everything in it) went missing for a while.
            keys = list(_FALLBACK) + [k for k in cfg if k not in _FALLBACK]
            merged = {k: dict(_FALLBACK.get(k, {}), **(cfg.get(k) or {})) for k in keys}
            merged["_source"] = _portable_config_source(path)
            return merged
        except Exception as e:                      # noqa: BLE001
            print(f"[warn] config read failed ({e}); using built-in fallback values", file=sys.stderr)
    print("[warn] adc_model_config.yaml not found; using built-in fallback values", file=sys.stderr)
    out = {k: dict(v) for k, v in _FALLBACK.items()}
    out["_source"] = "fallback"
    return out


CFG = load_cfg()

# ---- Expand into module-level constants (keep existing function signatures unchanged) --------------
PH = float(CFG["process"]["ph"])
MAB_UM = float(CFG["process"]["mab_uM"])
T_RED_H = float(CFG["process"]["reduction_time_h"])
T_H = float(CFG["process"]["conjugation_time_h"])
R_FEED = float(CFG["process"]["payload_feed_ratio"])
TCEP_CAL = float(CFG["process"]["tcep_eq"])

# ---- Temperature is an INPUT, not a hidden assumption (2026-09-26) ----------
# Why: the single fitted knob is K = k_TCEP x [mAb] x t_reduction, and k_TCEP moves
# a lot with temperature (assuming Q10 ~ 2, a 12 -> 37 C span changes it ~5.7x).
# A K fitted on anchors run at 37 C is simply the wrong number for a 22 C process.
# Rule set by the user: reduction temperature MUST be supplied (no default),
# conjugation temperature defaults to 22 C.
T_RED_C = CFG["process"].get("reduction_temp_C", None)
T_RED_C = None if T_RED_C is None else float(T_RED_C)
T_CONJ_C = float(CFG["process"].get("conjugation_temp_C", 22.0))
EA_TCEP = CFG["process"].get("ea_tcep_kJ_per_mol", 53.0)
EA_TCEP = None if EA_TCEP is None else float(EA_TCEP)
EA_K2 = CFG["process"].get("ea_k2_kJ_per_mol", None)
EA_K2 = None if EA_K2 is None else float(EA_K2)
R_GAS = 8.314462618          # J mol^-1 K^-1

# ---- pH of the REDUCTION step, and the pH the anchors were reduced at --------
# The conjugation pH (process.ph) was the only pH in the model until 2026-09-26.
# But the reduction step has its own, usually different pH, and TCEP's reactivity
# depends on it strongly (see TCEP_P_PKA below). Without this the model cannot be
# pointed at a protocol whose reduction pH differs from the anchors'.
PH_RED = float(CFG["process"].get("reduction_ph", PH))
ANCHOR_PH_RED = float(CFG["anchors"].get("reduction_ph", PH) or PH)

# ---- TCEP pH dependence: the nucleophilic phosphorus has a pKa of 7.6 --------
# Source: Cline DJ, Redding SE, Brohawn SG, Psathas JN, Schneider JP, Thorpe C.
#   "New water-soluble phosphines as reductants of peptide and protein disulfide
#   bonds: reactivity and membrane permeability." Biochemistry 2004;43(48):15195-203.
#   DOI 10.1021/bi048329a, PMID 15568811.
#   -> "phosphorus pK values of 6.8, 5.8 and 4.7 ... relative to that of TCEP
#      (pK = 7.6)". Only the deprotonated phosphorus is nucleophilic, so the
#      reduction rate carries the same sigmoid as a thiolate fraction.
# This is a MEASURED constant (tier B), unlike Ea above which is assumed.
TCEP_P_PKA = float(CFG["chemistry"].get("tcep_p_pka", 7.6))

# ---- Solution environment: how much of each payload is actually available ----
# A hydrophobic drug-linker in a mostly-aqueous medium self-associates once its
# concentration passes a critical aggregation concentration (CAC); the material
# in the aggregates is not reachable, so adding more of it stops helping. This
# is the term the user has been asking for since the payload-QC work started.
#
#   log10(CAC_water) = c0 - c1 * logP          (more hydrophobic -> lower CAC)
#   log10(CAC_mix)   = log10(CAC_water) + sigma * f_cosolvent
#                                              (more organic -> higher CAC)
#   [P]_effective    = min([P]_analytical, CAC_mix)
#   f_available      = [P]_effective / [P]_analytical
#
# ALL THREE PARAMETERS ARE TIER C. They are fitted on ONE payload pair in ONE
# medium (see adc_model_config.yaml), so they are a hypothesis with a functional
# form, not a validated model. See payload_availability() for what would test it.
SOL = CFG.get("solutions", {}) or {}
_MEDIUM = SOL.get("reaction_medium", {}) or {}
_PSOLV = SOL.get("payload_solvent", {}) or {}
COSOLVENT_VV = float(_MEDIUM.get("co_solvent_vv", 0.10))
COSOLVENT_NAME = str(_PSOLV.get("solvent", "DMSO"))
PSTOCK_MM = float(_PSOLV.get("stock_mM", 10.0))
_RSOLV = SOL.get("reductant_solvent", {}) or {}
TCEP_STOCK_MM = float(_RSOLV.get("stock_mM", 20.0))
_QC = CFG.get("payload_qc", {}) or {}
CAC_C0 = float(_QC.get("cac_logP_c0", -1.60))
CAC_C1 = float(_QC.get("cac_logP_c1", 0.60))
CAC_SIGMA = float(_QC.get("cosolvent_sigma", 1.60))

# ---- Steric discrimination across site classes (see class_factor_from_descriptors) ----
CLASS_GAMMA0 = float(_QC.get("class_gamma0", 4.41))
CLASS_CROWDING = dict(_QC.get("class_crowding", {"HC": 1.0, "LC": 0.0105}))

K2 = float(CFG["chemistry"]["k2_ref"])
PH_REF = float(CFG["chemistry"]["k2_ph_ref"])
K_HYD = float(CFG["chemistry"]["k_hyd"])
# Ceiling on the per-site conjugation probability.
# What it answers: once a thiol is open, exposed and activated, how often does a
# payload molecule actually end up covalently attached?
# Where it comes from: the covalent conjugation step computes it itself
# (raw conversion ~1.000 minus hydrolysis and steric loss). It is NOT a fitted
# number and NOT the old unsourced "process yield 0.92~0.98", which is deprecated
# and has no traceable source.
# When it is used: only as a fallback when a site has no MD-derived p_c
# (see `pc = P_CONJ if pc_uniform is None else pc_uniform`).
# Consequence: mean DAR cannot exceed 8 * P_CONJ = 7.96, so a target above that
# is reported as unreachable rather than silently met.
P_CONJ = float(CFG["chemistry"]["p_conj"])
YIELD_MAX = P_CONJ          # legacy alias, kept so older call sites keep working

SASA_LO = float(CFG["site_accessibility"]["sasa_lo"])
SASA_SPAN = float(CFG["site_accessibility"]["sasa_span"])

PKA_EFF = float(CFG["thiol_activation"]["pka_eff"])
PKA_SD = float(CFG["thiol_activation"]["pka_sd"])
PKA_SCAN = tuple(CFG["thiol_activation"]["pka_scan"])

N_BOND = int(CFG["disulfide_reduction"]["n_bond"])
ETA = float(CFG["disulfide_reduction"]["eta"])

ANCHOR_EQ = [float(x) for x in CFG["anchors"]["tcep_eq"]]
ANCHOR_DAR = [float(x) for x in CFG["anchors"]["mean_dar"]]
_ANCHOR_T = CFG["anchors"].get("reduction_temp_C", None)
if isinstance(_ANCHOR_T, (list, tuple)):
    ANCHOR_TEMP = [None if t is None else float(t) for t in _ANCHOR_T]
else:
    ANCHOR_TEMP = [None if _ANCHOR_T in (None, "") else float(_ANCHOR_T)] * len(ANCHOR_EQ)
# The anchors' own [mAb] and reduction time: back-solving k_TCEP from K_ref needs them.
ANCHOR_MAB_UM = float(CFG["anchors"].get("mab_uM", MAB_UM) or MAB_UM)
ANCHOR_T_RED_H = float(CFG["anchors"].get("reduction_time_h", T_RED_H) or T_RED_H)

DUAL = dict(CFG["dual_payload"])

# ---- Glycan conjugation: a second chemistry with its own site rule ----------
# Nothing here touches the cysteine route. The glycan route has no reduction
# step, its sites are the two conserved Fc N-glycans (Asn297, EU numbering),
# and attachment is a three-stage sequence (trim / transfer / click) rather
# than one bimolecular reaction. See adc_model_config.yaml:glycan_conjugation.
GLYCAN = dict(CFG.get("glycan_conjugation", {}) or {})

N_MC = int(CFG["stats"]["n_mc"])
SEED = int(CFG["stats"]["seed"])

PAIRS = [("A-Cys223--C-Cys214", "A-Cys223", "C-Cys214"),
         ("B-Cys223--D-Cys214", "B-Cys223", "D-Cys214"),
         ("A-Cys229--B-Cys229", "A-Cys229", "B-Cys229"),
         ("A-Cys232--B-Cys232", "A-Cys232", "B-Cys232")]

# v5 MD measurement: per-frame SASA files -> covered sites
MEASURED = {"A-Cys223": "A-Cys223_v5_sasa.npy",
            "B-Cys223": "B-Cys223_v5_sasa.npy",
            "A-Cys229": "A-Cys229_v5_sasa.npy",
            "C-Cys214": "C-Cys214_v5_sasa.npy"}
# mirror / conservative extrapolation (no MD, annotated)
MIRROR_OF = {"D-Cys214": "C-Cys214", "B-Cys229": "A-Cys229",
             "A-Cys232": "A-Cys229", "B-Cys232": "A-Cys229"}

# ---- Site classes -----------------------------------------------------------
# Two payloads competing for the same thiols do NOT compete with equal odds at
# every site: whether a given payload can physically get at a given cysteine
# depends on the payload (size, charge, hydrophobicity) AND on how cramped that
# particular site is. So the branching ratio has to carry a site dimension.
#
# Chain identity is derived from PAIRS topology rather than hard-coded letters,
# so this survives a PDB that labels its chains H/L or 1/2/3/4 instead of A/B/C/D.
_HC_CHAINS = set()
do = {}
for _p, _a, _b in PAIRS:
    key = tuple(sorted((_a.split("-")[0], _b.split("-")[0])))
    do[key] = do.get(key, 0) + 1
# The heavy-heavy pair is the one held together by MORE THAN ONE disulfide
# (IgG1 hinge: 2 inter-heavy bonds), while each heavy-light bond is single.
# Counting bonds per chain pair identifies the heavy chains without naming them,
# so a PDB labelled H/L or 1/2/3/4 works just as well.
for (_ca, _cb), _n in do.items():
    if _n >= 2:
        _HC_CHAINS.update((_ca, _cb))
if not _HC_CHAINS:
    raise SystemExit("[ERROR] could not identify the heavy chains from PAIRS")


def site_class(site):
    """(coarse, fine) class of one cysteine.

    coarse : "HC" | "LC"      — which kind of chain it sits on (this is the only
                                split the chain-level MS data can resolve)
    fine   : "hinge" | "fab_arm" | "light"
                              — region: the two hinge disulfides give 4 heavy
                                sites; the two H-L disulfides give 1 heavy site
                                (the one paired to the light chain) + 1 light site
    """
    for _p, a, b in PAIRS:
        if site not in (a, b):
            continue
        other = b if site == a else a
        ca, cb = site.split("-")[0], other.split("-")[0]
        same_kind = (ca in _HC_CHAINS) == (cb in _HC_CHAINS)
        coarse = "HC" if ca in _HC_CHAINS else "LC"
        fine = "hinge" if same_kind else ("fab_arm" if coarse == "HC" else "light")
        return coarse, fine
    raise KeyError(site)


def split_sites_by_class():
    """{coarse class: [sites]} — handy for reporting and for per-class calibration."""
    out = {}
    for s in _all_sites():
        out.setdefault(site_class(s)[0], []).append(s)
    return {k: sorted(v) for k, v in out.items()}


# =========================================================================
# Temperature correction (Arrhenius)
# -------------------------------------------------------------------------
# k(T) = k(T_ref) * exp[-Ea/R * (1/T - 1/T_ref)]
#
# Ea for the TCEP-disulfide step is BACK-DERIVED FROM Q10 ~ 2, which is a generic
# chemistry rule of thumb, NOT a measured activation energy for this reaction.
#   Ea = R * ln(Q10) * T1*T2 / (T2 - T1)  with Q10=2 over 25 K -> ~53 kJ/mol
# => TIER C. It must be labelled as an assumption in any report, and replaced the
# moment a measured temperature coefficient for TCEP reduction is found.
# Conjugation has no Ea by default (ea_k2_kJ_per_mol = null): k2_ref = 300 was taken
# as a room-temperature apparent rate, so at the default 22 C conjugation nothing moves.
# =========================================================================
def arrhenius(temp_C, ref_C, ea_kJ=EA_TCEP):
    """Rate multiplier going from ref_C to temp_C. Returns 1.0 when no Ea is set."""
    if ea_kJ is None or temp_C is None or ref_C is None:
        return 1.0
    T, Tr = temp_C + 273.15, ref_C + 273.15
    return float(math.exp(-(ea_kJ * 1000.0) / R_GAS * (1.0 / T - 1.0 / Tr)))


def k_at_temp(k_ref, temp_C, ref_C, ea_kJ=EA_TCEP):
    """Scale a rate / extent constant from its reference temperature to temp_C."""
    return k_ref * arrhenius(temp_C, ref_C, ea_kJ)


# =========================================================================
# Volume-exclusion correction (2026-09-29)
# -------------------------------------------------------------------------
# At high antibody concentration the molecules crowd each other, and some
# interchain disulfides are physically blocked by a neighbouring antibody.
# The excluded-volume fraction is  phi = c * NA * V_hydr, where V_hydr is the
# hydration-shell volume of one antibody. For IgG1 (MW ~150 kDa, R_hyd ~5.3
# nm) V_hydr ~620 nm^3. At 67 uM phi ~0.028 (negligible); at 167 uM (25 mg/mL)
# phi ~0.07 (still small but non-zero). The correction multiplies the
# reduction rate, not the TCEP feed: a blocked site sees no reductant, which
# is kinetically the same as a slower rate.
#
# No new fitted parameter: R_hydr is a physical constant of IgG1.
# TIER C: the excluded-volume model is a first-order (hard-sphere) approximation;
# it underestimates crowding at very high concentrations where soft repulsion
# and shape anisotropy matter. Report as a lower bound on the crowding effect.
# =========================================================================
_IGG1_MW_DA = 150_000.0
_IGG1_R_HYD_NM = 5.3
_NA = 6.02214076e23


def volume_exclusion_factor(mab_uM, r_hyd_nm=_IGG1_R_HYD_NM):
    """Fraction of disulfides still reachable at this antibody concentration.

    Returns (1 - phi) where phi is the excluded-volume fraction. At process
    concentrations (< 200 uM) this is close to 1 — the correction is small
    but physical and free of fitted parameters.
    """
    V_hydr_L = (4.0 / 3.0 * math.pi * (r_hyd_nm * 1e-7) ** 3) * 1e-3  # cm³ → L
    c_M = float(mab_uM) * 1e-6
    phi = c_M * _NA * V_hydr_L
    return float(max(0.0, 1.0 - phi))


# =========================================================================
# Local-charge correction (2026-09-29)
# -------------------------------------------------------------------------
# TCEP is an anion at process pH (phosphorus pKa 7.6). A patch of net
# negative charge near a disulfide repels it and slows the attack; a
# positive patch attracts it. The ss_reduction_profile.py already computes
# local_net_charge (signed sum of side-chain charges within 8 A of each SG)
# but does not feed it into the reduction rate — that is what we do here.
#
# The conversion from net charge to a rate multiplier uses a bounded
# exponential:  f = clip(exp(0.3 * q), 0.5, 2.0),  q = local net charge.
# The 0.3 factor is a rough estimate of the electrostatic screening effect
# on an ion approaching a charged patch in ionic-strength ~0.15 M buffer.
# TIER C: direction is physical, magnitude is an order-of-magnitude estimate.
# Bounded to [0.5, 2.0] so it cannot by itself move DAR beyond a factor of 2.
# =========================================================================
def charge_factor(local_net_q):
    """Rate multiplier from local electrostatics. Bounded [0.5, 2.0].

    Folded into the ease profile before K_ref fitting, so for the SAME antibody
    the effect is absorbed into K_ref and has no independent contribution to the
    prediction. It only shifts relative ease between bonds when switching to a
    different antibody with a different charge environment.
    """
    return float(max(0.5, min(2.0, math.exp(0.3 * local_net_q))))


def require_reduction_temp():
    """Reduction temperature is mandatory. There is deliberately no default."""
    if T_RED_C is None:
        raise SystemExit(
            "[ERROR] process.reduction_temp_C is not set.\n"
            "        K = k_TCEP x [mAb] x t, and k_TCEP depends strongly on temperature\n"
            "        (Q10 ~ 2 => 22 -> 37 C alone changes it ~2.7x).\n"
            "        Set reduction_temp_C in adc_model_config.yaml (or pass "
            "--reduction-temp). No default is provided on purpose.")
    return T_RED_C


def _reset_process_globals():
    """Restore every per-run mutable process global to the config default.

    Why this exists: main() applies caller / CLI overrides by rebinding these
    module globals (downstream functions take them as default arguments). Without
    a reset, a SECOND run inside the same Python process would silently inherit
    the FIRST run's temperature / concentration / TCEP / feed / durations.
    Resetting at the start of every run keeps the default behaviour identical to
    a fresh interpreter while still allowing per-run overrides.

    Only the six globals main() is allowed to override are reset; the fitted
    constants, anchors and chemistry parameters are config-level and never
    mutated by a run.
    """
    global T_RED_C, T_CONJ_C, MAB_UM, TCEP_CAL, T_RED_H, R_FEED
    T_RED_C = CFG["process"].get("reduction_temp_C", None)
    T_RED_C = None if T_RED_C is None else float(T_RED_C)
    T_CONJ_C = float(CFG["process"].get("conjugation_temp_C", 22.0))
    MAB_UM = float(CFG["process"]["mab_uM"])
    TCEP_CAL = float(CFG["process"]["tcep_eq"])
    T_RED_H = float(CFG["process"]["reduction_time_h"])
    R_FEED = float(CFG["process"]["payload_feed_ratio"])


# =========================================================================
def f_thiolate(pka, ph=PH):
    """Thiolate fraction = proportion of species that are actually nucleophilic"""
    return 1.0 / (1.0 + 10.0 ** (pka - ph))


def f_tcep_active(ph, pka=TCEP_P_PKA):
    """Fraction of TCEP whose nucleophilic phosphorus is in the active form.

    Same sigmoid as f_thiolate, but this is the REDUCTANT's pH dependence, not
    the cysteine's. Measured pKa = 7.6 (Cline 2004, see TCEP_P_PKA above), so at
    the pH values used in practice TCEP is far from saturated:

        pH 7.0 -> 0.201
        pH 7.4 -> 0.386
        pH 7.5 -> 0.443
        pH 8.0 -> 0.715

    i.e. moving reduction pH from 7.0 to 7.5 roughly DOUBLES the effective TCEP
    rate. Any K fitted at one pH must be divided by this fraction before it can
    be reused at another pH.
    """
    return 1.0 / (1.0 + 10.0 ** (pka - ph))


# =========================================================================
# ★ Double-counting trap (the single most important correction in this model)
# -------------------------------------------------------------------------
# Literature K2 = 300 M^-1 s^-1 is an **apparent rate measured at pH 7**, which already includes
# the thiolate fraction at pH 7. Multiplying by f_thio again would be double-counting.
#
# Empirical verdict (evidence from this project itself):
#   times f_thio    -> even/odd = 1.107   contradicts the experimental "even-DAR dominance"   ✗
#   without f_thio  -> even/odd = 5.60    consistent with experiment                         ✓
# Therefore K2 must be treated as an "apparent rate".
#
# Correct approach: anchor at the reference pH and scale by the **ratio** of thiolate fractions.
#   At ph == PH_REF, return K2_REF exactly (no double-counting)
#   Only when ph != PH_REF does the pH effect appear (this is the dimension DoE needs)
# =========================================================================
PH_REF = 7.0        # measurement pH of literature K2
K2_REF = 300.0      # apparent second-order rate at that pH


def k2_apparent(pka, ph=PH, ph_ref=PH_REF, k2_ref=K2_REF, temp_conj_C=None):
    ft, ft_ref = f_thiolate(pka, ph), f_thiolate(pka, ph_ref)
    k2_at_ref = k2_ref * (ft / ft_ref) if ft_ref > 0 else k2_ref
    if temp_conj_C is not None and EA_K2 is not None:
        k2_at_ref *= arrhenius(temp_conj_C, 22.0, EA_K2)
    return k2_at_ref


def auc_payload(t_eff, t_total=T_H * 3600.0, p0=None, k_hyd=K_HYD):
    """
    Integral of effective [P] over time (with hydrolysis decay).

    AUC = P0 * (t_eff/T) * (1 - exp(-k_hyd*T)) / k_hyd
    In the limit k_hyd -> 0: (1-exp(-k_hyd*T))/k_hyd -> T, so AUC -> P0 * t_eff  (consistent with v5)
    """
    if p0 is None:
        p0 = R_FEED * MAB_UM * 1e-6
    if k_hyd <= 1e-12:
        return p0 * t_eff
    return p0 * (t_eff / t_total) * (1.0 - math.exp(-k_hyd * t_total)) / k_hyd


def t_eff_from_frames(sasa):
    """site accessibility: per-frame gating -> effective exposure time (seconds). Irreversible reaction accumulates over time."""
    dt = T_H * 3600.0 / len(sasa)
    g = np.clip((sasa - SASA_LO) / SASA_SPAN, 0.0, 1.0)
    return float((g * dt).sum())


def pc_from_teff(t_eff, pka=PKA_EFF, k_hyd=K_HYD, ph=PH, chg_factor=1.0,
                 f_vol=1.0, t_conj=None):
    """thiol activation + covalent conjugation: full conversion.
    chg_factor : per-site charge multiplier (from same charge_factor module as
                 reduction), bounded [0.5,2.0].
    f_vol      : volume-exclusion multiplier, same factor as reduction.
    t_conj     : conjugation temperature for Arrhenius correction (None = skip).
    """
    ft = f_thiolate(pka, ph)
    auc = auc_payload(t_eff, k_hyd=k_hyd)
    k2 = k2_apparent(pka, ph=ph, temp_conj_C=t_conj) * chg_factor * f_vol
    raw = 1.0 - math.exp(-k2 * auc)
    return YIELD_MAX * raw, ft, auc, raw


# =========================================================================
#串联限制 (2026-09-28): every step can be the one that stops the payload
# -------------------------------------------------------------------------
# WHY THIS EXISTS. The old single-payload path asked only "is the reaction
# fast enough", so a few tens of micromolar were enough to fill all eight
# sites. Checked against the one failed batch we have -- 5 eq fed, eight
# thiols free, 8.2 % co-solvent, only 1.8 actually loaded -- it predicted
# 7.96. Off by a factor of 4.4. What was actually limiting there is the
# TOTAL USABLE EQUIVALENTS of payload, and that term was missing.
#
# But capping the feed alone would be the same mistake in reverse. Every step
# upstream can also be the one that stops the payload:
#
#   disulfide reduction   -> how many thiols exist at all
#   site accessibility    -> how long each thiol is actually exposed
#   thiol activation      -> what fraction is in the reacting form at this pH
#   steric fit            -> whether the molecule physically fits the site
#   payload availability  -> how many usable equivalents survive aggregation
#   reaction kinetics     -> whether the window is long enough
#
# So each step contributes a ceiling and the smallest one wins. Reporting
# WHICH one is smallest is the point: that is the attribution a failed batch
# needs, and it is the same number the forward prediction uses.
# =========================================================================

def conjugation_ceiling(t_eff_by_site, feed_eq, avail=1.0, n_sites=8,
                        bonds_opened=None, ph=PH, pka=PKA_EFF, k2=None,
                        k_hyd=K_HYD, steric_blocked=None, yield_max=YIELD_MAX,
                        mab_M=None, t_end_s=None):
    """DIAGNOSTIC / ATTRIBUTION UTILITY — not part of the main calculation path.

    The main model uses cap_by_available_drug() inside the Monte Carlo loop for
    drug-amount capping. This function is a standalone diagnostic that attributes
    DAR loss to each step (reduction, steric fit, kinetics, payload availability)
    for a single configuration. It is defined and tested but not called in main();
    wire it into the output JSON when step-level attribution is needed.

    Returns the per-site probability after the binding ceiling has been
    applied, the ceiling each step imposes, and the name of the binding one.
    Drug shortage compresses every site proportionally rather than truncating:
    a well-exposed thiol still wins the competition for what drug there is.
    """
    k2 = K2 if k2 is None else float(k2)
    blocked = set(steric_blocked or ())
    # sites handed in are the ones reduction actually freed
    candidates = list(t_eff_by_site or {})
    n_open = min(float(n_sites), 2.0 * float(bonds_opened)) \
        if bonds_opened is not None else float(len(candidates))
    # steric fit: a site the molecule cannot physically occupy drops out
    fit = [s for s in candidates if s not in blocked]
    # kinetics only runs on thiols reduction actually freed. Without this
    # truncation a half-reduced antibody still gets credit for all eight
    # sites, which inflates the kinetic ceiling and mis-charges the drug step.
    _allowed = int(round(n_open))
    if _allowed < len(fit):
        fit = fit[:_allowed]
    pc = {s: float(pc_from_teff(t_eff_by_site[s], pka=pka, k_hyd=k_hyd,
                                ph=ph)[0]) for s in fit}
    rate_cap = float(sum(pc.values()))
    # payload availability: usable equivalents, never more than was fed
    drug_cap = float(feed_eq) * float(avail)
    final = min(rate_cap, drug_cap, n_open)
    scale = (final / rate_cap) if rate_cap > 1e-12 else 0.0
    pc_scaled = {s: v * scale for s, v in pc.items()}
    # Attribution is "how many payloads would come back if this step were
    # fixed", not "which ceiling is numerically smallest" -- the latter
    # mislabels steric fit, which removes sites and therefore shrinks the
    # kinetic ceiling along with itself.
    lost = {
        "disulfide reduction": max(0.0, float(n_sites) - n_open),
        "steric fit": max(0.0, n_open - float(len(fit))),
        "accessibility + activation + kinetics": max(0.0, float(len(fit)) - rate_cap),
        "payload availability": max(0.0, rate_cap - final),
    }
    bottleneck = max(lost, key=lost.get)
    return dict(
        per_site_pc=pc_scaled, per_site_pc_unconstrained=pc,
        lost_dar={k: round(v, 4) for k, v in lost.items()},
        ceilings=dict(n_open=round(n_open, 3), n_fit=float(len(fit)),
                      rate=round(rate_cap, 4), drug=round(drug_cap, 4)),
        bottleneck=bottleneck,
        lost_to_bottleneck=round(lost[bottleneck], 4),
        scale=round(scale, 4),
        mean_dar=round(float(sum(pc_scaled.values())), 4),
        tier="B",
        note=("Each step is charged with the DAR it costs. 'disulfide reduction' "
              "counts thiols that never existed, 'steric fit' counts sites the "
              "molecule cannot occupy, 'accessibility + activation + kinetics' "
              "counts exposed thiols that ran out of time, and 'payload "
              "availability' counts what aggregation or a short feed took away. "
              "For a dual-payload batch use dual_loading_mass_balance: it carries "
              "both the feed cap and the two payloads competing for the same "
              "thiols."),
    )


def cap_by_available_drug(pcm, feed_eq, avail=1.0, open_by_pair=None):
    """Scale per-site probabilities so the total cannot exceed the usable feed.

    `open_by_pair` is the reduction result, {pair_name: P(bond open)}. Give it:
    the demand for drug is the loading the antibody is actually on course for,
    i.e. opened bonds x their two thiols, NOT all eight thiols. Charging the
    cap against all eight over-states the demand and therefore over-applies the
    cap -- a half-reduced antibody was being cut to a third of its real DAR.

    A payload fed at 6 equivalents cannot load more than 6, however fast the
    reaction is. When the feed is the binding constraint the shortfall is shared
    out proportionally rather than truncating whole sites: a well-exposed thiol
    still wins the competition for what drug there is.

    Returns (scaled pcm, scale applied). scale == 1.0 means the feed was not
    binding, which is the case for every batch fed in excess -- this is a no-op
    there, by design.
    """
    if open_by_pair is None:
        tot = float(sum(pcm.values()))
    else:
        tot = float(sum(float(open_by_pair.get(p, 1.0))
                        * (pcm.get(a, 0.0) + pcm.get(b, 0.0))
                        for p, a, b in PAIRS))
    cap = float(feed_eq) * float(avail)
    if tot <= cap or tot <= 1e-12:
        return dict(pcm), 1.0
    scale = cap / tot
    return {k: v * scale for k, v in pcm.items()}, scale


# =========================================================================
# disulfide reduction (2026-09-19 upgrade): shared reductant pool + stoichiometric conservation
# -------------------------------------------------------------------------
# Old form: r_i = 1 - exp(-λ * ease_i * eq)  implicitly assumes "TCEP is always in excess".
# In this system eq=2~3 and the bond count 4 are of the same order of magnitude; TCEP gets consumed, so this assumption does not hold.
#
# New form: four bonds compete for the same TCEP; each bond opened consumes one share; the reductant is exhausted and automatically capped at the limit.
#   dS_i/dt = -k * ease_i * S_i * T(t),   T(t) = eta*eq - Σ(1 - S_i)
#   substitute J = k*∫T dt, τ = k*t  =>  dJ/dτ = eta*eq - n + Σ exp(-ease_i * J)
#   final r_i = 1 - exp(-ease_i * J(K))
#
# ★ The only knob K (total opening extent) = k_TCEP × [mAb] × t_reduction
#   All three factors have units and can be measured separately -> K can be checked against literature, no longer a purely inverse-solved number.
#   Back-solving to an implied k_TCEP within the literature-measured range 1.5~6.25 M⁻¹s⁻¹ counts as passing.
# =========================================================================
PAIR_NAMES = [p[0] for p in PAIRS]
_TAB = {}
_TAU_MAX, _DT = 62.0, 0.002


def _rhs(J, ease, eq, eta):
    return eta * eq - len(ease) + float(np.sum(np.exp(-ease * J)))


def _J_table(eq, ease, eta=ETA):
    """The J(τ) trajectory depends only on (eq, ease), independent of K — integrate once, interpolate for any K via lookup table."""
    key = (round(float(eq), 9), tuple(np.round(ease, 9)), eta)
    if key in _TAB:
        return _TAB[key]
    n = int(_TAU_MAX / _DT)
    Js, J = np.zeros(n + 1), 0.0
    for i in range(n):
        k1 = _rhs(J, ease, eq, eta)
        k2 = _rhs(J + 0.5 * _DT * k1, ease, eq, eta)
        k3 = _rhs(J + 0.5 * _DT * k2, ease, eq, eta)
        k4 = _rhs(J + _DT * k3, ease, eq, eta)
        J = max(J + _DT * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0, 0.0)
        Js[i + 1] = J
    tau = np.arange(n + 1) * _DT
    _TAB[key] = (tau, Js)
    return tau, Js


def ease_norm(ease):
    """Normalize by the mean: keep only the "which bond opens easily" shape; fold the overall scale into K."""
    e = np.asarray([ease[p] for p in PAIR_NAMES], float) if isinstance(ease, dict) \
        else np.asarray(ease, float)
    return e / e.mean()


def exposure_J(eq, ease, K, eta=ETA):
    """Total opening extent J(K). Dimensionless, equal to k_TCEP × ∫[TCEP]dt."""
    tau, Js = _J_table(eq, ease_norm(ease), eta)
    return float(np.interp(K, tau, Js))


def pred_shared(eq, ease, K, eta=ETA):
    """disulfide reduction, new form: returns {bond name: opening probability}"""
    en = ease_norm(ease)
    J = exposure_J(eq, ease, K, eta)
    r = 1.0 - np.exp(-en * J)
    return {n: float(v) for n, v in zip(PAIR_NAMES, r)}


def pred_pseudo_first(eq, ease, lam):
    """disulfide reduction, legacy form (pseudo-first-order), for comparison only"""
    return {p: 1.0 - math.exp(-lam * ease[p] * eq) for p in PAIR_NAMES}


def mean_from_pred(pred, pc_uniform=None):
    """Closed-form mean: each opened bond contributes 2 thiols × conjugation probability.
    full_dist is exactly equivalent when pa=pb=pc and pen=1; this is only for fitting speed."""
    pc = P_CONJ if pc_uniform is None else pc_uniform
    return 2.0 * pc * float(np.sum([pred[p] for p in PAIR_NAMES]))


def _norm_anchor_temps(anchor_temps, n_rows):
    """Normalize per-anchor reduction temperatures into a list of length n_rows."""
    if anchor_temps is None:
        a = list(ANCHOR_TEMP)
    elif isinstance(anchor_temps, (int, float)):
        a = [float(anchor_temps)] * n_rows
    else:
        a = list(anchor_temps)
    if len(a) < n_rows:
        a = a + [a[-1] if a else None] * (n_rows - len(a))
    return a[:n_rows]


def anchor_temp_scales(anchor_temps, temp_ref=None, n_rows=None):
    """Per-anchor multiplier that carries K from temp_ref to each anchor's temperature.

    Returns (scales, temp_ref, note). With every anchor at one temperature the fit
    is unchanged from before; the point of the temperature term only shows up when
    anchors span temperatures, or when predicting at a temperature other than theirs.
    """
    n_rows = n_rows if n_rows is not None else len(ANCHOR_EQ)
    at = _norm_anchor_temps(anchor_temps, n_rows)
    known = [t for t in at if t is not None]
    note = None
    if not known:
        return [1.0] * n_rows, None, "no reduction temperature recorded for these anchors -> no temperature correction applied"
    if temp_ref is None:
        uniq = sorted(set(known))
        temp_ref = uniq[0] if len(uniq) == 1 else float(np.mean(uniq))
        if len(uniq) > 1:
            note = (f"anchors span {uniq} C; reference set to the mean ({temp_ref:.1f} C)")
    scales = [arrhenius(t, temp_ref) if t is not None else 1.0 for t in at]
    return scales, temp_ref, note


def fit_K(ease, obs=None, eta=ETA, lo=1e-3, hi=60.0, n=160, refine=60,
          pc_uniform=None, anchor_temps=None, temp_ref=None):
    """Fit the only knob K from measured anchors (coarse scan + ternary refinement)

    pc_uniform must be the conjugation probability the run is actually using.
    K and p_c are degenerate in the product (2 * p_c * opened bonds), so fitting
    K at the default 0.995 while the run uses a different p_c would silently
    misattribute the difference to the reduction step.

    anchor_temps: reduction temperature (C) of each anchor row. K is temperature
    dependent, so a K fitted on 37 C anchors is the wrong number at 22 C. The
    returned K_ref refers to temp_ref (the fitted value at that ONE temperature);
    use k_at_temp() to move it anywhere else.
    """
    obs = list(zip(ANCHOR_EQ, ANCHOR_DAR)) if obs is None else list(obs)
    scales, t_ref, note = anchor_temp_scales(anchor_temps, temp_ref, len(obs))

    def rms(K_ref):
        tot = 0.0
        for (e, d), sc in zip(obs, scales):
            K_i = K_ref * sc
            tot += (mean_from_pred(pred_shared(e, ease, K_i, eta), pc_uniform) - d) ** 2
        return math.sqrt(tot / len(obs))
    xs = np.linspace(lo, hi, n)
    vals = [rms(x) for x in xs]
    i = int(np.argmin(vals))
    a, b = xs[max(i - 1, 0)], xs[min(i + 1, n - 1)]
    for _ in range(refine):
        m1, m2 = a + (b - a) / 3, b - (b - a) / 3
        if rms(m1) < rms(m2):
            b = m2
        else:
            a = m1
    K = float((a + b) / 2)
    info = dict(rmse=round(rms(K), 4), n_anchors=len(obs),
                temp_ref_C=t_ref,
                residuals=[round(mean_from_pred(pred_shared(e, ease, K * sc, eta), pc_uniform) - d, 3)
                           for (e, d), sc in zip(obs, scales)])
    if EA_TCEP is not None:
        info["ea_kJ_per_mol"] = EA_TCEP
        info["ea_source"] = "back-derived from Q10~2 (generic rule of thumb, NOT measured) -> TIER C"
    if note:
        info["temp_note"] = note
    return K, info


def leave_one_out_K(ease, obs=None, eta=ETA, pc_uniform=None,
                    anchor_temps=None, temp_ref=None):
    """leave-one-out validation: hold out one anchor from fitting and use it to test predictive ability"""
    obs = list(zip(ANCHOR_EQ, ANCHOR_DAR)) if obs is None else list(obs)
    if len(obs) < 2:
        return dict(status="SKIPPED", reason="fewer than 2 anchors")
    rows = []
    for i in range(len(obs)):
        train = [o for j, o in enumerate(obs) if j != i]
        tr_temps = [t for j, t in enumerate(_norm_anchor_temps(anchor_temps, len(obs))) if j != i]
        K, _ = fit_K(ease, train, eta, pc_uniform=pc_uniform,
                     anchor_temps=tr_temps, temp_ref=temp_ref)
        eq, actual = obs[i]
        ti = _norm_anchor_temps(anchor_temps, len(obs))[i]
        sc = arrhenius(ti, temp_ref) if (ti is not None and temp_ref is not None) else 1.0
        pm = mean_from_pred(pred_shared(eq, ease, K * sc, eta), pc_uniform)
        rows.append(dict(held_out_tcep=eq, actual_mean_dar=actual,
                         predicted_mean_dar=round(pm, 3), error=round(pm - actual, 3)))
    return dict(status="OK", n_folds=len(rows), rows=rows,
                mean_abs_error=round(float(np.mean([abs(r["error"]) for r in rows])), 4))


def implied_k_tcep(K, mab_uM=None, t_red_h=None):
    """Convert K into a checkable physical quantity: k_TCEP = K / ([mAb] × t_reduction)"""
    mab_uM = MAB_UM if mab_uM is None else mab_uM
    t_red_h = T_RED_H if t_red_h is None else t_red_h
    return K / (mab_uM * 1e-6 * t_red_h * 3600.0)


def k_tcep_intrinsic(K_at_anchor, mab_uM=None, t_red_h=None, ph=None, temp_C=None,
                     temp_ref_C=None):
    """Strip every protocol-specific factor out of a fitted K, leaving the rate constant.

        K = k_intrinsic x f_TCEP(pH) x arrhenius(T) x [mAb] x t_reduction

    A fitted K is only meaningful for the protocol it was fitted on. Peeling the
    pH and temperature factors off first is what makes it portable: the result
    is a genuine second-order rate constant that can be recombined with any other
    protocol's pH / temperature / concentration / duration.
    """
    mab_uM = ANCHOR_MAB_UM if mab_uM is None else mab_uM
    t_red_h = ANCHOR_T_RED_H if t_red_h is None else t_red_h
    ph = ANCHOR_PH_RED if ph is None else ph
    k = implied_k_tcep(K_at_anchor, mab_uM, t_red_h)
    k = k / f_tcep_active(ph)
    if temp_C is not None and temp_ref_C is not None:
        k = k / arrhenius(temp_C, temp_ref_C)
    return k


def K_for_protocol(k_intrinsic, temp_C=None, temp_ref_C=None, ph=None,
                   mab_uM=None, t_red_h=None):
    """Recombine k_intrinsic with a NEW protocol's pH / T / [mAb] / duration."""
    temp_C = T_RED_C if temp_C is None else temp_C
    ph = PH_RED if ph is None else ph
    mab_uM = MAB_UM if mab_uM is None else mab_uM
    t_red_h = T_RED_H if t_red_h is None else t_red_h
    k = k_intrinsic * f_tcep_active(ph)
    if temp_C is not None and temp_ref_C is not None:
        k = k * arrhenius(temp_C, temp_ref_C)
    return k * mab_uM * 1e-6 * t_red_h * 3600.0


# =========================================================================
def full_dist(pc, pred, pen=1.0):
    """Poisson-binomial convolution of 4 disulfide pairs (each pair contributes 0/1/2) — not 8 independent Bernoulli"""
    poly = np.array([1.0])
    for p, a, b in PAIRS:
        pa, pb, pr = pc[a], pc[b], pred[p]
        p2 = pr * pa * pb * pen
        p1 = pr * (pa * (1 - pb) + pb * (1 - pa) + (1 - pen) * pa * pb)
        poly = np.convolve(poly, np.array([p2, p1, 1 - p2 - p1]))
    d = poly[::-1][:9]
    return d / d.sum()


def _conv2d(a, k):
    """2-D discrete convolution, small arrays only (<= 9x9 here)."""
    out = np.zeros((a.shape[0] + k.shape[0] - 1, a.shape[1] + k.shape[1] - 1))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i, j]:
                out[i:i + k.shape[0], j:j + k.shape[1]] += a[i, j] * k
    return out


def full_dist_dual(pcA, pcB, pred, pen=1.0):
    """Joint distribution of (n_A, n_B) when two payloads share the thiol pool.

    Two maleimide payloads A and B compete for the same thiols. Per thiol the
    rates simply add, so

        total conversion = 1 - exp(-(k2_A*AUC_A + k2_B*AUC_B))
        goes to A with probability  k2_A*AUC_A / (k2_A*AUC_A + k2_B*AUC_B)

    The branching ratio is the ONLY new quantity the dual-payload case needs.
    It is not fitted: for two maleimides it is the ratio of the two
    second-order rate constants, and at equal k2 it collapses to the feed
    ratio. Both are measurable independently, so this stays zero-parameter.

    Each of the 8 sites ends up as A / B / nothing, and the joint distribution
    is the convolution -- n_A and n_B are NOT independent, because a thiol
    taken by A cannot also be taken by B.
    """
    poly = np.zeros((1, 1)); poly[0, 0] = 1.0
    for p, a, b in PAIRS:
        pr = pred[p]
        for s in (a, b):
            qA, qB = pr * pcA[s] * pen, pr * pcB[s] * pen
            ker = np.zeros((2, 2))
            ker[0, 0], ker[1, 0], ker[0, 1] = 1.0 - qA - qB, qA, qB
            poly = _conv2d(poly, ker)
    return poly / poly.sum()


def summarize_dual(poly):
    na = np.arange(poly.shape[0])[:, None] * np.ones((1, poly.shape[1]))
    nb = np.ones((poly.shape[0], 1)) * np.arange(poly.shape[1])[None, :]
    tot = na + nb
    ma = float((na * poly).sum()); mb = float((nb * poly).sum())
    mt = float((tot * poly).sum())
    return dict(dar_A=round(ma, 3), dar_B=round(mb, 3), dar_total=round(mt, 3),
                split_A=round(ma / mt, 4) if mt > 0 else None,
                argmax=(int(np.unravel_index(poly.argmax(), poly.shape)[0]),
                        int(np.unravel_index(poly.argmax(), poly.shape)[1])))


def summarize(d):
    m = float((np.arange(9) * d).sum())
    ev, od = d[0::2].sum(), d[1::2].sum()
    return dict(mean_dar=round(m, 3),
                std_dar=round(float(np.sqrt((np.arange(9) ** 2 * d).sum() - m ** 2)), 3),
                mode_dar=int(d.argmax()),
                even_odd_ratio=round(float(ev / od), 3) if od > 1e-12 else None,
                p_dar_ge6=round(float(d[6:].sum()), 4),
                dist=[round(float(x), 4) for x in d])


# =========================================================================
def solve_lambda(ease, target, pc, tcep):
    """single-point inverse solve for LAMBDA (v5 behavior, retained for comparison)"""
    lo, hi = 0.001, 500.0
    for _ in range(200):
        mid = (lo + hi) / 2
        pred = {p: 1 - math.exp(-mid * ease[p] * tcep) for p in ease}
        if (np.arange(9) * full_dist(pc, pred)).sum() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def solve_lambda_multipoint(ease, obs, pc):
    """
    Multipoint calibration: obs = [(tcep, mean_dar), ...], least-squares fit of a single LAMBDA.
    With only one point it degenerates to solve_lambda (and issues a non-falsifiable warning).
    """
    if len(obs) < 2:
        lam = solve_lambda(ease, obs[0][1], pc, obs[0][0])
        return lam, dict(warning="only one calibration anchor -> essentially curve fitting, not falsifiable; "
                                 "need measured DAR at different TCEP equivalents to enable leave-one-out validation")
    grid = np.linspace(0.001, 500.0, 4000)
    best, best_ss = None, float("inf")
    for g in grid:
        ss = 0.0
        for tcep, md in obs:
            pred = {p: 1 - math.exp(-g * ease[p] * tcep) for p in ease}
            ss += ((np.arange(9) * full_dist(pc, pred)).sum() - md) ** 2
        if ss < best_ss:
            best, best_ss = g, ss
    return float(best), dict(n_points=len(obs), rmse=round(math.sqrt(best_ss / len(obs)), 4))


def leave_one_out(ease, obs, pc):
    """leave-one-out validation: hold out one point from fitting and use it to test predictive ability"""
    if len(obs) < 2:
        return dict(status="SKIPPED", reason="fewer than 2 anchors; leave-one-out validation not possible")
    rows = []
    for i in range(len(obs)):
        train = [o for j, o in enumerate(obs) if j != i]
        lam, _ = solve_lambda_multipoint(ease, train, pc)
        tcep, actual = obs[i]
        pred = {p: 1 - math.exp(-lam * ease[p] * tcep) for p in ease}
        d = full_dist(pc, pred)
        pm = float((np.arange(9) * d).sum())
        rows.append(dict(held_out_tcep=tcep, actual_mean_dar=actual,
                         predicted_mean_dar=round(pm, 3),
                         error=round(pm - actual, 3)))
    errs = [abs(r["error"]) for r in rows]
    return dict(status="OK", n_folds=len(rows), rows=rows,
                mean_abs_error=round(float(np.mean(errs)), 4))


# =========================================================================
# Inverse solve — target DAR -> TCEP equivalents (the TRUE inverse)
# -------------------------------------------------------------------------
# The forward chain is monotonic in TCEP equivalents:
#     more TCEP -> more bonds open -> more thiols released -> higher mean DAR
# so a target mean DAR can be inverted by root-finding (bisection, dependency-free;
# the mean-DAR-vs-TCEP curve is provably monotonic so bisection is exact).
#
# LP equivalents are NOT the controlling knob (see min_lp_eq_for_target_dar): in the
# operating window LP is in heavy excess, so DAR is set by TCEP, not by LP. Reporting
# an "LP eq that yields DAR X" would be misleading — we only report the minimum LP
# feed that keeps LP from becoming the bottleneck.
# =========================================================================
def _all_sites():
    return sorted({a for p, a, b in PAIRS for a in (a, b)})


def mean_dar_from_eq(eq, ease, K, pc, eta=ETA):
    """Forward helper used by the inverse: TCEP eq -> mean DAR through the full
    combinatorial (Poisson-binomial) chain. pc = per-site conjugation-probability dict."""
    pred = pred_shared(eq, ease, K, eta)
    d = full_dist(pc, pred)
    return float((np.arange(9) * d).sum())


def max_reachable_mean_dar(pc):
    """Saturation ceiling of the mean DAR: every bond opened, every released thiol
    conjugated. Equals sum over the 8 cysteines of their conjugation probability.
    A target above this ceiling is physically UNREACHABLE — report it, never fake it."""
    return float(np.sum([pc[s] for s in _all_sites()]))


def solve_tcep_eq_for_target_dar(target_dar, ease, K, pc, eta=ETA,
                                 eq_lo=1e-3, eq_hi=80.0, xtol=1e-7):
    """Inverse (true): recover the TCEP equivalents that yield a target mean DAR.

    Raises ValueError (with the reachable ceiling) when the target cannot be met:
      * target_dar <= 0                                 -> not physically meaningful
      * target_dar > max_reachable_mean_dar(pc) + 1e-9  -> saturation plateau reached

    The bisection bracket is [eq_lo, eq_hi]; f(eq_lo) < 0 (mean ~ 0) and f(eq_hi) > 0
    (mean at the saturation ceiling), which holds for any reachable target.
    """
    if target_dar <= 0:
        raise ValueError(f"target DAR must be > 0 (got {target_dar:.4f})")
    ceiling = max_reachable_mean_dar(pc)
    if target_dar > ceiling + 1e-9:
        raise ValueError(
            f"target DAR {target_dar:.3f} is UNREACHABLE. Disulfide reduction + covalent "
            f"conjugation saturates at max mean DAR = {ceiling:.3f} (all 4 interchain bonds "
            f"opened, site conjugation probability capped at p_conj={P_CONJ}). Raising the "
            f"ceiling needs higher p_conj / site accessibility or more interchain disulfides "
            f"-- not more TCEP.")

    def f(eq):
        return mean_dar_from_eq(eq, ease, K, pc, eta) - target_dar

    flo, fhi = f(eq_lo), f(eq_hi)
    if not (flo <= 0 <= fhi):
        raise RuntimeError(
            f"inverse bracket failed: f({eq_lo})={flo:.4f}, f({eq_hi})={fhi:.4f}; "
            f"mean DAR is not monotonic in TCEP eq within [{eq_lo}, {eq_hi}]")
    lo, hi = eq_lo, eq_hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= xtol:
            break
    return 0.5 * (lo + hi)


def min_lp_eq_for_target_dar(target_dar, ease, K, pc, eta=ETA, lp_overage_margin=0.5):
    """Minimum LP (linker-payload) feed equivalents that keeps LP OUT of the
    rate-limiting regime.

    IMPORTANT: this answers "at least how much LP", NOT "how much LP yields this DAR".
    DAR is controlled by TCEP (see solve_tcep_eq_for_target_dar). LP is currently in
    excess over the released thiols, so within the window it does not set DAR.

    released thiols at the target = 2 * (bonds opened) = 2 * sum(pred_shared(eq*))
    where eq* is the TCEP equivalent that solves for target_dar.
    LP_min = released_thiols * (1 + lp_overage_margin).

    lp_overage_margin is the user-chosen safety buffer (default 0.5 = 50% excess).
    At exact stoichiometry (margin 0) LP merely matches the released thiols; a positive
    margin keeps the conjugation driven to completion and buffered against maleimide
    hydrolysis / sampling error.
    """
    eq_target = solve_tcep_eq_for_target_dar(target_dar, ease, K, pc, eta)
    released = 2.0 * float(np.sum(list(pred_shared(eq_target, ease, K, eta).values())))
    return dict(tcep_eq=eq_target,
                released_thiols_eq=released,
                lp_min_eq=released * (1.0 + lp_overage_margin),
                lp_overage_margin=lp_overage_margin)


def lp_feed_at_pc_floor(t_eff_by_site, pc_floor=0.98, ph=PH):
    """The LP feed (equivalents) at which the LEAST accessible site's conjugation
    probability first drops to pc_floor — i.e. the point where LP starts to become
    the bottleneck. Derived from the conjugation kinetics:
        raw = 1 - exp(-k2 * p0 * t_eff),  p0 = R_FEED * [mAb] * 1e-6  (k_hyd = 0)
    Inverted for R_FEED at each site; the binding constraint is the largest required
    feed (the least accessible site = smallest t_eff).
    Returns {'threshold_eq': <binding LP feed>, 'per_site_eq': {...}}.
    If the actual LP feed >> threshold_eq, LP is NOT limiting in this window.
    """
    k2 = k2_apparent(PKA_EFF, ph=ph)
    per_site = {}
    for s, te in t_eff_by_site.items():
        raw = min(max(pc_floor / YIELD_MAX, 1e-12), 1.0 - 1e-12)
        need = -math.log(1.0 - raw)            # = k2 * AUC = k2 * p0 * t_eff
        p0 = need / (k2 * te) if te > 0 else float("inf")
        per_site[s] = p0 / (MAB_UM * 1e-6)
    return dict(threshold_eq=float(max(per_site.values())), per_site_eq=per_site)


def pc_from_feed(t_eff, feed_eq, pka=PKA_EFF, ph=PH, k_hyd=K_HYD):
    """Conjugation probability at an arbitrary payload feed (equivalents vs antibody).
    Same kinetics as pc_from_teff, only the payload concentration changes."""
    k2 = k2_apparent(pka, ph=ph)
    auc = auc_payload(t_eff, k_hyd=k_hyd, p0=feed_eq * MAB_UM * 1e-6)
    raw = 1.0 - math.exp(-k2 * auc)
    return YIELD_MAX * raw, auc, raw


def cosolvent_fraction(feed_eq_total, mab_uM=None, stock_mM=None, antibody_vol_frac=1.0,
                       tcep_eq=0.0, tcep_mM=None):
    """Organic co-solvent fraction (v/v) actually present in the reaction mixture.

    This is not a free parameter: it follows from how the protocol was pipetted.
    The payload arrives as a stock solution in DMSO/DMA, so feeding more
    equivalents silently adds more organic solvent. Typical processes land at
    5-15 % v/v, and the antibody starts to suffer above ~20 %.

        f = V_organic / (V_antibody_solution + V_organic + V_TCEP)
    """
    mab_uM = MAB_UM if mab_uM is None else mab_uM
    stock_mM = PSTOCK_MM if stock_mM is None else stock_mM
    tcep_mM = TCEP_STOCK_MM if tcep_mM is None else float(tcep_mM)
    # per litre of antibody solution: feed_eq * [mAb] moles of payload
    v_org = float(feed_eq_total) * mab_uM * 1e-6 / (stock_mM * 1e-3)      # L per L
    v_tcep = float(tcep_eq) * mab_uM * 1e-6 / (tcep_mM * 1e-3)
    return float(v_org / (antibody_vol_frac + v_org + v_tcep))


def payload_availability(analytical_M, cac_M=None, logP=None, cosolvent_vv=None):
    """Fraction of a payload present as reactive monomer, and why.

    Above the critical aggregation concentration the extra material goes into
    aggregates whose maleimide is buried, so the effective concentration
    saturates at the CAC. Returns (f_available, diagnostics).

    Give either cac_M directly (best) or a logP to estimate it. With neither,
    returns 1.0 and says so — the model then behaves as if everything dissolves,
    which is the old behaviour and is only correct for a genuinely soluble
    payload.
    """
    f_cos = COSOLVENT_VV if cosolvent_vv is None else float(cosolvent_vv)
    if cac_M is None:
        if logP is None:
            return 1.0, dict(status="NO_DATA",
                             reason="neither cac_M nor logP supplied; assuming fully available",
                             f_available=1.0)
        cac_M = 10.0 ** (CAC_C0 - CAC_C1 * float(logP))
        src = f"estimated from logP {float(logP):.2f}"
    else:
        src = "given directly"
    cac_mix = float(cac_M) * 10.0 ** (CAC_SIGMA * f_cos)
    a = float(analytical_M)
    if a <= 0:
        return 1.0, dict(status="OK", cac_M=cac_mix, f_available=1.0)
    fav = min(1.0, cac_mix / a)
    return fav, dict(status="OK", cac_water_M=float(cac_M), cac_in_medium_M=cac_mix,
                     cosolvent_vv=f_cos, cosolvent=COSOLVENT_NAME,
                     analytical_M=a, effective_M=min(a, cac_mix),
                     f_available=round(fav, 4), cac_source=src,
                     saturated=(a > cac_mix),
                     tier="C — fitted on one payload pair in one medium")


def availability_for_payload(name, feed_eq, mab_uM=None, tcep_eq=None,
                             cosolvent_vv=None, total_feed_eq=None):
    """Availability of a payload that has a measured anchor, by name.

    payload_availability needs a CAC, which needs either a measurement or a
    logP. This is the branch for the first case: a payload that has been run
    alone at a known feed with a known outcome, so its usable ceiling is
    anchored rather than guessed. Anything not in the table falls back to the
    logP route (or to 1.0, the old behaviour).
    """
    if not name:
        return 1.0, dict(status="NO_NAME")
    rec = _QC.get("known_payloads", {}).get(str(name).strip().upper())
    if not rec:
        return 1.0, dict(status="UNKNOWN_PAYLOAD", payload=name)
    mab_uM = MAB_UM if mab_uM is None else float(mab_uM)
    tcep_eq = TCEP_CAL if tcep_eq is None else float(tcep_eq)
    feed_eq = float(feed_eq)
    f_cos = cosolvent_vv
    if f_cos is None:
        # the co-solvent comes from this payload's own stock plus everything
        # else fed alongside it, because it is all one pot
        f_cos = cosolvent_fraction(feed_eq if total_feed_eq is None else float(total_feed_eq),
                                   mab_uM=mab_uM, tcep_eq=tcep_eq)
    a = feed_eq * mab_uM * 1e-6
    sigma = float(rec.get("cosolvent_sigma", CAC_SIGMA))
    cac_mix = float(rec["cac_water_M"]) * 10.0 ** (sigma * float(f_cos))
    fav = min(1.0, cac_mix / a) if a > 0 else 1.0
    return fav, dict(status="OK", payload=name, analytical_M=a,
                     cac_in_medium_M=cac_mix, usable_eq=round(feed_eq * fav, 3),
                     cosolvent_vv=round(float(f_cos), 4),
                     f_available=round(fav, 4), saturated=(a > cac_mix),
                     source=rec.get("source"), tier=rec.get("tier", "C"))


def dual_loading_mass_balance(class_capacity, feed_A, feed_B, class_factor=None,
                              k_A=K2, k_ratio=1.0, mab_M=None, t_end_s=None,
                              t_stage1_s=None, avail_A=1.0, avail_B=1.0,
                              nstep=200000, dtau=2.0):
    """Dual-payload loading with BOTH payloads consumed as they react.

    The branching-ratio form (dual_split) treats the two concentrations as
    constant, which is only true while both are in large excess. In practice each
    payload is fed at a few equivalents against 8 thiols, so it runs out:
    a payload fed at 6 eq cannot possibly load more than 6, however fast it is.
    That cap is what makes the fast payload's loading track its FEED almost 1:1
    and leaves the slow one only the leftovers.

    Integrated per site class, because competitiveness differs by class while the
    feed does not (it is one pot, one budget per payload):

        dA_c/dt = k_A * f_c * [mAb] * (fA - sum_c A_c) * (N_c - A_c - B_c)
        dB_c/dt = k_B        * [mAb] * (fB - sum_c B_c) * (N_c - A_c - B_c)

    The shared factor (N_c - A_c - B_c) is the competition: whatever A takes is no
    longer reachable by B. Solid-angle / accessibility effects are already inside
    N_c and inside pc, so they are not double counted.

    t_stage1_s: seconds during which only payload A is present, before B is added.
    This is how published dual-payload ADCs with a controlled ratio are usually
    made, and it is what keeps the second payload's loading small no matter how
    much of it is fed.

    Limits it reproduces:
      k_A == k_B          -> the two stay at their feed ratio (old behaviour)
      k_A >> k_B          -> A takes min(fA, N), B gets the remainder
      fA + fB < N         -> neither fills the antibody; total = fA + fB
    """
    k_A = float(k_A)
    k_B = k_A / float(k_ratio) if k_ratio else k_A
    rkb = k_B / k_A if k_A else 1.0
    mab_M = MAB_UM * 1e-6 if mab_M is None else float(mab_M)
    t_end_s = T_H * 3600.0 if t_end_s is None else float(t_end_s)
    cf = dict(class_factor or {})
    caps = {c: float(v) for c, v in dict(class_capacity).items()}
    fA = float(feed_A) * float(avail_A)
    fB = float(feed_B) * float(avail_B)
    N = float(sum(caps.values()))
    A = {c: 0.0 for c in caps}
    B = {c: 0.0 for c in caps}
    if N <= 0 or (fA <= 0 and fB <= 0):
        out = {c: dict(A=0.0, B=0.0) for c in caps}
        return out, dict(status="EMPTY", n_sites=round(N, 4))
    steps = [0]

    def _advance(dt_s, with_B):
        """Explicit Euler in dimensionless time tau = k_A * [mAb] * t.

        The step is capped so no single update moves a loading by more than 0.05,
        which keeps the integration monotone and stable even when tau is large.
        """
        tau_left = k_A * mab_M * float(dt_s)
        while tau_left > 1e-12 and steps[0] < nstep:
            a_tot, b_tot = sum(A.values()), sum(B.values())
            da, db, tot = {}, {}, 0.0
            for c in caps:
                free = max(caps[c] - A[c] - B[c], 0.0)
                da[c] = float(cf.get(c, 1.0)) * max(fA - a_tot, 0.0) * free
                db[c] = rkb * max(fB - b_tot, 0.0) * free if with_B else 0.0
                tot += da[c] + db[c]
            # stop as soon as nothing more can happen: no free thiols left, or
            # both payloads spent. Without this the loop would grind through the
            # remaining reaction time doing nothing.
            if tot < 1e-9 * (1.0 + N + fA + fB):
                break
            if (a_tot >= fA - 1e-12) and (b_tot >= fB - 1e-12 or not with_B):
                break
            h = min(dtau, tau_left, 0.05 / tot)
            for c in caps:
                A[c] += da[c] * h
                B[c] += db[c] * h
            # a class cannot hold more thiols than it has
            for c in caps:
                s = A[c] + B[c]
                if s > caps[c]:
                    A[c] *= caps[c] / s
                    B[c] *= caps[c] / s
            # ...and a payload cannot load more than was fed, in total
            sa = sum(A.values())
            if sa > fA:
                for c in caps:
                    A[c] *= fA / sa
            sb = sum(B.values())
            if sb > fB:
                for c in caps:
                    B[c] *= fB / sb
            tau_left -= h
            steps[0] += 1

    if t_stage1_s and float(t_stage1_s) > 0:
        t1 = min(float(t_stage1_s), t_end_s)
        _advance(t1, with_B=False)
        a_stage1 = dict(A)
        if t_end_s > t1:
            _advance(t_end_s - t1, with_B=True)
    else:
        a_stage1 = dict(A)
        _advance(t_end_s, with_B=True)

    out = {}
    for c in caps:
        a, b, nc = A[c], B[c], caps[c]
        out[c] = dict(A=round(float(a), 4), B=round(float(b), 4),
                      A_stage1=round(float(a_stage1.get(c, 0.0)), 4),
                      capacity=round(float(nc), 4),
                      occupancy=round(float(a + b) / nc, 4) if nc > 0 else 0.0,
                      frac_B=round(float(b) / (a + b), 4) if a + b > 1e-9 else None)
    tot_a, tot_b = sum(A.values()), sum(B.values())
    return out, dict(status="OK", n_sites=round(N, 4), steps=steps[0],
                     tau_total=round(k_A * mab_M * t_end_s, 2),
                     feed_eff_A=round(fA, 4), feed_eff_B=round(fB, 4),
                     loaded_A=round(float(tot_a), 4), loaded_B=round(float(tot_b), 4),
                     stage1_s=(round(float(t_stage1_s), 2) if t_stage1_s else 0.0),
                     thiol_limited=bool(tot_a + tot_b >= N - 1e-3),
                     feed_limited=bool(tot_a >= fA - 1e-3 or tot_b >= fB - 1e-3))


def dual_split(feed_eq_A, feed_eq_B, k_ratio=1.0):
    """Per-thiol branching ratio q_A for two payloads competing for the same thiols.

        q_A = k_A[A] / (k_A[A] + k_B[B])

    Only two things move this ratio: the rate-constant ratio k_A/k_B, and the
    effective concentration ratio. pH, accessibility and reduction extent are
    common factors to both payloads -> they change TOTAL DAR, not the A:B split.
    """
    a, b = k_ratio * float(feed_eq_A), float(feed_eq_B)
    tot = a + b
    return (a / tot) if tot > 0 else 0.5


def dual_split_by_class(feed_eq_A, feed_eq_B, k_ratio=1.0, class_factor=None):
    """Per-site-class branching ratios: {"LC": q, "HC": q}.

    class_factor[c] multiplies payload A's effective competitiveness *at that
    class of site only*. It collects everything that is not bulk concentration:
    how well each payload physically reaches this kind of site (size, charge,
    hydrophobicity against local crowding). factor 1.0 everywhere reduces to
    dual_split() and therefore to the plain feed ratio.

    Only the coarse HC/LC split is resolvable from the available data. Within the
    heavy chain, hinge sites and the light-chain-paired site are known to behave
    differently, but no measurement can separate them yet.
    """
    cf = dict(class_factor or {})
    out = {}
    for cls in ("LC", "HC"):
        a = float(k_ratio) * float(feed_eq_A) * float(cf.get(cls, 1.0))
        b = float(feed_eq_B)
        tot = a + b
        out[cls] = (a / tot) if tot > 0 else 0.5
    return out


def class_factor_from_descriptors(mw_A, mw_B, gamma0=None, crowding=None,
                                  logp_A=None, logp_B=None, logp_weight=0.0):
    """Predict the per-site-class competitiveness factor from the two payloads' size.

    This is the step that turns class_factor from "a number you must measure for
    every new payload pair" into "a number you can compute". The measured form is

        ln f_class = gamma0 * ln(MW_B / MW_A) * crowding[class]

    with crowding[class] describing how cramped that kind of site is, and
    gamma0 the strength of steric discrimination per unit size difference.
    Both are shared across payload pairs — that is what makes them portable;
    the only thing a new pair contributes is its own size ratio.

    CALIBRATION (DL003 / DL012, MediLink WO2024235127A1, reduced-chain MS):
        MW ratio 1673 / 1348 = 1.241   ->   ln ratio = 0.216
        f_HC = 2.59, f_LC = 1.01
        taking the heavy-chain sites as the crowding reference (C_HC = 1):
            gamma0 = ln(2.59)/0.216 = 4.41
            C_LC   = ln(1.01)/ln(2.59) = 0.0105
    Reading: on the light-chain site the two payloads are ~100x less
    discriminated by size than on the heavy-chain sites. That is consistent with
    Cys214 sitting on an exposed, flexible C-terminal tail while the hinge and
    Fab-arm cysteines sit in a crowded crevice.

    ZERO DEGREES OF FREEDOM. One payload pair fixes two constants, so this
    reproduces that pair by construction and predicts nothing testable until a
    second pair with a DIFFERENT size ratio is measured. TIER C.
    """
    gamma0 = CLASS_GAMMA0 if gamma0 is None else float(gamma0)
    crowd = dict(CLASS_CROWDING if crowding is None else crowding)
    dsize = math.log(float(mw_B) / float(mw_A)) if mw_A and mw_B else 0.0
    if logp_weight and logp_A is not None and logp_B is not None:
        dsize = dsize + float(logp_weight) * (float(logp_B) - float(logp_A))
    out = {}
    for cls in ("LC", "HC"):
        out[cls] = round(float(math.exp(gamma0 * dsize * float(crowd.get(cls, 1.0)))), 4)
    return out, dict(gamma0=gamma0, crowding=crowd, ln_size_ratio=round(dsize, 4),
                     mw_A=float(mw_A), mw_B=float(mw_B),
                     tier="C — one payload pair, zero degrees of freedom")


def fit_class_factor_from_fractions(obs):
    """Recover the relative HC/LC preference from measured per-class B fractions.

    obs: [(experiment_id, feed_A, feed_B, class, observed_B_fraction)]

    Only the RATIO f_HC/f_LC is identifiable from a raw feed ratio: the overall
    level also depends on how much of each payload was actually available in that
    particular run. If the per-experiment ratios scatter, the experiments are
    moving the balance by something this model does not yet carry; that scatter
    is reported as `per_example_ratio` / `residual_spread_ln` rather than being
    averaged away silently.
    """
    rows, by_ex = [], {}
    for exp, fa, fb, cls, frac in obs:
        frac = min(max(float(frac), 1e-6), 1 - 1e-6)
        # q_A = kA*f*A / (kA*f*A + B) and frac_B = 1 - q_A = B / (kA*f*A + B)
        #   =>  f = (B / (k*A)) * (1 - frac_B) / frac_B
        # so this is how much more competitive A really is here, on top of the
        # feed ratio. f > 1 means A wins more often than it was fed.
        f = float(fb) / float(fa) * (1.0 - frac) / frac
        rows.append((cls, math.log(f)))
        by_ex.setdefault(exp, {})[cls] = math.log(f)
    lc = [v for c, v in rows if c == "LC"]
    hc = [v for c, v in rows if c == "HC"]
    if not lc or not hc:
        return dict(status="SKIPPED", reason="need both LC and HC observations")
    lo = float(np.mean(hc)) - float(np.mean(lc))          # ln(f_HC / f_LC)
    diffs = [d["HC"] - d["LC"] for d in by_ex.values() if "LC" in d and "HC" in d]
    return dict(status="OK",
                class_factor={"LC": 1.0, "HC": round(float(math.exp(lo)), 4)},
                preference_ratio_HC_over_LC=round(float(math.exp(lo)), 4),
                per_example_ratio=[round(float(math.exp(x)), 3) for x in diffs],
                residual_spread_ln=round(float(np.std(diffs)), 4) if diffs else None,
                note=("residual_spread_ln is the scatter that remains AFTER taking "
                      "out the class effect. Non-zero means another factor "
                      "(payload availability, mixing, MS quantitation) is at work."))


# =========================================================================
# Conjugation-rate estimation for a payload nobody has measured yet
# -------------------------------------------------------------------------
# Everything below answers one question: a new small molecule arrives, what
# rate should step 5 use — and what should NOT be put in a rate.
# =========================================================================

WARHEAD_OF_K2_REF = "N-substituted maleimide"   # what K2_REF=300 M^-1 s^-1 belongs to


def estimate_k2(warhead=WARHEAD_OF_K2_REF, mayr_E=None, mayr_E_ref=None, s_N=None,
                pka=PKA_EFF, ph=PH, k2_ref=K2_REF, ph_ref=PH_REF):
    """Baseline conjugation rate for a new payload, from the warhead it carries.

    THE QUESTION
    ------------
    "This molecule has never been conjugated before. Before running anything,
    what second-order rate should the model use?"

    FIRST, THE THING THAT DECIDES 90 % OF CASES
    -------------------------------------------
    In thiol conjugation the reacting centre is the WARHEAD, and in essentially
    every ADC that warhead sits on the LINKER, not on the payload. Swap DXd for
    a TLR7/8 agonist and the maleimide doing the reacting is literally the same
    bond. Therefore, for two payloads carrying the same warhead, the honest
    answer is

        k_A / k_B = 1.0

    regardless of how different the two molecules look elsewhere. That is why
    the config default is 1.0. A hand-typed 12 was once used instead; nothing
    supported it and it made the fit worse (7.600 -> 7.585).

    What genuinely DIFFERS between two payloads — and is already modelled, so
    putting it here too would count it twice:
        size / steric reach  -> class_factor   (class_factor_from_descriptors)
        self-aggregation     -> availability   (payload_availability, CAC)
    Neither of those is a rate constant. Do not fold them into k2.

    WHEN THE WARHEAD REALLY IS DIFFERENT
    ------------------------------------
    Use Mayr's linear free-energy relationship, which is built for exactly this
    "rate I have not measured" situation:

        log10 k(20 C) = s_N * (N + E)

    with the same nucleophile (here: the protein thiolate) on both sides:

        log10(k_a / k_b) = s_N * (E_a - E_b)

    E and s_N come from Mayr's free database of reactivity parameters
    (https://www.cup.lmu.de/oc/mayr/DBintro.html ; ~352 electrophiles E,
    ~1251 nucleophiles N/s_N for 2023 — see the "computational road to
    reactivity scales" review, PCCP 2023, DOI 10.1039/D2CP03937K).
    Thiolate nucleophile parameters specifically are established in
    "Nucleophilicity of Glutathione: A Link to Michael Acceptor Reactivities"
    (PMID 31560405), which is what makes this route applicable to thiols.

    Honest accuracy, quoted from Mayr's own database notes: within a factor of
    ~2 for reactions against REFERENCE partners, and a factor of 10-100 when
    the two partners are outside the reference sets. The equation is also
    explicitly free of steric effects. So this gives a defensible ORDER and
    a rough magnitude, never a precise number. TIER C.

    HOW MUCH LEVERAGE THIS ACTUALLY HAS
    -----------------------------------
    Very little in absolute terms at reference conditions: p_c = 1-exp(-k2*AUC)
    is already pinned at p_conj = 0.995 for every site (measured cp50 values are
    0.50-1.14 uM against feeds of hundreds of uM). Doubling k2 changes nothing
    there. What k2 DOES move is (a) the A:B split when both payloads compete —
    and only the RATIO enters — and (b) behaviour at very short times or very
    low effective concentration, i.e. when availability has cut the payload down
    to near its cp50.
    """
    same_warhead = str(warhead).strip().lower() in (
        "maleimide", "n-substituted maleimide", WARHEAD_OF_K2_REF.lower())
    out = dict(warhead=warhead, k2_ref_M_s=float(k2_ref), ph_ref=float(ph_ref),
               reference_warhead=WARHEAD_OF_K2_REF,
               k2_ref_source=("apparent rate at pH 7 for N-substituted maleimide + "
                              "thiol; see k2_apparent docstring (300 M^-1 s^-1)"))
    rel = 1.0
    if same_warhead:
        out.update(status="OK", basis="same warhead as the reference",
                   relative_to_reference=1.0,
                   note=("Rate is the reference rate exactly. Payload-to-payload "
                         "differences belong in class_factor (size) and availability "
                         "(CAC), never here."))
    elif mayr_E is None or mayr_E_ref is None:
        return dict(status="NEED_INPUT", warhead=warhead, tier="C",
                    reason=("warhead differs from the reference, so the relative rate "
                            "must come from Mayr electrophilicity parameters. Supply "
                            "mayr_E and mayr_E_ref."),
                    how_to=("look E up for your electrophile AND for N-substituted "
                            "maleimide at https://www.cup.lmu.de/oc/mayr/DBintro.html"))
    elif s_N is None:
        return dict(status="NEED_INPUT", warhead=warhead, tier="C",
                    reason=("reactivity ratio is s_N*(E - E_ref); s_N belongs to the "
                            "nucleophile and is not defaulted to a guessed value. "
                            "Read it off the same database entry as N."),
                    how_to="supply s_N for the thiolate nucleophile you are using")
    else:
        rel = 10.0 ** (float(s_N) * (float(mayr_E) - float(mayr_E_ref)))
        out.update(status="OK", basis="Mayr equation, relative to N-substituted maleimide",
                   relative_to_reference=round(rel, 4),
                   mayr=dict(E=float(mayr_E), E_ref=float(mayr_E_ref), s_N=float(s_N),
                             log10_ratio=round(float(s_N) * (float(mayr_E) - float(mayr_E_ref)), 4)),
                   tier="C — semiquantitative: factor ~2 within reference sets, 10-100 outside; no steric term",
                   source="Mayr database + PMID 31560405 (thiolate nucleophilicity)")
    out["k2_apparent_M_s"] = round(float(k2_apparent(pka, ph=ph, ph_ref=ph_ref,
                                                     k2_ref=k2_ref * rel)), 4)
    return out


def solve_dual_feed_for_split(target_A, target_B, class_capacity, total_feed_eq=None,
                              k_ratio=1.0, class_factor=None, avail_A=1.0, avail_B=1.0,
                              k_A=K2, tol=1e-4, itmax=80):
    """Feed ratio that lands a wanted A:B split, given the thiols actually available.

    This is the second half of the inverse problem, and it is SEPARABLE from the
    first: how many thiols exist is decided by TCEP (solve_tcep_eq_for_target_dar).
    How those thiols are divided between two payloads is decided here, by the feed
    ratio — the total neither helps nor hurts once it exceeds the thiol count.

    Monotone in the feed split (more of A fed -> more of A loaded), so a plain
    bisection on x = feed_A / (feed_A + feed_B) converges. Uses the same
    mass-balance integrator as the forward run, so consumption of each payload
    and the site-class steric factor are both in play.
    """
    caps = {c: float(v) for c, v in dict(class_capacity).items()}
    n_thiol = float(sum(caps.values()))
    if total_feed_eq is None:
        total_feed_eq = max(float(target_A), 0.0) + max(float(target_B), 0.0) + 2.0
    total_feed_eq = float(total_feed_eq)
    want = float(target_A) / (float(target_A) + float(target_B)) if (target_A + target_B) else 0.5

    def achieved(x):
        fa = total_feed_eq * x
        fb = total_feed_eq * (1.0 - x)
        mb, info = dual_loading_mass_balance(
            caps, fa, fb, class_factor=class_factor, k_ratio=k_ratio,
            k_A=k_A, avail_A=avail_A, avail_B=avail_B)
        a = sum(v["A"] for v in mb.values())
        b = sum(v["B"] for v in mb.values())
        frac = a / (a + b) if (a + b) > 1e-12 else 0.5
        return frac, a, b, info

    lo, hi = 1e-6, 1.0 - 1e-6
    flo, *_ = achieved(lo)
    fhi, *_ = achieved(hi)
    if (flo - want) * (fhi - want) > 0:
        return dict(status="UNREACHABLE", want_A_fraction=round(want, 4),
                    achievable_A_fraction=[round(flo, 4), round(fhi, 4)],
                    reason=("even feeding one payload exclusively cannot reach the wanted "
                            "share — check target_A/target_B against the available thiols "
                            "and the availability fractions"))
    for _ in range(itmax):
        mid = 0.5 * (lo + hi)
        fmid, *_ = achieved(mid)
        if (fmid - want) * (flo - want) <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
        if abs(hi - lo) < tol:
            break
    x = 0.5 * (lo + hi)
    frac, a, b, info = achieved(x)
    return dict(status="OK", feed_A_eq=round(total_feed_eq * x, 4),
                feed_B_eq=round(total_feed_eq * (1.0 - x), 4),
                feed_split_A=round(float(x), 4), total_feed_eq=total_feed_eq,
                target_A=float(target_A), target_B=float(target_B),
                predicted_A=round(float(a), 4), predicted_B=round(float(b), 4),
                predicted_total=round(float(a + b), 4),
                n_thiols_available=round(n_thiol, 4),
                mass_balance=info, tier="B for the direction, C for the exact number")


def dual_payload_dist(t_eff_by_site, pred, feed_eq_A, feed_eq_B, k_ratio=1.0,
                      pc_fallback=None, pen=1.0, class_factor=None,
                      avail_A=1.0, avail_B=1.0,
                      stage1_h=None, stage2_h=None, sequential=False,
                      mass_balance=True, k2_A=None, k2_B=None):
    """Dual-payload run: returns joint P(n_A, n_B), its summary and the per-site inputs.

    Per-thiol branch into A vs B; each of the 8 sites ends as A / B / nothing, and the
    joint distribution is a 2-D convolution because a thiol taken by A cannot also
    carry B (n_A and n_B are NOT independent).

    class_factor  per site class (see dual_split_by_class). Omitting it reproduces
                  the old behaviour exactly: one odds for all 8 sites.
    avail_A/B     fraction of each payload that is actually monomeric in the
                  reaction medium (see payload_availability). Scales the
                  concentration that enters the competition, not the feed.
    sequential    True  -> A is added first and reacts alone for stage1_h, then B
                  is added and the two compete for whatever thiols are LEFT for
                  stage2_h. This is how most published dual-payload ADCs with a
                  controlled ratio are actually made, and it is what makes the
                  second payload's loading small no matter how much is fed.
                  False -> both present from t=0 (default, old behaviour).
    mass_balance   True  -> the two payloads are consumed as they react, so each
                  one's loading is capped by what was fed and the fast one cannot
                  simply take everything (see dual_loading_mass_balance).
                  False -> concentrations held constant: the split stays at the
                  branching odds q (the original form, kept for comparison).
    """
    if k2_A is not None and k2_B is not None and float(k2_B) > 0:
        k_ratio = float(k2_A) / float(k2_B)
    a_eff, b_eff = float(feed_eq_A) * float(avail_A), float(feed_eq_B) * float(avail_B)
    q_by = dual_split_by_class(a_eff, b_eff, k_ratio, class_factor)
    q_uniform = dual_split(a_eff, b_eff, k_ratio)
    tot_feed = float(feed_eq_A) + float(feed_eq_B)
    # 1. per-site probability of being conjugated at all (unchanged from single payload)
    pc_tot = {}
    all_sites = sorted({s for p, a, b in PAIRS for s in (a, b)})
    for s in all_sites:
        v = (pc_fallback or {}).get(s)
        te = t_eff_by_site.get(s)
        if te and te > 0:
            v, _, _ = pc_from_feed(te, tot_feed)
        pc_tot[s] = v if v is not None else 0.0
    # 2. how many thiols each site class actually offers. Reduction extent (pred),
    #    exposure (pc) and the steric penalty are all already in here, so the
    #    competition below must not put them in a second time.
    cap = {}
    for p, a, b in PAIRS:
        for s in (a, b):
            c = site_class(s)[0]
            cap[c] = cap.get(c, 0.0) + float(pred.get(p, 0.0)) * pc_tot[s] * pen
    # 3. split those thiols between the two payloads
    if mass_balance:
        seq = bool(sequential and stage1_h)
        t_stage1_s = float(stage1_h) * 3600.0 if seq else None
        t_total_s = (float(stage1_h) + float(stage2_h or 0.0)) * 3600.0 if seq \
            else T_H * 3600.0
        mb, mb_info = dual_loading_mass_balance(
            cap, feed_eq_A, feed_eq_B, class_factor=class_factor,
            k_A=(K2 if k2_A is None else float(k2_A)), k_ratio=k_ratio,
            t_end_s=t_total_s, t_stage1_s=t_stage1_s,
            avail_A=avail_A, avail_B=avail_B)
        split = {}
        for c in cap:
            a_c, b_c = mb[c]["A"], mb[c]["B"]
            tot_c = a_c + b_c
            occ = min(1.0, tot_c / cap[c]) if cap[c] > 1e-12 else 0.0
            split[c] = (occ, (a_c / tot_c) if tot_c > 1e-9 else q_by[c])
    else:
        mb, mb_info = {}, dict(status="DISABLED")
        split = {c: (1.0, q_by[c]) for c in cap}
    pcA, pcB, stage1_fill = {}, {}, {}
    for s in all_sites:
        c = site_class(s)[0]
        occ, qa = split[c]
        pcA[s] = pc_tot[s] * occ * qa
        pcB[s] = pc_tot[s] * occ * (1.0 - qa)
        if mb:
            a_c, b_c = mb[c]["A"], mb[c]["B"]
            tot_c = a_c + b_c
            f1 = (mb[c]["A_stage1"] / tot_c) if tot_c > 1e-9 else 0.0
            stage1_fill[s] = round(float(pc_tot[s] * occ * f1), 4)
    poly = full_dist_dual(pcA, pcB, pred, pen=pen)
    # Expected per-class loading. This is the only form the output can take that
    # is directly comparable with reduced-LC/MS data, which reports light chain
    # and heavy chain separately.
    by_class = {}
    for cls in ("LC", "HC"):
        sites = [s for p, a, b in PAIRS for s in (a, b) if site_class(s)[0] == cls]
        ea = sum(pred[p] * pcA[s] * pen for p, a, b in PAIRS for s in (a, b)
                 if site_class(s)[0] == cls)
        eb = sum(pred[p] * pcB[s] * pen for p, a, b in PAIRS for s in (a, b)
                 if site_class(s)[0] == cls)
        by_class[cls] = dict(n_sites=len(sites), dar_A=round(float(ea), 3),
                             dar_B=round(float(eb), 3),
                             frac_B=round(float(eb) / (ea + eb), 4) if ea + eb > 0 else None)
    return poly, summarize_dual(poly), dict(q_A=round(q_uniform, 4),
                                            q_by_class={k: round(v, 4) for k, v in q_by.items()},
                                            class_factor=dict(class_factor or {}),
                                            availability=dict(A=round(float(avail_A), 4),
                                                             B=round(float(avail_B), 4),
                                                             feed_eff_A=round(a_eff, 4),
                                                             feed_eff_B=round(b_eff, 4)),
                                            mode=("sequential" if (sequential and stage1_h)
                                                  else "simultaneous"),
                                            mass_balance=bool(mass_balance),
                                            mass_balance_info=mb_info,
                                            class_capacity={c: round(v, 4) for c, v in cap.items()},
                                            stage1_fill_by_site=stage1_fill,
                                            by_class=by_class,
                                            total_feed_eq=tot_feed,
                                            pc_total={s: round(v, 4) for s, v in pc_tot.items()})


# =========================================================================
# Glycan conjugation branch (second chemistry, not a variant of the first)
# -------------------------------------------------------------------------
# Sites:   the conserved Fc N-glycan at Asn297 (EU numbering), one per heavy
#          chain -> 2 sites. Found from sequence by seqqc.find_glyco_sites()
#          (N-X-S/T, X != P), which already reports the Fc sites separately.
# No reduction step: nothing has to be opened, so TCEP / K / temperature-drop
#   into this branch not at all.
# Attachment is three sequential stages, each with its own yield:
#     1 trim      endoglycosidase cuts the glycan back to core GlcNAc
#     2 transfer  glycosyl transferase tags it with an azidosugar
#     3 click     metal-free click (SPAAC) attaches the linker-drug
#   p_site = eff_trim x eff_transfer x eff_click   (staged form)
#   p_site = eff_total                             (lumped form, anchored today)
# Output: the payload count per site is 0..payloads_per_site, so the DAR
#   distribution is that per-site law convolved over the sites.
# =========================================================================
def glycan_site_prob(cfg=None):
    """Per-site attachment probability, and which form produced it."""
    g = dict(GLYCAN if cfg is None else cfg)
    mode = str(g.get("efficiency_mode", "total")).lower()
    if mode == "staged":
        keys = ("eff_trim", "eff_transfer", "eff_click")
        vals = [g.get(k) for k in keys]
        if any(v is None for v in vals):
            raise SystemExit("[ERROR] efficiency_mode=staged needs eff_trim, "
                             "eff_transfer and eff_click in glycan_conjugation")
        vals = [float(v) for v in vals]
        p = 1.0
        for v in vals:
            p *= v
        info = dict(mode="staged", eff_trim=vals[0], eff_transfer=vals[1],
                    eff_click=vals[2], p_site=round(p, 4),
                    note="three stages multiplied; each needs its own measurement")
        return min(max(p, 0.0), 1.0), info
    p = float(g.get("eff_total", 1.0))
    info = dict(mode="total", eff_total=round(p, 4), p_site=round(p, 4),
                note=("one lumped number; the staged split is only honest once "
                      "each stage is measured on its own"))
    return min(max(p, 0.0), 1.0), info


def glycan_dar_dist(n_sites=2, payloads_per_site=1, p_site=1.0):
    """DAR distribution: per-site law convolved over the glycan sites.

    One site carries k = payloads_per_site attachment slots; each slot is filled
    independently with probability p_site (binomial), and the sites themselves
    are independent, so the total is the k-slot law convolved n_sites times.
    """
    k = max(int(payloads_per_site), 1)
    n = max(int(n_sites), 0)
    p = min(max(float(p_site), 0.0), 1.0)
    site = np.array([math.comb(k, j) * (p ** j) * ((1.0 - p) ** (k - j))
                     for j in range(k + 1)])
    dist = np.array([1.0])
    for _ in range(n):
        dist = np.convolve(dist, site)
    return dist


def glycan_summary(dist):
    idx = np.arange(len(dist))
    mean = float((idx * dist).sum())
    return dict(dar_values=[int(i) for i in idx],
                p=[round(float(x), 4) for x in dist],
                mean_dar=round(mean, 4),
                mode_dar=int(idx[int(np.argmax(dist))]),
                sd_dar=round(float(np.sqrt(((idx - mean) ** 2 * dist).sum())), 4),
                p_empty=round(float(dist[0]), 4),
                p_full=round(float(dist[-1]), 4),
                max_dar=int(len(dist) - 1))


def glycan_inverse_eff(target_mean_dar, n_sites=2, payloads_per_site=1):
    """Per-site efficiency that would be needed to land on a target mean DAR.

    mean DAR = n_sites x payloads_per_site x p, so this is a straight line and
    needs no search. It exists so a measured DAR can be turned into the only
    number this branch actually carries.
    """
    cap = float(max(int(n_sites), 0)) * float(max(int(payloads_per_site), 1))
    if cap <= 0:
        return None
    p = float(target_mean_dar) / cap
    return dict(target_mean_dar=float(target_mean_dar),
                capacity=cap, eff_needed=round(min(max(p, 0.0), 1.0), 4),
                reachable=bool(0.0 <= p <= 1.0))


def glycan_staged_split(p_total, n_stage=3):
    """What splitting one efficiency into three would imply, if equal.

    A look at the cost of the split, not a prediction: with three equal stages
    each must be p^(1/3), which is always HIGHER than the lumped number, so any
    stage that is actually worse than that has to be paid for by the other two.
    """
    p = min(max(float(p_total), 1e-9), 1.0)
    per = p ** (1.0 / float(n_stage))
    rows = []
    for worse in (0.99, 0.95, 0.90, 0.80):
        # one stage drops to `worse`; what do the other two have to be?
        rest = (p / worse) ** 0.5 if worse > 0 else None
        rows.append(dict(one_stage_at=round(worse, 3),
                         other_two_each=round(float(rest), 4) if rest else None,
                         feasible=bool(rest is not None and rest <= 1.0)))
    return dict(equal_split_each=round(float(per), 4), n_stage=n_stage,
                if_one_stage_is_worse=rows,
                note=("equal split is the optimistic case; a single bad stage "
                      "forces the others above what they would otherwise need"))


def process_window_table(target_dars, ease, K, pc, eta=ETA,
                         lp_overage_margin=0.5, ph=PH):
    """Process-window table for wet-lab scoping: per target DAR give the TCEP eq
    needed, the minimum LP eq, the predicted actual mean DAR, and two quality
    metrics — P(DAR>=6) [high-DAR tail] and P(DAR<=1) [under-conjugated tail]."""
    rows = []
    for td in target_dars:
        try:
            inv = min_lp_eq_for_target_dar(td, ease, K, pc, eta, lp_overage_margin)
            eq = inv["tcep_eq"]
            s = summarize(full_dist(pc, pred_shared(eq, ease, K, eta)))
            p_le1 = float(np.sum(s["dist"][0:2]))      # DAR 0 and 1
            rows.append(dict(target_dar=td, reachable=True,
                             tcep_eq=round(eq, 4),
                             released_thiols_eq=round(inv["released_thiols_eq"], 3),
                             lp_min_eq=round(inv["lp_min_eq"], 3),
                             lp_overage_margin=inv["lp_overage_margin"],
                             pred_mean_dar=s["mean_dar"],
                             pred_mode_dar=s["mode_dar"],
                             even_odd_ratio=s["even_odd_ratio"],
                             p_dar_ge6=s["p_dar_ge6"],
                             p_dar_le1=round(p_le1, 4)))
        except ValueError as e:
            rows.append(dict(target_dar=td, reachable=False, error=str(e)))
    return rows


# =========================================================================
def run_glycan_mode(out_path=None, tag=None, verbose=True):
    """Glycan-conjugation run. Separate chemistry, so it prints its own block.

    Order of authority for the one number it needs:
      1. a measured mean DAR (glycan_conjugation.anchor_mean_dar)  -> back-solve p
      2. staged efficiencies, if all three are given                -> multiply
      3. eff_total                                                  -> placeholder
    Without 1 or 2 the branch still runs, but it answers "what shape is the
    distribution", not "what DAR will I get", and says so.
    """
    g = dict(GLYCAN)
    n_sites = int(g.get("n_sites", 2))
    per_site = int(g.get("payloads_per_site", 1))
    p, pinfo = glycan_site_prob(g)
    anchor = g.get("anchor_mean_dar")
    inv = None
    if anchor is not None:
        inv = glycan_inverse_eff(float(anchor), n_sites, per_site)
        if inv and inv["reachable"]:
            p = inv["eff_needed"]
            pinfo = dict(p_site=round(p, 4), mode="from_anchor",
                         anchor_mean_dar=float(anchor),
                         anchor_source=g.get("anchor_source"),
                         note="efficiency back-solved from a measured mean DAR")
    dist = glycan_dar_dist(n_sites, per_site, p)
    s = glycan_summary(dist)
    tier = "B" if anchor is not None else "C"
    if verbose:
        print("\n" + "=" * 92)
        print("[glycan conjugation] N-glycan at Asn297 (EU numbering), "
              f"{n_sites} sites x {per_site} payload slot(s)")
        print(f"  site rule = {g.get('site_rule')}   "
              f"max DAR = {s['max_dar']}   "
              f"({'DAR 2 linear' if per_site == 1 else 'branched'} construct)")
        print("  no disulfide reduction step: nothing has to be opened, so TCEP, "
              "K and reduction temperature do not enter")
        print(f"  per-site attachment probability = {p:.4f}   {pinfo}")
        print(f"  {'DAR':>4} : " + "  ".join(f"{i:>6}" for i in s["dar_values"]))
        print(f"  {'P':>4}   : " + "  ".join(f"{x:>6.3f}" for x in s["p"]))
        print(f"  mean DAR = {s['mean_dar']:.3f}   mode = {s['mode_dar']}   "
              f"sd = {s['sd_dar']:.3f}   P(empty) = {s['p_empty']:.3f}   "
              f"P(full) = {s['p_full']:.3f}")
        if inv:
            print(f"  anchor {anchor} -> efficiency needed "
                  f"{inv['eff_needed']:.4f} against a capacity of {inv['capacity']:.0f}")
        if tier == "C":
            print("  TIER C: eff_total is a placeholder. Set anchor_mean_dar (with "
                  "anchor_source) or the three staged efficiencies before quoting "
                  "any number from this branch.")
        sp = glycan_staged_split(p)
        print(f"  if the three stages were equal, each would be "
              f"{sp['equal_split_each']:.4f}; if one stage is only 0.90 the other "
              f"two must each reach "
              f"{[r for r in sp['if_one_stage_is_worse'] if r['one_stage_at'] == 0.90][0]['other_two_each']}")
        print("=" * 92)
    out = dict(run_tag=tag, chemistry="glycan (N297, enzymatic remodel + click)",
               n_sites=n_sites, payloads_per_site=per_site,
               p_site=round(float(p), 4), p_site_info=pinfo,
               inverse=inv, staged_split=glycan_staged_split(p),
               tier=tier, **s)
    path = out_path or os.path.join(RES, "dar_glycan.json")
    try:
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        if verbose:
            print(f"  written to {path}")
    except OSError as e:                                     # noqa: BLE001
        print(f"[warn] could not write {path}: {e}", file=sys.stderr)
    return out


def load_frames():
    """Load v5 measured per-frame SASA; return None if missing"""
    out = {}
    for site, fn in MEASURED.items():
        p = os.path.join(RES, fn)
        if os.path.exists(p):
            a = np.load(p)
            out[site] = np.asarray(a, float).ravel()
    return out


def main(profile_path=None, pc_override=None, out_path=None, tag=None,
         extra_targets=(), dual=None, reduction_temp_C=None, conjugation_temp_C=None,
         mab_uM=None, tcep_eq=None, reduction_time_h=None, payload_feed=None):
    """Run the whole pipeline.

    Every input is overridable, so the same code can be pointed at a different
    input structure without editing anything:

      profile_path : reduction-ease profile json (default: the Boltz-derived one)
      pc_override  : {site: p_c} that replaces the MD-derived conjugation
                     probability — used when a structure has no MD yet and only
                     a static-SASA estimate exists
      out_path     : where the result json is written
      tag          : short label recorded in the output, so two runs can be told
                     apart later
      extra_targets: additional target mean DARs for the inverse-solve table
                     (e.g. 7.7 for the DXd / DS-8201a case)
      dual         : {payload_A, payload_B, feed_eq_A, feed_eq_B, k_ratio_A_over_B}
                     enables the dual-payload joint-distribution run
      mab_uM / tcep_eq / reduction_time_h / payload_feed :
                     per-run process overrides (e.g. to reproduce a literature or
                     patent protocol without editing the yaml). K = k_TCEP x [mAb] x t
                     uses the overridden values, so a protocol with a different
                     antibody concentration or reduction time is handled correctly.
    """
    # ---- Glycan conjugation: different chemistry, handled by its own branch ----
    # It has no reduction step and different sites, so none of the cysteine
    # machinery below applies to it. Bail out before any of it prints, or the
    # output would show a cysteine run followed by a glycan run.
    if bool(GLYCAN.get("enabled")):
        run_glycan_mode(out_path=out_path, tag=tag)
        return None

    global T_RED_C, T_CONJ_C, MAB_UM, TCEP_CAL, T_RED_H, R_FEED
    # Reset to the config defaults FIRST. main() applies per-run overrides by
    # rebinding these module globals, so without this reset a second run in the
    # same Python process would inherit the previous run's temperature /
    # concentration / TCEP / feed / durations. Resetting keeps the default
    # behaviour identical to a fresh interpreter and makes repeat calls
    # independent (required before packaging main() as a callable API).
    _reset_process_globals()
    if reduction_temp_C is not None:
        T_RED_C = float(reduction_temp_C)
    if conjugation_temp_C is not None:
        T_CONJ_C = float(conjugation_temp_C)
    if mab_uM is not None:
        MAB_UM = float(mab_uM)
    if tcep_eq is not None:
        TCEP_CAL = float(tcep_eq)
    if reduction_time_h is not None:
        T_RED_H = float(reduction_time_h)
    if payload_feed is not None:
        R_FEED = float(payload_feed)

    prof_path = profile_path or os.path.join(RES, "ss_reduction_profile.json")
    prof = json.load(open(prof_path))
    # A profile from another antibody labels its bonds with that antibody's own
    # residue numbers. Both IgG1 profiles hold the same set: 2 HL-interface and
    # 2 HH-hinge bonds. Sort each set by (kind, first site) and match by index,
    # so the model keeps its internal labels while the numbers come from the
    # profile. When the labels already match, this is the identity mapping.
    _prof_sorted = sorted(prof["pairs"], key=lambda p: (p["kind"], p["site_a"]))
    _model_sorted = sorted(PAIRS, key=lambda p: (p[1].split("-")[0] + p[1], p[0]))
    _relabel = {}
    if len(_prof_sorted) == len(_model_sorted):
        for pp, mp in zip(_prof_sorted, _model_sorted):
            _relabel[pp["site_a"] + "--" + pp["site_b"]] = mp[0]
    ease = {_relabel.get(p["site_a"] + "--" + p["site_b"],
                         p["site_a"] + "--" + p["site_b"]): p["reduction_ease"]
            for p in prof["pairs"]}
    # Local electrostatics per bond (reactant side): TCEP is an anion at process
    # pH, a net-negative patch near the bond repels it and slows the attack.
    # The profile stores the signed sum of side-chain charges within 8 A of
    # each SG. Missing field (old profiles) falls back to 0.0 = no correction.
    q_local = {_relabel.get(p["site_a"] + "--" + p["site_b"],
                            p["site_a"] + "--" + p["site_b"]):
               float(p.get("local_net_charge", 0.0)) for p in prof["pairs"]}

    print("=" * 92)
    print("DAR v6 — thiol activation integrated + hydrolysis competition + uncertainty propagation + multipoint calibration framework")
    print("=" * 92)
    print(f"[input] reduction profile : {os.path.basename(prof_path)}")
    if tag:
        print(f"[input] run tag           : {tag}")
    if pc_override:
        print(f"[input] p_c override      : {len(pc_override)} sites from static estimate "
              f"(no MD for this structure) -> {sorted(pc_override)}")

    frames = load_frames()
    print(f"\n[v5 MD measured per-frame SASA] loaded {len(frames)} sites: {sorted(frames)}")

    # ---- temperature first: it changes K, so it cannot be an afterthought ----
    t_red = require_reduction_temp()
    t_conj = T_CONJ_C
    print(f"\n[temperature] reduction {t_red:g} C (required input) | "
          f"conjugation {t_conj:g} C (default 22)" +
          ("" if EA_K2 else "  — no Ea set for conjugation, k2 left at its 22 C value"))

    # ---- Per-site computation of t_eff and p_c ----
    # Coupling corrections (2026-09-29): volume exclusion shared with reduction;
    # per-bond charge from profile, mapped to sites.
    _f_vol_couple = volume_exclusion_factor(MAB_UM) / volume_exclusion_factor(ANCHOR_MAB_UM)
    _bond_to_sites = {p[0]: (p[1], p[2]) for p in PAIRS}
    _site_chg = {}
    for bn, (sa, sb) in _bond_to_sites.items():
        cf = charge_factor(q_local.get(bn, 0.0))
        _site_chg[sa] = cf
        _site_chg[sb] = cf
    pc, detail, t_eff_by_site = {}, {}, {}
    for p, a, b in PAIRS:
        for s in (a, b):
            if s in pc:
                continue
            src = s if s in frames else MIRROR_OF.get(s)
            if src is None or src not in frames:
                pc[s], note = 0.0, "NO DATA"
            else:
                te = t_eff_from_frames(frames[src])
                v, ft, auc, raw = pc_from_teff(te, chg_factor=_site_chg.get(s, 1.0),
                                                f_vol=_f_vol_couple, t_conj=t_conj)
                pc[s] = v
                t_eff_by_site[s] = te
                note = ("v5 MD measured" if s in frames else f"extrapolated from {src} (no MD at this site)")
                detail[s] = dict(source=src, t_eff_s=round(te, 1),
                                 f_thiolate=round(ft, 5), auc_M_s=round(auc, 6),
                                 raw_conversion=round(raw, 4), p_c=round(v, 4))
            if s not in detail:
                detail[s] = dict(source=note, p_c=round(pc[s], 4))

    # ---- Optional: per-site steric discount from payload_qc.steric_discount ----
    # In bench terms: a payload that is bulkier or less flexible than vc-MMAE
    # (the reference, which reached DAR 8) will couple to fewer thiols in the
    # same time, because it simply does not fit as often. The discount is the
    # ratio of "frames where it clears the cut-off" vs the same fraction for
    # vc-MMAE on the same MD snapshots. 1.0 = no penalty; <1 = harder to
    # conjugate. Leave null in config to skip (equivalent to 1.0 everywhere).
    _STERIC_DISC = (CFG.get("site_accessibility", {}) or {}).get("steric_discount")
    if _STERIC_DISC and isinstance(_STERIC_DISC, dict):
        for s in pc:
            d = _STERIC_DISC.get(s, 1.0)
            pc[s] *= float(d)
            if s in detail:
                detail[s]["p_c"] = round(pc[s], 4)
                detail[s]["steric_discount"] = round(float(d), 4)
        print(f"[steric discount] applied to {len(_STERIC_DISC)} sites "
              f"(mean={sum(_STERIC_DISC.values())/len(_STERIC_DISC):.3f})")

    # ---- Optional override: static-structure estimate instead of MD frames ----
    # Applies to a structure with no MD yet. The overridden p_c is inverted back to
    # an equivalent t_eff so every downstream consumer (LP bottleneck check, MC)
    # still sees a consistent number rather than a hole.
    if pc_override:
        k2 = k2_apparent(PKA_EFF, ph=PH)
        p0 = R_FEED * MAB_UM * 1e-6
        for s, v in pc_override.items():
            v = float(v)
            pc[s] = v
            raw = min(max(v / YIELD_MAX, 1e-12), 1.0 - 1e-12)
            te = -math.log(1.0 - raw) / (k2 * p0) if p0 > 0 else 0.0
            t_eff_by_site[s] = te
            ft = f_thiolate(PKA_EFF, PH)
            detail[s] = dict(source="static-structure estimate (no MD for this structure)",
                             t_eff_s=round(te, 1), f_thiolate=round(ft, 5),
                             auc_M_s=round(auc_payload(te), 6),
                             raw_conversion=round(raw, 4), p_c=round(v, 4))

    print("\n[site accessibility / thiol activation / covalent conjugation per-site] (pH=%.1f, exposure-conditioned pKa=%.1f±%.1f, k_hyd=%.1e)" %
          (PH, PKA_EFF, PKA_SD, K_HYD))
    print(f"  {'site':<12}{'t_eff(s)':>10}{'f_thio':>10}{'AUC(M*s)':>12}{'p_c':>8}  source")
    for s in sorted(detail):
        d = detail[s]
        if "t_eff_s" in d:
            print(f"  {s:<12}{d['t_eff_s']:>10.1f}{d['f_thiolate']:>10.5f}"
                  f"{d['auc_M_s']:>12.5f}{d['p_c']:>8.4f}  {d['source']}")
        else:
            print(f"  {s:<12}{'-':>10}{'-':>10}{'-':>12}{d['p_c']:>8.4f}  {d['source']}")

    # ---- disulfide reduction calibration: new form (shared-pool K) primary, old form (λ) for comparison only ----
    obs = list(zip(ANCHOR_EQ, ANCHOR_DAR))     # 4 real anchors, all measured
    anchor_temps = ANCHOR_TEMP[:len(obs)]
    pc_uniform = float(np.mean(list(pc.values())))
    # ---- local electrostatics (2026-09-29): apply before fitting K_ref ----
    # TCEP is an anion at process pH; a net-negative patch near a disulfide
    # repels it and slows the attack. Fold the charge factor into ease BEFORE
    # fitting, so the K_ref that comes out already accounts for the charge
    # environment of this antibody.
    chg = {p: charge_factor(q_local.get(p, 0.0)) for p in PAIR_NAMES}
    f_chg_mean = float(np.mean(list(chg.values())))
    ease = {p: ease[p] * (chg[p] / f_chg_mean) for p in PAIR_NAMES}
    K_ref, k_info = fit_K(ease, obs, pc_uniform=pc_uniform, anchor_temps=anchor_temps)
    t_ref = k_info.get("temp_ref_C")
    # K_ref is fitted at (t_ref, the anchors' [mAb], the anchors' reduction time).
    # The physical constant is k_TCEP; a new protocol must recombine it with ITS OWN
    # [mAb] and reduction time:  K_run = k_TCEP(T_run) x [mAb]_run x t_run.
    k_tcep_ref = K_ref / (ANCHOR_MAB_UM * 1e-6 * ANCHOR_T_RED_H * 3600.0)  # M^-1 s^-1 at t_ref
    # Peel the anchors' pH off before moving anywhere else: TCEP's own reactivity
    # is pH dependent (phosphorus pKa 7.6), so a K fitted at pH 7.0 is not the
    # same rate constant as a K needed at pH 7.4.
    k_intrinsic = k_tcep_intrinsic(K_ref, ph=ANCHOR_PH_RED)
    ph_ratio = f_tcep_active(PH_RED) / f_tcep_active(ANCHOR_PH_RED)
    k_tcep_run = k_tcep_ref * arrhenius(t_red, t_ref) * ph_ratio
    # ---- volume exclusion (2026-09-29): run-time only, not during fit ----
    # The anchors were measured at 67 µM; f_vol(67) is baked into K_ref.
    # For a run at a different concentration the ratio f_vol(run)/f_vol(anchor)
    # is the correction. At process concentrations (<200 µM) this is <5%.
    f_vol = volume_exclusion_factor(MAB_UM) / volume_exclusion_factor(ANCHOR_MAB_UM)
    k_tcep_run *= f_vol
    K_open = k_tcep_run * MAB_UM * 1e-6 * T_RED_H * 3600.0
    lam, lam_info = solve_lambda_multipoint(ease, obs, pc)   # legacy form, for comparison
    pred = pred_shared(TCEP_CAL, ease, K_open)
    d6 = full_dist(pc, pred)
    s6 = summarize(d6)

    print("\n" + "-" * 92)
    print(f"[disulfide reduction calibration] shared reductant pool: total opening extent "
          f"K = {K_open:.4f}  (at {t_red:g} C, {MAB_UM:g} µM, {T_RED_H:g} h)")
    print(f"   k_TCEP = {k_tcep_ref:.3f} M⁻¹s⁻¹ @ {t_ref:g} C / pH {ANCHOR_PH_RED:g}"
          f" (from K_ref={K_ref:.4f} on anchors @ {ANCHOR_MAB_UM:g} µM / {ANCHOR_T_RED_H:g} h)")
    print(f"            pH-stripped intrinsic rate = {k_intrinsic:.3f} M⁻¹s⁻¹"
          f" (TCEP phosphorus pKa {TCEP_P_PKA:g}, Cline 2004 PMID 15568811)")
    print(f"   -> {k_tcep_run:.3f} M⁻¹s⁻¹ @ {t_red:g} C / pH {PH_RED:g}"
          f" (Arrhenius x{arrhenius(t_red, t_ref):.3f}, pH x{ph_ratio:.3f},"
          f" vol x{f_vol:.4f}; charge factor is baked into the ease profile)")
    print(f"   {k_info}")
    print(f"            legacy form (pseudo-first-order) λ = {lam:.4f}   {lam_info}   <- comparison only")
    print("-" * 92)
    print(f"   {'TCEP eq':>8}{'T red C':>9}{'bonds open':>11}{'pred DAR':>9}{'meas DAR':>9}{'resid':>9}")
    for (e, dd), at in zip(obs, anchor_temps):
        Ki = k_at_temp(K_ref, at, t_ref) if (at is not None and t_ref is not None) else K_ref
        pr = pred_shared(e, ease, Ki)
        md = float((np.arange(9) * full_dist(pc, pr)).sum())
        print(f"   {e:>8.2f}{(at if at is not None else float('nan')):>9.0f}"
              f"{sum(pr.values()):>11.3f}{md:>9.3f}{dd:>9.2f}{md-dd:>+9.3f}")
    print("   residuals alternating +/- = only experimental noise remains; same-sign runs = wrong functional form")

    print(f"\n[main result] TCEP {TCEP_CAL} eq — bonds opened {sum(pred.values()):.3f} "
          f"(stoichiometric cap {TCEP_CAL:.2f}, achieved {sum(pred.values())/TCEP_CAL*100:.1f}%)")
    print("   DAR : " + "  ".join(f"{i:>6}" for i in range(9)))
    print("   P   : " + "  ".join(f"{x:>6.3f}" for x in d6))
    print(f"\n   mean = {s6['mean_dar']}   mode = {s6['mode_dar']}   "
          f"even/odd = {s6['even_odd_ratio']}   P(DAR>=6) = {s6['p_dar_ge6']}")

    # Physical identity of K: can it be checked against literature
    k_imp = implied_k_tcep(K_open)
    print(f"\n[physical identity of K] K = k_TCEP × [mAb] × t_reduction")
    print(f"   process {MAB_UM:.0f} µM / {T_RED_H:.0f} h / {t_red:g} °C  ->  implied k_TCEP = {k_imp:.2f} M⁻¹s⁻¹")
    print("   literature measured TCEP opening protein disulfides: PTP1B 1.5±0.5 / Prx1 3.66 / Prx2 6.25 M⁻¹s⁻¹")
    print(f"   -> {'within range ✓' if 1.0 <= k_imp <= 6.5 else 'outside range ✗ (indicates other chemistry: TCEP oxidized / hits intrachain bonds / insufficient time)'}")

    # What the same protocol would do at other reduction temperatures.
    # This is the temperature response surface the user actually needs: with everything
    # else fixed, only k_TCEP moves.
    if EA_TCEP is not None and t_ref is not None:
        print(f"\n[temperature response] same protocol, reduction temperature scanned "
              f"(Ea={EA_TCEP:g} kJ/mol, from Q10≈2 — assumption, not measured)")
        print(f"  {'T red (°C)':>11}{'K at T':>9}{'bonds open':>12}{'mean DAR':>10}{'K to aby K@22':>14}")
        temp_rows = []
        k22 = k_tcep_ref * arrhenius(22.0, t_ref)
        for tv in (4.0, 12.0, 22.0, 25.0, 30.0, 37.0):
            Kv = k_tcep_ref * arrhenius(tv, t_ref) * MAB_UM * 1e-6 * T_RED_H * 3600.0
            pv = pred_shared(TCEP_CAL, ease, Kv)
            sv = summarize(full_dist(pc, pv))
            temp_rows.append(dict(reduction_temp_C=tv, K=round(Kv, 4),
                                  bonds_opened=round(float(sum(pv.values())), 3),
                                  mean_dar=sv["mean_dar"]))
            print(f"  {tv:>11.0f}{Kv:>9.3f}{sum(pv.values()):>12.3f}{sv['mean_dar']:>10.3f}"
                  f"{(k_tcep_ref * arrhenius(tv, t_ref))/k22:>14.2f}")

    # ---- STEP 4: leave-one-out validation ----
    loo = leave_one_out_K(ease, obs, pc_uniform=pc_uniform,
                          anchor_temps=anchor_temps, temp_ref=t_ref)
    print(f"\n[leave-one-out validation] shared pool: {loo.get('status')}  mean absolute error = {loo.get('mean_abs_error')}")
    loo_old = leave_one_out(ease, obs, pc)
    print(f"           legacy form: {loo_old.get('status')}  mean absolute error = {loo_old.get('mean_abs_error')}")

    # ---- Uncertainty propagation (Monte Carlo) ----
    rng = np.random.default_rng(SEED)
    base = {s: t_eff_from_frames(frames[src]) for s, src in
            [(k, (k if k in frames else MIRROR_OF.get(k))) for k in
             [a for p, a, b in PAIRS for a in (a, b)]]
            if src in frames}
    means = []
    for _ in range(N_MC):
        pk = rng.normal(PKA_EFF, PKA_SD)
        pcm = {}
        for s in {a for p, a, b in PAIRS for a in (a, b)}:
            src = s if s in frames else MIRROR_OF.get(s)
            te = t_eff_by_site.get(s, base.get(src, 0.0))
            # Frame-level bootstrap resampling (simulating the sampling error from unconverged 2 ns).
            # Only possible for sites that actually have MD frames; a static-structure
            # estimate carries no frame ensemble to resample.
            if src in frames and not pc_override:
                fr = frames[src]
                te = t_eff_from_frames(rng.choice(fr, size=len(fr), replace=True))
            pcm[s] = pc_from_teff(te, pka=pk, chg_factor=_site_chg.get(s, 1.0),
                                  f_vol=_f_vol_couple, t_conj=t_conj)[0]
        # A payload fed at N equivalents cannot load more than N. Without this
        # the kinetic term alone reports a full DAR8 no matter how little was
        # fed, which is exactly how the one failed batch we own (5 eq fed,
        # 1.8 loaded) was missed.
        # A LOWER BOUND anchor says "at least this much dissolves", so it must
        # not be read backwards as "anything above this precipitates": doing so
        # capped a real process that plainly works. Only a measured value caps.
        _anc = (_QC.get("known_payloads", {}).get(
            _QC.get("payload_name", "vc-MMAE"), {}) or {})
        _is_lb = "LOWER BOUND" in str(_anc.get("source", "")).upper()
        _avail = 1.0 if _is_lb else payload_availability(
            R_FEED * MAB_UM * 1e-6, cac_M=_anc.get("cac_water_M"))[0]
        pcm, _feed_scale = cap_by_available_drug(pcm, R_FEED, _avail, pred)
        dd = full_dist(pcm, pred)
        means.append(float((np.arange(9) * dd).sum()))
    means = np.array(means)
    ci = np.percentile(means, [2.5, 97.5])
    print(f"\n[uncertainty propagation] {N_MC} Monte Carlo runs (pKa±{PKA_SD} @anchor pH, frame bootstrap resampling)")
    print(f"   mean DAR = {means.mean():.3f}   95% uncertainty range = [{ci[0]:.3f}, {ci[1]:.3f}]   "
          f"sd = {means.std():.3f}")
    if means.std() < 1e-6:
        print("   -> uncertainty range collapses to a point, indicating the uncertainty of site accessibility / thiol activation / covalent conjugation **cannot propagate through**:")
        print("      the conjugation probability is saturated (input perturbations are absorbed by the cap), which is direct evidence that")
        print("      accessibility / activation / kinetics are not rate-limiting under this process window.")
        print("      This is not a bug: the real error is in disulfide reduction (calibration of total opening extent K),")
        print("      which can only be constrained by multiple experimental anchors — see the leave-one-out result above.")

    # ---- pH scan: the real DoE dimension of thiol activation ----
    # At the pH=PH_REF anchor, k2_apparent exactly equals K2_REF (no double-counting);
    # only when deviating from the anchor does the pKa/pH effect appear. This is exactly the dimension needed for "process-window virtual screening".
    all_sites = sorted({a for p, a, b in PAIRS for a in (a, b)})

    def teff_of(s):
        if s in t_eff_by_site:
            return t_eff_by_site[s]
        src = s if s in frames else MIRROR_OF.get(s)
        return t_eff_from_frames(frames[src]) if src in frames else 0.0

    print(f"\n[thiol activation DoE] pH scan (pKa={PKA_EFF}; pH={PH_REF} anchors K2={K2_REF}, no double-counting)")
    print(f"  {'pH':>6}{'k2_app':>10}{'mean DAR':>10}{'even/odd':>10}{'P(DAR>=6)':>11}")
    sens = []
    for phv in (6.5, 7.0, 7.5, 8.0, 8.5, 9.0):
        pcm = {s: pc_from_teff(teff_of(s), pka=PKA_EFF, ph=phv)[0] for s in all_sites}
        ss = summarize(full_dist(pcm, pred))
        ka = k2_apparent(PKA_EFF, ph=phv)
        sens.append(dict(pH=phv, k2_apparent=round(float(ka), 1),
                         **{k: ss[k] for k in ("mean_dar", "even_odd_ratio", "p_dar_ge6")}))
        print(f"  {phv:>6.2f}{ka:>10.1f}{ss['mean_dar']:>10.3f}"
              f"{ss['even_odd_ratio']:>10.3f}{ss['p_dar_ge6']:>11.4f}")

    # Robustness of pKa itself: at the anchor pH, pKa jitter has no effect on the result (ratio is always 1)
    print(f"\n[thiol activation robustness] effect of pKa jitter ±{PKA_SD} at the pH={PH_REF} anchor (should be 0, proving no double-counting)")
    for pk in (PKA_EFF - PKA_SD, PKA_EFF, PKA_EFF + PKA_SD):
        pcm = {s: pc_from_teff(teff_of(s), pka=pk)[0] for s in all_sites}
        ss = summarize(full_dist(pcm, pred))
        print(f"   pKa={pk:>5.2f}  k2_app={k2_apparent(pk):>7.1f}  "
              f"mean DAR={ss['mean_dar']:.3f}  even/odd={ss['even_odd_ratio']:.3f}")

    # ---- STEP 7b: inverse solve — target DAR -> TCEP / LP equivalents (process-window scoping) ----
    tgt = [2.0, 3.0, 4.0, 5.0, 6.0] + [float(x) for x in extra_targets]
    pw = process_window_table(tgt, ease, K_open, pc, lp_overage_margin=0.5)
    print("\n" + "=" * 92)
    print("[inverse solve — PROCESS WINDOW for wet-lab scoping]")
    print("  Invert: target mean DAR -> TCEP eq (the true knob) and minimum LP eq (keep LP non-limiting).")
    print("  Units: TCEP eq / LP eq = molar equivalents vs antibody; DAR = payloads per antibody (mean); P(...) in %.")
    print("=" * 92)
    print(f"  {'target':>6}{'TCEP eq':>9}{'bonds open':>12}{'released SH eq':>15}"
          f"{'LP min eq':>11}{'pred DAR':>10}{'mode':>6}{'P(DAR>=6)':>12}{'P(DAR<=1)':>12}")
    for r in pw:
        if r["reachable"]:
            bo = float(np.sum(list(pred_shared(r["tcep_eq"], ease, K_open).values())))
            print(f"  {r['target_dar']:>6.1f}{r['tcep_eq']:>9.3f}{bo:>12.3f}"
                  f"{r['released_thiols_eq']:>15.3f}{r['lp_min_eq']:>11.3f}"
                  f"{r['pred_mean_dar']:>10.3f}{r['pred_mode_dar']:>6.0f}"
                  f"{r['p_dar_ge6']*100:>11.2f}%{r['p_dar_le1']*100:>11.2f}%")
        else:
            print(f"  {r['target_dar']:>6.1f}  UNREACHABLE — {r['error']}")

    # self-consistency: feed the solved TCEP eq back into the forward model
    print("\n  [self-consistency] re-solve TCEP eq -> feed back into forward -> check mean DAR == target")
    print(f"  {'target':>6}{'solved TCEP eq':>15}{'forward mean DAR':>18}{'residual':>12}")
    pw_self = []
    for r in pw:
        if not r["reachable"]:
            continue
        eq = r["tcep_eq"]
        fwd = mean_dar_from_eq(eq, ease, K_open, pc)
        res = fwd - r["target_dar"]
        pw_self.append(dict(target_dar=r["target_dar"], tcep_eq=round(eq, 4),
                            forward_mean_dar=round(fwd, 4), residual=round(res, 4)))
        print(f"  {r['target_dar']:>6.1f}{eq:>15.4f}{fwd:>18.4f}{res:>+12.4f}")
    max_abs = max(abs(x["residual"]) for x in pw_self) if pw_self else 0.0
    print(f"  max |residual| = {max_abs:.4f}  (inverse is self-consistent when ~0)")

    # LP bottleneck analysis: at what feed does LP start to become the limiting factor?
    lp_lim = lp_feed_at_pc_floor(t_eff_by_site, pc_floor=0.98)
    print(f"\n[LP bottleneck check] current payload feed = {R_FEED:.2f} eq")
    print(f"  LP becomes the bottleneck only below ~{lp_lim['threshold_eq']:.3f} eq "
          f"(conjugation probability of the least-accessible site drops to 0.98).")
    print("  -> within the whole realistic window LP is in heavy excess; DAR is set by TCEP, not LP.")

    # ---- Dual payload: two payloads competing for the same 8 thiols ----
    dual_out = None
    cfg_dual = dict(DUAL)
    if dual:
        cfg_dual.update({k: v for k, v in dual.items() if v is not None})
    feed_A, feed_B = cfg_dual.get("feed_eq_A"), cfg_dual.get("feed_eq_B")
    if cfg_dual.get("enabled") and feed_A and feed_B:
        kr = float(cfg_dual.get("k_ratio_A_over_B", 1.0) or 1.0)
        cf = cfg_dual.get("class_factor") or None
        te_map = {s: teff_of(s) for s in all_sites}

        # --- How much of each payload is monomeric in THIS medium -------------
        # Co-solvent fraction follows from how the protocol was actually pipetted
        # (the payload arrives as a DMSO/DMA stock, so more feed = more organic).
        f_cos = cosolvent_fraction(float(feed_A) + float(feed_B),
                                   tcep_eq=TCEP_CAL)
        f_cos = float(_MEDIUM.get("co_solvent_vv", f_cos)) if _MEDIUM.get(
            "co_solvent_vv") is not None else f_cos
        cA = float(feed_A) * MAB_UM * 1e-6
        cB = float(feed_B) * MAB_UM * 1e-6
        avail_A, diag_A = payload_availability(cA, logP=cfg_dual.get("logp_A"),
                                               cosolvent_vv=f_cos)
        avail_B, diag_B = payload_availability(cB, logP=cfg_dual.get("logp_B"),
                                               cosolvent_vv=f_cos)
        # A measured anchor beats an estimate from logP whenever one exists.
        tot_feed = float(feed_A) + float(feed_B)
        for side, nm, fed in (("A", cfg_dual.get("payload_A"), feed_A),
                              ("B", cfg_dual.get("payload_B"), feed_B)):
            fav, diag = availability_for_payload(nm, fed, total_feed_eq=tot_feed,
                                                 cosolvent_vv=f_cos)
            if diag.get("status") == "OK":
                if side == "A":
                    avail_A, diag_A = fav, diag
                else:
                    avail_B, diag_B = fav, diag

        # --- Class factor: compute it from the payloads if their sizes are known ---
        if cf is None and cfg_dual.get("mw_A") and cfg_dual.get("mw_B"):
            cf, cf_diag = class_factor_from_descriptors(
                cfg_dual.get("mw_A"), cfg_dual.get("mw_B"),
                logp_A=cfg_dual.get("logp_A"), logp_B=cfg_dual.get("logp_B"),
                logp_weight=float(_QC.get("class_logp_weight", 0.0) or 0.0))
        else:
            cf_diag = None

        seq = str(cfg_dual.get("dosing", "simultaneous")).lower() == "sequential"
        poly, sdual, dinfo = dual_payload_dist(te_map, pred, feed_A, feed_B,
                                               k_ratio=kr, pc_fallback=pc,
                                               class_factor=cf,
                                               avail_A=avail_A, avail_B=avail_B,
                                               sequential=seq,
                                               stage1_h=cfg_dual.get("stage1_h"),
                                               stage2_h=cfg_dual.get("stage2_h"))
        jd = poly
        na = np.arange(jd.shape[0])[:, None] * np.ones((1, jd.shape[1]))
        nb = np.ones((jd.shape[0], 1)) * np.arange(jd.shape[1])[None, :]
        print("\n" + "=" * 92)
        print("[dual payload] two payloads sharing the same thiol pool")
        print(f"  A = {cfg_dual.get('payload_A') or 'A'} @ {feed_A:g} eq | "
              f"B = {cfg_dual.get('payload_B') or 'B'} @ {feed_B:g} eq | "
              f"k_A/k_B = {kr:g}")
        if dinfo.get("mass_balance"):
            mb = dinfo["mass_balance_info"]
            what = ("thiols ran out" if mb.get("thiol_limited") else "payload ran out")
            print(f"  both payloads are consumed as they react -> {what}")
            print(f"  usable feed: A {mb['feed_eff_A']:g} eq, B {mb['feed_eff_B']:g} eq "
                  f"against {mb['n_sites']:g} free thiols")
            for cls in ("LC", "HC"):
                c = dinfo["by_class"][cls]
                print(f"    {cls}: A {c['dar_A']:.2f}  B {c['dar_B']:.2f}  "
                      f"(B share {100 * (c['frac_B'] or 0):.1f}%)")
            print(f"  odds q_A = {dinfo['q_A']:.4f} is only the instantaneous preference; "
                  f"the split above is what the consumed feeds actually allow")
        else:
            print(f"  branching ratio q_A = k_A[A] / (k_A[A] + k_B[B]) = {dinfo['q_A']:.4f}"
                  f"   (feed ratio alone would give "
                  f"{float(feed_A)/(float(feed_A)+float(feed_B)):.4f})")
            print("  concentrations held constant (mass_balance off) - only valid while "
                  "both payloads are in large excess")
        if cf:
            print(f"  site-class factor applied: {cf}  ->  q_A per class {dinfo['q_by_class']}")
            print(f"  {'class':>8}{'sites':>7}{'DAR_A':>9}{'DAR_B':>9}{'frac B':>9}")
            for cls in ("LC", "HC"):
                bc = dinfo["by_class"][cls]
                print(f"  {cls:>8}{bc['n_sites']:>7}{bc['dar_A']:>9.3f}"
                      f"{bc['dar_B']:>9.3f}{bc['frac_B']:>9.3f}")
        print(f"  {'':>10}DARA = {sdual['dar_A']:.3f}   DARB = {sdual['dar_B']:.3f}   "
              f"total = {sdual['dar_total']:.3f}   most likely (A,B) = {sdual['argmax']}")
        print(f"  spread: A count per molecule ~ Binom(8, q_A), sd = "
              f"{math.sqrt(8*dinfo['q_A']*(1-dinfo['q_A'])):.3f} "
              f"-> '6+2' is an AVERAGE, not what every molecule carries")
        top = np.dstack(np.unravel_index(np.argsort(jd.ravel())[::-1][:6], jd.shape))[0]
        print(f"  most probable species (n_A, n_B, P):")
        for i, j in top:
            print(f"      ({i}, {j})  P = {jd[i, j]:.4f}")
        dual_out = dict(payload_A=cfg_dual.get("payload_A"), payload_B=cfg_dual.get("payload_B"),
                        feed_eq_A=float(feed_A), feed_eq_B=float(feed_B),
                        k_ratio_A_over_B=kr, **dinfo, **sdual,
                        joint_top=[dict(n_A=int(i), n_B=int(j), p=round(float(jd[i, j]), 5))
                                   for i, j in top],
                        note=("n_A and n_B are NOT independent (one cannot take the same "
                              "thiol twice); the joint distribution is a 2-D convolution."))

    # ---- STEP 8: serialize results ----
    out = dict(
        run_tag=tag,
        inputs=dict(reduction_profile=os.path.basename(prof_path),
                    pc_source=("static-structure override" if pc_override
                               else "v5 MD per-frame SASA"),
                    reduction_temp_C=t_red, conjugation_temp_C=t_conj),
        inverse=dict(
            max_reachable_mean_dar=round(max_reachable_mean_dar(pc), 4),
            process_window=pw,
            self_consistency=pw_self,
            max_abs_self_residual=round(max_abs, 6),
            lp_bottleneck=dict(current_feed_eq=R_FEED,
                               lp_limiting_below_eq=round(lp_lim["threshold_eq"], 4),
                               pc_floor=0.98, per_site_eq={k: round(v, 4) for k, v in lp_lim["per_site_eq"].items()})),
            note=("TRUE inverse = target DAR -> TCEP eq (root-find, monotonic). "
                  "LP eq is reported only as the minimum feed that keeps LP non-limiting "
                  "(released thiols x (1 + overage margin)); it does NOT set DAR."),

        method=("DAR v6: disulfide reduction (shared-pool K, stoichiometric cap) x "
                "site accessibility (per-frame SASA -> effective exposure time) x "
                "thiol activation (thiolate fraction) x "
                "covalent conjugation (hydrolysis-aware AUC) "
                "x combinatorial statistics (4 disulfide-pair convolution + MC uncertainty)"),
        config_source=CFG.get("_source"),
        params=dict(K2=K2, mAb_uM=MAB_UM, payload_eq=R_FEED, time_h=T_H, pH=PH,
                    P_CONJ=P_CONJ, SASA_LO=SASA_LO, SASA_SPAN=SASA_SPAN,
                    PKA_EFF=PKA_EFF, PKA_SD=PKA_SD, K_HYD=K_HYD, TCEP_cal_eq=TCEP_CAL,
                    reduction_time_h=T_RED_H, reduction_temp_C=t_red,
                    conjugation_temp_C=t_conj, ea_tcep_kJ_per_mol=EA_TCEP,
                    ea_k2_kJ_per_mol=EA_K2),
        temperature=dict(reduction_temp_C=t_red, conjugation_temp_C=t_conj,
                         K_ref=round(K_ref, 4), K_ref_at_temp_C=t_ref,
                         K_at_run_temp=round(K_open, 4),
                         k_TCEP_M_inv_s_at_ref=round(k_tcep_ref, 3),
                         k_TCEP_M_inv_s_at_run=round(k_tcep_run, 3),
                         ea_tcep_kJ_per_mol=EA_TCEP,
                         ea_source=("back-derived from Q10~2, a generic chemistry rule of "
                                    "thumb — NOT a measured activation energy for TCEP + "
                                    "disulfide. TIER C until replaced."),
                         scan=locals().get("temp_rows", [])),
        dual_payload=dual_out,
        disulfide_reduction=dict(model="shared_pool", K_open=round(K_open, 4), **k_info,
                   k_TCEP_M_inv_s_at_ref=round(k_tcep_ref, 3),
                   k_TCEP_M_inv_s_at_run=round(k_tcep_run, 3),
                   anchor_conditions=dict(mab_uM=ANCHOR_MAB_UM,
                                          reduction_time_h=ANCHOR_T_RED_H),
                   implied_k_TCEP_M_inv_s=round(k_imp, 3),
                   bonds_opened=round(float(sum(pred.values())), 4),
                   stoich_cap=TCEP_CAL,
                   old_pseudo_first_order=dict(lam=round(lam, 4), **lam_info),
                   leave_one_out=loo, leave_one_out_old=loo_old),
        fixes_vs_v5=["thiol activation: f_thio now in the kinetic chain",
                     "payload hydrolysis [P](t) decay",
                     "MC uncertainty propagation",
                     "multipoint calibration + LOO framework"],
        key_caveat=("exposure-conditioned pKa used instead of static PROPKA: "
                    "static C-Cys214 pKa=20.76 is an artifact of a buried conformer that "
                    "site accessibility already excludes; using it would double-count burial."),
        sites=detail, lambda_calibrated=round(lam, 4), lambda_info=lam_info,
        primary=s6, leave_one_out=loo,
        uncertainty=dict(n_mc=N_MC, mean_dar=round(float(means.mean()), 3),
                         ci95=[round(float(ci[0]), 3), round(float(ci[1]), 3)],
                         ci95_note="95% uncertainty range from MC (pKa jitter + frame bootstrap), not a frequentist confidence interval",
                         sd=round(float(means.std()), 3)),
        pka_sensitivity=sens)
    dest = out_path or OUT
    json.dump(out, open(dest, "w"), indent=2, ensure_ascii=False)
    print("\nsaved ->", dest)
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="DAR v6 complete model")
    ap.add_argument("--profile", help="reduction-ease profile json "
                                      "(default: results_hinge/ss_reduction_profile.json)")
    ap.add_argument("--pc-json", help="json {site: p_c} overriding the MD-derived "
                                      "conjugation probability (for structures without MD)")
    ap.add_argument("--out", help="output json path (default: results_hinge/dar_v6_complete.json)")
    ap.add_argument("--tag", help="short label recorded inside the output json")
    ap.add_argument("--target-dar", type=float, action="append", default=[],
                    help="extra target mean DAR for the inverse-solve table (repeatable)")
    ap.add_argument("--reduction-temp", type=float, default=None,
                    help="reduction temperature in C (REQUIRED: set here or in the "
                         "config; there is deliberately no default)")
    ap.add_argument("--conjugation-temp", type=float, default=None,
                    help="conjugation temperature in C (default 22)")
    # ---- dual payload ----
    ap.add_argument("--payload-a", help="name of payload A")
    ap.add_argument("--payload-b", help="name of payload B")
    ap.add_argument("--feed-a", type=float, help="payload A molar equivalents vs antibody")
    ap.add_argument("--feed-b", type=float, help="payload B molar equivalents vs antibody")
    ap.add_argument("--k-ratio", type=float, default=None,
                    help="k2_A / k2_B (default 1.0: two maleimides of the same class)")
    # ---- per-run process overrides (to reproduce a literature / patent protocol) ----
    ap.add_argument("--mab-um", type=float, default=None,
                    help="antibody concentration in µM for this run (overrides yaml)")
    ap.add_argument("--tcep-eq", type=float, default=None,
                    help="TCEP equivalents for this run (overrides yaml)")
    ap.add_argument("--reduction-time-h", type=float, default=None,
                    help="reduction duration in hours for this run (overrides yaml)")
    ap.add_argument("--payload-feed", type=float, default=None,
                    help="single-payload feed equivalents for this run (overrides yaml)")
    ap.add_argument("--class-factor", action="append", default=[], metavar="CLS=VAL",
                    help="per-site-class competitiveness factor for payload A, e.g. "
                         "HC=2.56 (repeatable; LC and HC accepted)")
    ap.add_argument("--glycan", action="store_true",
                    help="run the glycan-conjugation branch (N-glycan at Asn297) "
                         "instead of the cysteine route")
    ap.add_argument("--glycan-sites", type=int, default=None,
                    help="number of glycan sites (default: yaml, normally 2)")
    ap.add_argument("--glycan-per-site", type=int, default=None,
                    help="payload slots per glycan site: 1 -> DAR2, 2 -> DAR4")
    ap.add_argument("--glycan-anchor-dar", type=float, default=None,
                    help="measured mean DAR of a glycan ADC; back-solves the "
                         "per-site efficiency")
    ap.add_argument("--glycan-eff", type=float, default=None,
                    help="per-site attachment efficiency (overrides yaml eff_total)")
    a = ap.parse_args()
    if a.glycan:
        GLYCAN["enabled"] = True
        if a.glycan_sites is not None:
            GLYCAN["n_sites"] = int(a.glycan_sites)
        if a.glycan_per_site is not None:
            GLYCAN["payloads_per_site"] = int(a.glycan_per_site)
        if a.glycan_anchor_dar is not None:
            GLYCAN["anchor_mean_dar"] = float(a.glycan_anchor_dar)
        if a.glycan_eff is not None:
            GLYCAN["eff_total"] = float(a.glycan_eff)
            GLYCAN["efficiency_mode"] = "total"
    pcov = json.load(open(a.pc_json)) if a.pc_json else None
    dual = None
    if a.feed_a and a.feed_b:
        dual = dict(enabled=True, payload_A=a.payload_a, payload_B=a.payload_b,
                    feed_eq_A=a.feed_a, feed_eq_B=a.feed_b)
        if a.k_ratio is not None:
            dual["k_ratio_A_over_B"] = a.k_ratio
        cf = {}
        for item in (a.class_factor or []):
            k, _, v = item.partition("=")
            if k.strip() in ("LC", "HC"):
                cf[k.strip()] = float(v)
        if cf:
            dual["class_factor"] = cf
    raise SystemExit(main(profile_path=a.profile, pc_override=pcov,
                          out_path=a.out, tag=a.tag, extra_targets=a.target_dar,
                          dual=dual, reduction_temp_C=a.reduction_temp,
                          conjugation_temp_C=a.conjugation_temp,
                          mab_uM=a.mab_um, tcep_eq=a.tcep_eq,
                          reduction_time_h=a.reduction_time_h,
                          payload_feed=a.payload_feed))
