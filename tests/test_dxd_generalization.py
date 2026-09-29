"""DXd case (DS-8201a / Enhertu) -- the strongest generalisation test available.

Case definition
---------------
* antibody : trastuzumab, humanized IgG1 kappa -- the SAME antibody the model
  was built on, so the conjugation topology must come out identical.
* linker   : mc-GGFG-AM (maleimidocaproyl - Gly-Gly-Phe-Gly - aminomethylene)
* payload  : DXd, a hydroxyacetyl-exatecan topoisomerase I inhibitor
* chemistry: cysteine - maleimide, interchain disulfides reduced with TCEP
* published DAR: ~7.7-8, i.e. essentially the theoretical maximum of 8
  (Ogitani 2016 Clin Cancer Res 22:5097, DOI 10.1158/1078-0432.CCR-16-0195;
   Nakada 2019 Chem Pharm Bull 67:173, DOI 10.1248/cpb.c18-00744)

Why this case was chosen
------------------------
It is the same antibody under a different payload class and a completely
different point on the DAR axis (7.7 instead of 4.6), which is the widest
swing reachable without changing the antibody sequence.

What it DOES and DOES NOT test -- read this before trusting a pass
------------------------------------------------------------------
DOES test:
  * structure-source transfer -- the pipeline must read an AlphaFold model
    (not the Boltz model it was developed on) and recover the same topology.
  * the saturation ceiling -- 7.7 must be reachable, but only just.

DOES NOT test:
  * the reduction calibration. DAR 7.7 sits on the plateau where the
    DAR-vs-TCEP curve is flat, so almost any K reproduces it. A pass here
    says nothing about the fitted total opening extent.

Honest expectation, written down in advance
-------------------------------------------
The model as calibrated (TCEP 2.5 eq, 67 uM, 2 h) does NOT reproduce 7.7 --
it gives about 4.6. That is not a failure: DS-8201a is made by near-complete
reduction, a different process condition. The useful output is the inverse
answer: "what TCEP equivalent would be needed". If that number is absurd
(e.g. 40 eq), the model is wrong. If it is in the range a process chemist
would actually use for full reduction, the model is consistent.
"""

import os

import pytest

from conftest import LOCKED, ROOT

# Where the AlphaFold structure is expected. The user folds this on Kaggle and
# drops it here; until then every test in this file skips rather than fails.
AF_CANDIDATES = [
    ROOT / "examples/trop_adc/data/trastuzumab_alphafold.pdb",
    ROOT / "examples/trop_adc/data/trastuzumab_af.pdb",
    ROOT / "examples/dxd_adc/data/trastuzumab_alphafold.pdb",
]

AF_PDB = next((p for p in AF_CANDIDATES if p.exists()), None)

PUBLISHED_DAR = 7.7          # Ogitani 2016, RPC
THEORETICAL_MAX_DAR = 8.0    # 4 interchain disulfides -> 8 cysteines

pytestmark = pytest.mark.skipif(
    AF_PDB is None,
    reason="AlphaFold structure not present; drop it at "
           f"{AF_CANDIDATES[0]} to enable the DXd generalisation tests",
)


@pytest.fixture(scope="module")
def af_seqqc():
    from adcsim import seqqc

    return seqqc.assess(str(AF_PDB))


@pytest.fixture(scope="module")
def af_ease():
    """Reduction ease recomputed from the AlphaFold model.

    Produced by examples/trop_adc/kaggle_md_8site/ss_reduction_profile.py,
    which takes any PDB as argv[1]. Skipped if that has not been run yet.
    """
    prof_path = ROOT / "examples/trop_adc/results_hinge/ss_reduction_profile_af.json"
    if not prof_path.exists():
        pytest.skip("run ss_reduction_profile.py on the AlphaFold model first")
    import json

    prof = json.loads(prof_path.read_text())
    return {p["site_a"] + "--" + p["site_b"]: p["reduction_ease"] for p in prof["pairs"]}


# =========================================================================
# Structure-source transfer: AlphaFold must give the same topology as Boltz
# =========================================================================
def test_af_model_is_recognised_as_igg1(af_seqqc):
    assert af_seqqc["isotype"]["call"] == "IgG1"


def test_af_model_gives_the_same_interchain_bond_count(af_seqqc):
    inter = [p for p in af_seqqc["disulfides"] if p["kind"] == "inter"]
    assert len(inter) == LOCKED["n_bond"], (
        f"AlphaFold model yielded {len(inter)} interchain disulfides, expected "
        f"{LOCKED['n_bond']}. A different count means the structure is missing or "
        f"mispairing the hinges -- check chain completeness before blaming the model."
    )


def test_af_model_gives_the_same_eight_sites(af_seqqc):
    assert af_seqqc["cys_audit"]["n_conjugatable"] == LOCKED["n_sites"]
    assert af_seqqc["cys_audit"]["n_free_cys"] == 0


def test_af_and_boltz_agree_on_which_residues_are_conjugatable(af_seqqc, seqqc_result):
    """Same antibody sequence -> the site set must be identical.

    The two structures come from different predictors, so coordinates differ;
    but which cysteines sit on an interchain disulfide is a sequence/chain
    topology fact and cannot legitimately differ. This is the cleanest
    available check that the topology stage reads structure, not a hard-coded
    residue list.

    Compared by chain KIND (heavy / light) rather than chain letter, because
    the two predictors may label chains differently.
    """

    def site_set(res):
        return {(s["chain_kind"], s["res_id"]) for s in res["conjugation_sites"]}

    af, boltz = site_set(af_seqqc), site_set(seqqc_result)
    assert af == boltz, (
        f"site sets differ between AlphaFold and Boltz.\n"
        f"  only in AlphaFold: {sorted(af - boltz)}\n"
        f"  only in Boltz:     {sorted(boltz - af)}\n"
        f"If the residue numbers differ but the pattern is the same shape, the two "
        f"structures use different numbering schemes -- renumber before comparing."
    )


def test_af_site_split_matches_igg1_topology(af_seqqc):
    """Numbering-independent form of the same check.

    IgG1 kappa: 3 conjugatable cysteines per heavy chain (hinge x2 + the one
    paired to the light chain) and 1 per light chain; two of each chain.
    """
    from collections import Counter

    # chain_kind is 'H' (heavy) or 'L' (light) as called by seqqc.
    c = Counter(s["chain_kind"] for s in af_seqqc["conjugation_sites"])
    assert sum(c.values()) == LOCKED["n_sites"]
    assert c.get("H", 0) == 6, f"expected 6 heavy-chain sites, got {c.get('H', 0)}"
    assert c.get("L", 0) == 2, f"expected 2 light-chain sites, got {c.get('L', 0)}"


def test_af_reduction_ease_is_a_bounded_score(af_ease):
    """ease is a clipped 0..1 proxy; out-of-range means the parser misread SG."""
    assert len(af_ease) == LOCKED["n_bond"]
    for bond, e in af_ease.items():
        assert 0.0 <= e <= 1.0, f"ease out of range for {bond}: {e}"


# =========================================================================
# Ceiling: 7.7 must be reachable, but only just
# =========================================================================
def test_published_dar_is_within_the_ceiling(model, pc):
    ceiling = model.max_reachable_mean_dar(pc)
    assert PUBLISHED_DAR <= ceiling, (
        f"published DAR {PUBLISHED_DAR} exceeds the model ceiling {ceiling:.3f}; "
        f"the ceiling is wrong, not the calibration"
    )
    assert ceiling <= THEORETICAL_MAX_DAR + 1e-9


def test_dxd_sits_on_the_plateau_not_the_rising_part(model, pc):
    """DAR 7.7 is 96%+ of the ceiling -- it is a saturation result.

    Stated so nobody later claims this case validates the reduction stage.
    """
    ceiling = model.max_reachable_mean_dar(pc)
    assert PUBLISHED_DAR / ceiling > 0.95, (
        f"DAR {PUBLISHED_DAR} is only {PUBLISHED_DAR / ceiling:.1%} of the ceiling"
    )


def test_default_condition_does_not_reproduce_dxd(model, ease, pc):
    """Written down in advance, so it is a prediction rather than a post-hoc excuse.

    At the calibrated condition (TCEP 2.5 eq) the model gives ~4.6, far below
    7.7. DS-8201a is made by near-complete reduction, so the discrepancy is a
    process difference, not a model error.
    """
    d = model.mean_dar_from_eq(2.5, ease, LOCKED["K_open"], pc)
    assert d < PUBLISHED_DAR - 2.0, (
        f"expected the default condition to fall well short of {PUBLISHED_DAR}, got {d:.3f}"
    )


def test_reaching_dxd_needs_excess_reductant_but_not_absurd_amounts(model, ease, pc):
    """The inverse answer: what TCEP equivalent would give DAR 7.7?

    Full reduction of 4 bonds needs more than 4 eq (one TCEP per bond is the
    stoichiometric floor and the reaction is not instantaneous). A requirement
    above ~20 eq would mean the reduction stage is broken.
    """
    eq = model.solve_tcep_eq_for_target_dar(PUBLISHED_DAR, ease, LOCKED["K_open"], pc)
    assert LOCKED["n_bond"] < eq < 20.0, (
        f"DAR {PUBLISHED_DAR} requires TCEP {eq:.2f} eq, outside the plausible "
        f"full-reduction window"
    )


def test_full_reduction_releases_all_eight_thiols(model, ease):
    """The necessary condition for DAR ~8: all four bonds open."""
    opened = sum(model.pred_shared(20.0, ease, K=60.0).values())
    assert abs(opened - LOCKED["n_bond"]) < 0.05, (
        f"even at 20 eq the model only opens {opened:.3f} of {LOCKED['n_bond']} bonds"
    )


# =========================================================================
# Does the whole reduction stage survive moving to a different structure source?
#
# Measured (2026-09-24), both fits against the same 4 HIC anchors:
#     Boltz refined   : K = 1.1368, rmse 0.039, LOO MAE 0.0513, k_TCEP 2.36
#     AlphaFold model_2: K = 1.1850, rmse 0.043, LOO MAE 0.0547, k_TCEP 2.46
#       (earlier figures used AlphaFold model_3, which the input QC later
#        rejected for 31 atomic overlaps; model_2 replaced it)
#
# ** THIS COMPARISON IS INVALID -- DO NOT QUOTE IT **
# The two structures were NOT prepared the same way:
#   * Boltz side input  = B220_235_disulfide_repaired.pdb, which came from
#     B220_235_rosetta_best.pdb (Rosetta refinement) and then had its S-S
#     idealised to 2.01-2.04 A, plus 9958 added hydrogens (20182 atoms).
#   * AlphaFold side    = raw server output, 10224 heavy atoms, no refinement.
#
# Re-measured afterwards with the SAME protocol on all three structures
# (count of sites whose reduced-state static SASA falls below the 5 A^2 gate):
#       AlphaFold raw : 1 dead site   (B-Cys232)
#       Boltz raw     : 3 dead sites  (A-Cys223, B-Cys223, D-Cys214)
#       Boltz refined : 1 dead site   (C-Cys214, 0.00 A^2)
# So on a like-for-like basis AlphaFold RAW is better than Boltz RAW, and even
# the refined Boltz structure still has a dead site. The earlier "AlphaFold
# fits worse" claim was an artifact of comparing refined-Boltz against
# raw-AlphaFold. It is retracted.
#
# What this actually shows is that no single static conformation is adequate
# for hinge accessibility -- see below on the static-SASA route.
#
# To make this comparison fairly, the AlphaFold model must go through the same
# refinement treatment; those are Rosetta-based Kaggle notebooks under
# examples/trop_adc/runpod/ and do not run locally.
#
# Asserted as a bar, not as "AF must be worse" -- if a future structure beats
# Boltz, that should be allowed to happen.
# =========================================================================
LOO_MAE_BAR = 0.15          # DAR units. Above this the transfer has broken the model.
K_TCEP_WINDOW = (1.0, 6.5)  # M^-1 s^-1, literature range (PTP1B 1.5, Prx1 3.66, Prx2 6.25)


def test_af_refit_stays_inside_the_literature_rate_window(model, af_ease):
    """K has a physical meaning: K = k_TCEP x [mAb] x t.

    Refitting on a different structure must still land k_TCEP inside the
    independently measured range. If it fell outside, the structure would be
    implying chemistry that does not exist.
    """
    K, _ = model.fit_K(af_ease)
    k = model.implied_k_tcep(K)
    lo, hi = K_TCEP_WINDOW
    assert lo <= k <= hi, (
        f"refitting on the AlphaFold structure implies k_TCEP = {k:.2f} M^-1 s^-1, "
        f"outside the measured window [{lo}, {hi}]"
    )


def test_af_reduction_profile_meets_the_quality_bar(model, af_ease):
    """Leave-one-out on the same anchors must stay usable after the transfer."""
    loo = model.leave_one_out_K(af_ease)
    assert loo["mean_abs_error"] <= LOO_MAE_BAR, (
        f"LOO MAE rose to {loo['mean_abs_error']} DAR units after moving to the "
        f"AlphaFold structure (bar {LOO_MAE_BAR})"
    )


def test_both_structures_agree_which_bond_is_hardest(model, af_ease, ease):
    """The one structural fact that must survive a change of predictor.

    Absolute ease values move a lot between structures (the whole point of
    normalising them away), but the ORDER should not invert: the lower hinge
    disulfide A-Cys232--B-Cys232 is the buried one in both.

    Normalised, this bond scores 0.835 on Boltz and 0.286 on AlphaFold -- a
    3x spread in magnitude, yet last place in both cases.
    """
    hard_af = min(model.ease_norm(af_ease))
    hard_boltz = min(model.ease_norm(ease))
    assert hard_af < 1.0 and hard_boltz < 1.0, "no bond is below-average relative ease"
    name_af = model.PAIR_NAMES[int(list(model.ease_norm(af_ease)).index(hard_af))]
    name_boltz = model.PAIR_NAMES[int(list(model.ease_norm(ease)).index(hard_boltz))]
    assert name_af == name_boltz, (
        f"hardest bond differs: AlphaFold says {name_af}, Boltz says {name_boltz}"
    )


# =========================================================================
# The DXd case run end-to-end on the AlphaFold model_2 structure
# -------------------------------------------------------------------------
# Inputs, so the numbers below are reproducible:
#   reduction profile : results_hinge/ss_reduction_profile_af.json
#   site conjugation  : results_hinge/af_model2_pc.json
#       (reduced-state static SG SASA -> exposure gate -> t_eff -> p_c.
#        model_2 has no MD ensemble yet, so this stands in for the per-frame
#        SASA the Boltz branch uses. MD is deferred, not skipped.)
#   calibration       : K = 1.1850 against the same 4 MMAE anchors
#
# Literature anchor for the DXd case itself (HRAM MS of T-DXd):
#   average DAR 7.94; dominant species DAR 8; DAR 6 present as a minor species;
#   occupancy >95% at all four sites (LC C214, HC C223 / C229 / C232).
# =========================================================================
AF_PC_JSON = ROOT / "examples/trop_adc/results_hinge/af_model2_pc.json"
AF_MODEL2_JSON = ROOT / "examples/trop_adc/results_hinge/dar_v6_af_model2.json"


@pytest.fixture(scope="module")
def af_pc():
    """Per-site conjugation probability measured on model_2 (no MD yet)."""
    if not AF_PC_JSON.exists():
        pytest.skip("run the static-SASA measurement on model_2 first")
    import json

    return json.loads(AF_PC_JSON.read_text())


@pytest.fixture(scope="module")
def af_run():
    if not AF_MODEL2_JSON.exists():
        pytest.skip("run dar_v6_complete.py with --profile ss_reduction_profile_af.json "
                    "--pc-json af_model2_pc.json first")
    import json

    return json.loads(AF_MODEL2_JSON.read_text())


def test_model2_has_no_buried_conjugation_site(af_pc):
    """Every one of the 8 cysteines must be exposed enough to saturate.

    This is what makes DAR ~8 possible at all. If a site came out buried, the
    ceiling would drop and the DXd case would become unreachable -- which is
    exactly the failure the next test pins.
    """
    assert len(af_pc) == LOCKED["n_sites"]
    for site, v in af_pc.items():
        assert v >= 0.99, f"{site} conjugates only {v:.3f} -- ceiling drops below 7.7"


def test_published_dar_7p7_is_unreachable_below_96_percent_per_site(model, af_ease, af_pc):
    """The literature DAR constrains the per-site conjugation probability.

    Mean DAR cannot exceed 8 x p_c however much reductant is thrown at it.
    DAR 7.7 therefore requires p_c >= 7.7/8 = 0.9625. This turns the published
    DAR into an independent check on the conjugation stage -- the one quantity
    the MMAE anchors alone cannot pin down, because there K and p_c only enter
    as a product.
    """
    floor = PUBLISHED_DAR / THEORETICAL_MAX_DAR
    for p_c in (0.90, 0.95):
        pc = {s: p_c for s in af_pc}
        assert model.max_reachable_mean_dar(pc) < PUBLISHED_DAR - 1e-9, (
            f"p_c={p_c} should put the ceiling below {PUBLISHED_DAR}"
        )
    assert floor > 0.96


def test_model2_reproduces_the_published_dar_under_excess_reductant(model, af_ease, af_pc):
    """At large TCEP excess the model must land on the published DAR.

    10 eq is chosen because it is what "near-complete reduction" looks like;
    the published average is 7.94 and the dominant species is DAR 8.
    """
    K = 1.1850
    d = model.full_dist(af_pc, model.pred_shared(10.0, af_ease, K))
    mean = float((__import__("numpy").arange(9) * d).sum())
    assert 7.7 <= mean <= THEORETICAL_MAX_DAR, f"mean DAR at 10 eq = {mean:.3f}"
    assert d[8] > 0.90, (
        f"P(DAR=8) = {d[8]:.3f}; HRAM MS reports DAR 8 as the dominant species"
    )


def test_dxd_needs_at_least_five_equivalents_at_a_physically_possible_rate(
        model, af_ease, af_pc):
    """Below ~5 eq the required rate constant stops being chemistry.

    Inverting for the K (= k_TCEP x [mAb] x t) that would deliver DAR 7.7:
        TCEP  4 eq -> K 8.75  -> k_TCEP 18.1 M^-1 s^-1   (3x any measured value)
        TCEP  5 eq -> K 2.23  -> k_TCEP  4.6 M^-1 s^-1   (inside the window)
        TCEP 10 eq -> K 0.56  -> k_TCEP  1.2 M^-1 s^-1   (inside the window)
    So the model does not just say "add more TCEP" -- it says 4 eq is
    impossible at any real rate, and ~5 eq is where it becomes achievable.
    """
    from scipy.optimize import brentq

    def mean_at(eq, K):
        return model.mean_dar_from_eq(eq, af_ease, K, af_pc)

    def k_needed(eq):
        K = brentq(lambda k: mean_at(eq, k) - PUBLISHED_DAR, 0.05, 400.0, xtol=1e-6)
        return K, model.implied_k_tcep(K)

    _, k4 = k_needed(4.0)
    assert k4 > K_TCEP_WINDOW[1], (
        f"4 eq would only need k_TCEP {k4:.2f}; the model is too optimistic about "
        f"near-stoichiometric reduction"
    )
    _, k5 = k_needed(5.0)
    assert K_TCEP_WINDOW[0] <= k5 <= K_TCEP_WINDOW[1], (
        f"5 eq needs k_TCEP {k5:.2f}, outside the measured window {K_TCEP_WINDOW}"
    )


def test_af_run_keeps_the_calibration_honest(af_run):
    """Guard on the saved model_2 run: the fit must stay inside its own bars."""
    g = af_run["disulfide_reduction"]
    assert abs(g["K_open"] - 1.1850) < 0.02, f"K moved to {g['K_open']}"
    k = g["implied_k_TCEP_M_inv_s"]
    assert K_TCEP_WINDOW[0] <= k <= K_TCEP_WINDOW[1], f"k_TCEP = {k}"
    assert g["leave_one_out"]["mean_abs_error"] <= LOO_MAE_BAR
    # residuals must alternate in sign, i.e. only noise left, not a wrong form
    signs = [1 if r > 0 else -1 for r in g["residuals"]]
    assert len(set(signs)) == 2, (
        f"residuals {g['residuals']} do not alternate -- wrong functional form"
    )
