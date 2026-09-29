"""Regression: the calibrated numbers must not drift.

These are not "truth" checks -- they are drift checks. If a refactor changes any
of these, the model's behaviour changed, and that has to be a deliberate decision
rather than a side effect.
"""

from conftest import LOCKED

TOL = 1e-6


def test_total_opening_extent_is_locked(dar_result):
    """K_open is fitted from the 4 HIC anchors."""
    assert dar_result["disulfide_reduction"]["K_open"] == LOCKED["K_open"]


def test_mean_dar_is_locked(dar_result):
    assert dar_result["primary"]["mean_dar"] == LOCKED["mean_dar"]


def test_mode_dar_is_locked(dar_result):
    assert dar_result["primary"]["mode_dar"] == LOCKED["mode_dar"]


def test_leave_one_out_error_is_locked(dar_result):
    """The only real evidence that the reduction stage generalises."""
    assert dar_result["leave_one_out"]["mean_abs_error"] == LOCKED["loo_mae"]


def test_distribution_is_normalised(dar_result):
    """The DAR distribution must sum to 1 -- it is a probability distribution."""
    dist = dar_result["primary"]["dist"]
    assert len(dist) == 9, f"DAR 0-8 expected, got {len(dist)} entries"
    assert abs(sum(dist) - 1.0) < 1e-4, f"distribution sums to {sum(dist)}"


def test_distribution_is_non_negative(dar_result):
    assert all(p >= 0 for p in dar_result["primary"]["dist"])


def test_reported_mean_matches_its_own_distribution(dar_result):
    """Internal consistency: the reported mean must equal sum(i * p_i).

    Guards against the mean being computed on a different code path than the
    distribution that gets reported.
    """
    dist = dar_result["primary"]["dist"]
    recomputed = sum(i * p for i, p in enumerate(dist))
    assert abs(recomputed - dar_result["primary"]["mean_dar"]) < 1e-3, (
        f"reported mean {dar_result['primary']['mean_dar']} != "
        f"distribution mean {recomputed:.4f}"
    )


def test_even_dar_dominates(dar_result):
    """Near-quantitative conjugation on paired sites implies even DAR dominance.

    This is a physical consequence, not a fitted result, so it is the strongest
    single check on the combinatorial stage.
    """
    dist = dar_result["primary"]["dist"]
    even = sum(dist[i] for i in range(0, 9, 2))
    odd = sum(dist[i] for i in range(1, 9, 2))
    assert even > odd, f"even {even:.3f} should exceed odd {odd:.3f}"


# =========================================================================
# Temperature input (2026-09-26) + dual payload (wired into main the same day)
# -------------------------------------------------------------------------
def test_reduction_temperature_is_mandatory(model):
    """No default temperature: the model must refuse to run without one."""
    import pytest
    if model.T_RED_C is None:
        with pytest.raises(SystemExit):
            model.require_reduction_temp()


def test_arrhenius_identity_and_25_to_37_factor(model):
    """Same temperature -> factor 1.0; 25 -> 37 C with Ea=53 kJ/mol -> ~2.29x."""
    assert model.arrhenius(25.0, 25.0) == 1.0
    f = model.arrhenius(37.0, 25.0, 53.0)
    assert 2.2 < f < 2.4, f"Arrhenius 25->37 C factor {f:.3f}, expected ~2.29 (Q10~2)"


def test_dual_split_is_feed_ratio_at_equal_k(model):
    """Two maleimides with equal k2: branching ratio = feed ratio (5.6:2 -> 0.7368)."""
    q = model.dual_split(5.6, 2.0, 1.0)
    assert abs(q - 5.6 / 7.6) < 1e-9


def test_dual_joint_distribution_is_consistent(model, ease):
    """50/50 dual feed: symmetric joint, P sums to 1, total DAR == single-payload mean."""
    import numpy as np
    pred = model.pred_shared(2.5, ease, 1.1368)
    pc = {s: 0.995 for s in model._all_sites()}
    poly, s, info = model.dual_payload_dist({}, pred, 4.0, 4.0, k_ratio=1.0,
                                            pc_fallback=pc)
    assert abs(poly.sum() - 1.0) < 1e-9
    assert abs(poly - poly.T).max() < 1e-12, "50/50 feeds must give a symmetric joint"
    assert abs(s["dar_A"] - s["dar_B"]) < 1e-6
    single = float((np.arange(9) * model.full_dist(pc, pred)).sum())
    assert abs(s["dar_total"] - single) < 1e-3, (  # summarize_dual rounds to 3 decimals
        f"dual total {s['dar_total']} != single-payload mean {single:.4f}"
    )


def test_dual_branching_depletes_the_other_payload(model, ease):
    """A thiol taken by A cannot carry B: at 8:0 feed, B must get ~nothing."""
    pred = model.pred_shared(2.5, ease, 1.1368)
    pc = {s: 0.995 for s in model._all_sites()}
    poly, s, _ = model.dual_payload_dist({}, pred, 8.0, 0.001, k_ratio=1.0,
                                         pc_fallback=pc)
    assert s["dar_B"] < 0.01


# =========================================================================
# Per-site-class branching ratio (2026-09-26)
# Two payloads do not compete with equal odds at every cysteine: reaching a
# given site depends on the payload AND on how cramped that site is.
# -------------------------------------------------------------------------
def test_site_classes_split_into_light_and_heavy(model):
    """8 sites: 2 on the light chains, 6 on the heavy chains (4 hinge + 2 arm)."""
    classes = model.split_sites_by_class()
    assert sorted(classes) == ["HC", "LC"]
    assert len(classes["LC"]) == 2, classes["LC"]
    assert len(classes["HC"]) == 6, classes["HC"]
    assert {model.site_class(s)[1] for s in classes["HC"]} == {"hinge", "fab_arm"}
    assert {model.site_class(s)[1] for s in classes["LC"]} == {"light"}


def test_class_factor_reduces_to_uniform_when_all_one(model):
    """factor 1 everywhere must reproduce the plain feed-ratio split exactly."""
    q = model.dual_split_by_class(5.6, 2.0, 1.0, {"LC": 1.0, "HC": 1.0})
    assert abs(q["LC"] - q["HC"]) < 1e-12
    assert abs(q["LC"] - model.dual_split(5.6, 2.0, 1.0)) < 1e-12


def test_class_bias_redistributes_without_changing_total_dar(model, ease):
    """Favouring A at one class moves A vs B only; total occupancy is untouched."""
    pred = model.pred_shared(2.5, ease, 1.1368)
    pc = {s: 0.995 for s in model._all_sites()}
    _, s0, i0 = model.dual_payload_dist({}, pred, 5.6, 2.0, 1.0, pc_fallback=pc)
    _, s1, i1 = model.dual_payload_dist({}, pred, 5.6, 2.0, 1.0, pc_fallback=pc,
                                        class_factor={"LC": 1.0, "HC": 2.59})
    assert abs(s0["dar_total"] - s1["dar_total"]) < 1e-3, (
        f"total DAR moved from {s0['dar_total']} to {s1['dar_total']}; "
        f"splitting A/B between classes cannot change how many sites get filled")
    assert i1["by_class"]["HC"]["frac_B"] < i0["by_class"]["HC"]["frac_B"]
    assert i1["by_class"]["LC"]["frac_B"] > i1["by_class"]["HC"]["frac_B"]
    assert s1["dar_A"] > s0["dar_A"] and s1["dar_B"] < s0["dar_B"]


def test_fit_recovers_a_known_class_factor(model):
    """Round trip: build synthetic fractions from a known factor, fit it back."""
    known = 3.0
    obs = []
    for k, (fa, fb) in enumerate(((5.6, 2.0), (7.0, 4.0), (4.7, 4.5))):
        q = model.dual_split_by_class(fa, fb, 1.0, {"LC": 1.0, "HC": known})
        for cls in ("LC", "HC"):
            obs.append((f"run{k}", fa, fb, cls, 1.0 - q[cls]))
    fit = model.fit_class_factor_from_fractions(obs)
    assert fit["status"] == "OK"
    assert abs(fit["preference_ratio_HC_over_LC"] - known) < 1e-3, fit
    assert fit["residual_spread_ln"] == 0.0, fit


# =========================================================================
# Mass balance: both payloads are consumed (2026-09-26)
# The branching-ratio form holds the two concentrations constant, which is
# only true while both are in huge excess. Real feeds are a few equivalents
# against 8 thiols, so a payload fed at 6 eq cannot load more than 6 however
# fast it is. MediLink WO2024235127A1 shows exactly that: the fast payload's
# loading tracks its FEED almost 1:1 and the slow one gets only the leftovers.
# -------------------------------------------------------------------------
def test_mass_balance_never_loads_more_than_was_fed(model):
    """A payload fed at 3 eq cannot load 3+, however many thiols are free."""
    cap = {"LC": 2.0, "HC": 6.0}
    out, info = model.dual_loading_mass_balance(cap, feed_A=3.0, feed_B=0.0)
    assert info["status"] == "OK"
    assert out["LC"]["A"] + out["HC"]["A"] <= 3.0 + 1e-3


def test_mass_balance_never_exceeds_the_thiols_that_exist(model):
    cap = {"LC": 1.9, "HC": 5.8}
    out, _ = model.dual_loading_mass_balance(cap, feed_A=20.0, feed_B=20.0)
    assert sum(v["A"] + v["B"] for v in out.values()) <= sum(cap.values()) + 1e-3


def test_equal_rates_split_at_the_feed_ratio(model):
    """Same rate constant, thiol-limited -> the split is the feed ratio.

    dA/dB = (fA-a)/(fB-b) integrates to a/fA = b/fB, so a = N*fA/(fA+fB).
    """
    cap = {"LC": 1.9, "HC": 5.8}
    N = sum(cap.values())
    out, _ = model.dual_loading_mass_balance(cap, feed_A=6.0, feed_B=5.0)
    a = sum(v["A"] for v in out.values())
    assert abs(a - N * 6.0 / 11.0) < 0.02, (a, N * 6.0 / 11.0)


def test_fast_payload_takes_its_feed_and_slow_gets_leftovers(model):
    """Given enough time: A fills to its own feed, B only ever sees the rest."""
    cap = {"LC": 1.9, "HC": 5.8}
    N = sum(cap.values())
    out, _ = model.dual_loading_mass_balance(cap, feed_A=5.6, feed_B=4.0,
                                             k_ratio=200.0, t_end_s=1e6)
    a = sum(v["A"] for v in out.values())
    b = sum(v["B"] for v in out.values())
    assert abs(a - 5.6) < 0.05, a
    assert abs(b - (N - 5.6)) < 0.05, (b, N - 5.6)


def test_a_slow_second_payload_is_left_unfinished_at_2_h(model):
    """Same run stopped at 2 h: the slow payload has NOT reached its ceiling.

    This is the regime the patent reports (its table 9: the second payload needs
    >2 h). It is why the mass-balance form matters - a branching-ratio split
    would have the slow payload at its final ratio from the first minute.
    """
    cap = {"LC": 1.9, "HC": 5.8}
    N = sum(cap.values())
    out, _ = model.dual_loading_mass_balance(cap, feed_A=5.6, feed_B=4.0,
                                             k_ratio=200.0, t_end_s=7200.0)
    b = sum(v["B"] for v in out.values())
    assert 0.5 < b < (N - 5.6) - 0.1, b


def test_availability_caps_the_loading_absolutely(model):
    """avail scales the feed, so it caps loading: 8 eq fed but 20% usable."""
    cap = {"LC": 1.9, "HC": 5.8}
    out, info = model.dual_loading_mass_balance(cap, feed_A=8.0, feed_B=0.0,
                                                avail_A=0.2)
    assert abs(info["feed_eff_A"] - 1.6) < 1e-9
    assert sum(v["A"] for v in out.values()) <= 1.6 + 1e-3


def test_mass_balance_on_matches_the_old_branching_split(model, ease):
    """mass_balance=False must give exactly the pre-2026-09-26 behaviour."""
    pred = model.pred_shared(2.5, ease, 1.1368)
    pc = {s: 0.995 for s in model._all_sites()}
    _, s_off, i_off = model.dual_payload_dist({}, pred, 5.6, 2.0, 1.0,
                                             pc_fallback=pc, mass_balance=False)
    q = 5.6 / 7.6
    assert abs(i_off["q_A"] - round(q, 4)) < 1e-9   # q_A is rounded to 4 dp
    assert abs(s_off["dar_A"] / s_off["dar_total"] - q) < 1e-3
    assert i_off["mass_balance_info"]["status"] == "DISABLED"


def test_staged_addition_gives_the_first_payload_a_head_start(model, ease):
    """Adding A 1 h before B must raise A's share over a simultaneous run."""
    pred = model.pred_shared(5.5, ease, 3.9)
    pc = {s: 0.995 for s in model._all_sites()}
    _, s_sim, _ = model.dual_payload_dist({}, pred, 5.0, 4.0, 1.0,
                                          pc_fallback=pc)
    _, s_seq, i_seq = model.dual_payload_dist({}, pred, 5.0, 4.0, 1.0,
                                              pc_fallback=pc, sequential=True,
                                              stage1_h=1.0, stage2_h=1.0)
    assert i_seq["mode"] == "sequential"
    assert s_seq["dar_A"] > s_sim["dar_A"], (s_seq["dar_A"], s_sim["dar_A"])


# =========================================================================
# Stability / boundary guards (added before packaging)
# -------------------------------------------------------------------------
# These do NOT add features. They pin behaviour the code ALREADY has, so a
# future refactor (packaging into a toolkit / callable API) cannot silently
# break it.
# =========================================================================
import json  # noqa: E402


def _run(model, tmp_path, name, **kw):
    """Run main() once into a temp json and return the parsed result."""
    out = tmp_path / f"{name}.json"
    model.main(out_path=str(out), reduction_temp_C=37.0, **kw)
    return json.loads(out.read_text())


def test_repeat_runs_do_not_leak_global_state(model, tmp_path):
    """Two runs in one process: the second must not inherit the first's inputs.

    main() applies per-run overrides by rebinding module globals, and resets
    them at the start of every run. Config B run after config A must equal
    config B run after config B — otherwise packaging main() as a callable
    API would return contaminated results.
    """
    cfg_a = dict(mab_uM=67.0, tcep_eq=2.5, payload_feed=6.0, reduction_time_h=2.0)
    cfg_b = dict(mab_uM=100.0, tcep_eq=3.0, payload_feed=8.0, reduction_time_h=3.0)

    res_a = _run(model, tmp_path, "a", **cfg_a)
    res_b_after_a = _run(model, tmp_path, "b_after_a", **cfg_b)
    res_b_after_b = _run(model, tmp_path, "b_after_b", **cfg_b)

    # running B after A must give exactly the same numbers as running B after B
    assert res_b_after_a["primary"]["mean_dar"] == res_b_after_b["primary"]["mean_dar"], (
        f"global state leaked: B after A = {res_b_after_a['primary']['mean_dar']}, "
        f"B after B = {res_b_after_b['primary']['mean_dar']}"
    )
    assert res_b_after_a["disulfide_reduction"]["K_open"] == res_b_after_b["disulfide_reduction"]["K_open"]
    # and the two configurations must actually differ (the test is not vacuous)
    assert res_a["primary"]["mean_dar"] != res_b_after_a["primary"]["mean_dar"], (
        "configs A and B produced the same mean DAR - the test proves nothing"
    )
    assert res_a["disulfide_reduction"]["K_open"] == LOCKED["K_open"], (
        f"run A (the default configuration) drifted: "
        f"{res_a['disulfide_reduction']['K_open']} != {LOCKED['K_open']}"
    )


def test_repeat_default_run_is_stable(model, tmp_path):
    """Three consecutive default runs must return identical locked numbers."""
    first = _run(model, tmp_path, "d1")
    for i in (2, 3):
        again = _run(model, tmp_path, f"d{i}")
        assert again["primary"]["mean_dar"] == first["primary"]["mean_dar"]
        assert again["primary"]["mode_dar"] == first["primary"]["mode_dar"]
        assert again["disulfide_reduction"]["K_open"] == first["disulfide_reduction"]["K_open"]
    assert first["disulfide_reduction"]["K_open"] == LOCKED["K_open"]
    assert first["primary"]["mean_dar"] == LOCKED["mean_dar"]


def test_glycan_branch_is_off_by_default_and_runs_when_enabled(model, tmp_path):
    """The glycan (N297) route is a separate chemistry, disabled by default."""
    assert not bool(model.GLYCAN.get("enabled")), (
        "glycan route must stay off by default - the main model is the cysteine route"
    )
    # the branch itself works when invoked directly
    g = model.run_glycan_mode(out_path=str(tmp_path / "glycan.json"), verbose=False)
    assert g["chemistry"].startswith("glycan"), g["chemistry"]
    assert g["n_sites"] == 2
    # eff_total is a placeholder: the branch runs but absolute DAR is not anchored
    assert g["tier"] == "C", g["tier"]


def test_pc_override_marks_sites_as_static_estimate(model, tmp_path):
    """pc_override (no MD for this structure) must be labelled in the output.

    Guards the path used for antibodies that have no MD frame library yet.
    """
    sites = model._all_sites()
    override = {s: 0.90 for s in sites}
    res = _run(model, tmp_path, "pcov", pc_override=override)
    src = res["inputs"]["pc_source"]
    assert "static-structure override" in src, src
    for s in sites:
        assert "static-structure estimate" in res["sites"][s]["source"], res["sites"][s]
    # the override really is used: p_c is what was supplied, not the MD value
    for s in sites:
        assert abs(res["sites"][s]["p_c"] - 0.90) < 1e-6


def test_empty_reduction_profile_does_not_silently_return_default(model, tmp_path):
    """An empty / degenerate profile must not quietly produce the calibrated DAR."""
    import pytest

    prof = tmp_path / "empty_profile.json"
    prof.write_text(json.dumps({"pairs": []}))
    out = tmp_path / "empty_out.json"
    try:
        model.main(profile_path=str(prof), out_path=str(out), reduction_temp_C=37.0)
    except Exception:
        # raising is acceptable; silently returning the calibrated result is not
        return
    res = json.loads(out.read_text())
    assert res["primary"]["mean_dar"] != LOCKED["mean_dar"], (
        "an empty reduction profile returned the calibrated default DAR - "
        "the model must not silently fall back to a working configuration"
    )


def test_volume_exclusion_is_monotonic_and_bounded(model):
    """Crowding correction: ~1 at process concentration, smaller when crowded."""
    f67 = model.volume_exclusion_factor(67.0)
    f200 = model.volume_exclusion_factor(200.0)
    f400 = model.volume_exclusion_factor(400.0)
    assert 0.0 < f400 < f200 < f67 <= 1.0, (f67, f200, f400)
    # at the anchor concentration the correction is close to 1 (small effect)
    assert f67 > 0.95, f"crowding at 67 uM should be a few percent, got {f67}"
    # it is a ratio-based correction: same concentration -> exactly 1
    ratio = model.volume_exclusion_factor(67.0) / model.volume_exclusion_factor(67.0)
    assert abs(ratio - 1.0) < 1e-12


def test_tcep_stock_default_is_config_driven(model):
    """TCEP stock concentration comes from config, not a hidden constant."""
    # default value must still be 20.0 mM (behaviour unchanged), and readable
    assert abs(float(model.TCEP_STOCK_MM) - 20.0) < 1e-9, model.TCEP_STOCK_MM
    # it is what cosolvent_fraction() actually uses when not told otherwise
    f = model.cosolvent_fraction(6.0, mab_uM=67.0, tcep_eq=2.5)
    f_explicit = model.cosolvent_fraction(6.0, mab_uM=67.0, tcep_eq=2.5,
                                          tcep_mM=model.TCEP_STOCK_MM)
    assert abs(f - f_explicit) < 1e-12
    # a different stock changes the organic fraction (the value is live, not fixed)
    f_other = model.cosolvent_fraction(6.0, mab_uM=67.0, tcep_eq=2.5, tcep_mM=5.0)
    assert f_other != f
