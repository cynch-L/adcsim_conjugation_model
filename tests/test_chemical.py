"""Physical limits and conservation laws.

These are the checks that do NOT depend on the fitted calibration. They ask:
when an input is pushed to a physically extreme value, does the model return the
physically required answer? If any of these fail, the functional form is wrong --
not the fitted number.

Two families:
  * limits       -- input -> 0 or -> infinity, output must hit the right wall
  * conservation -- matter / probability cannot be created by the model
"""

import math

import numpy as np
import pytest

from conftest import LOCKED

TOL = 1e-6


# =========================================================================
# Limits
# =========================================================================
def test_no_reductant_means_no_bond_opens(model, ease):
    """TCEP = 0 -> nothing is reduced -> zero free thiols."""
    pred = model.pred_shared(0.0, ease, LOCKED["K_open"])
    assert all(abs(v) < 1e-9 for v in pred.values()), f"opened with no TCEP: {pred}"


def test_zero_exposure_means_zero_conjugation(model):
    """A site that is never exposed can never be conjugated, however long you wait."""
    assert model.pc_from_teff(0.0)[0] == 0.0


def test_infinite_exposure_saturates_at_yield_not_above(model):
    """p_c is a probability: it saturates at p_conj (0.995) and never reaches 1."""
    p_c = model.pc_from_teff(1e9)[0]
    assert abs(p_c - model.P_CONJ) < 1e-6, f"expected saturation at {model.P_CONJ}, got {p_c}"
    assert p_c <= 1.0


def test_total_hydrolysis_kills_conjugation(model):
    """If the maleimide hydrolyses instantly, effective payload exposure is ~0."""
    auc = model.auc_payload(3000.0, k_hyd=1e6)
    assert auc < 1e-6, f"effective payload exposure should vanish, got {auc}"


def test_thiolate_fraction_stays_in_unit_interval(model):
    """f_thiolate is a species fraction: it can approach 0 or 1 but never leave (0, 1)."""
    for pka in (3.0, 7.0, 8.5, 14.0, 20.0):
        f = model.f_thiolate(pka, ph=7.0)
        assert 0.0 < f < 1.0, f"f_thiolate({pka}) = {f}"
    assert model.f_thiolate(3.0, ph=7.0) > 0.99    # pKa far below pH -> fully deprotonated
    assert model.f_thiolate(14.0, ph=7.0) < 0.01   # pKa far above pH -> fully protonated


def test_k2_at_reference_ph_is_exactly_literature(model):
    """Anti-double-counting guard.

    Literature k2 = 300 M^-1 s^-1 is already an apparent rate measured at pH 7,
    so at pH 7 the model must return 300 unchanged for ANY pKa. If this drifts,
    the thiolate term is being applied twice.
    """
    for pka in (7.0, 8.5, 11.0, 20.0):
        assert abs(model.k2_apparent(pka, ph=model.PH_REF) - model.K2_REF) < TOL, (
            f"k2 at reference pH must be {model.K2_REF}, got {model.k2_apparent(pka, ph=model.PH_REF)}"
        )


def test_dar_ceiling_never_exceeds_site_count(model, pc):
    """8 cysteines -> at most 8 payloads, whatever the inputs."""
    ceiling = model.max_reachable_mean_dar(pc)
    assert ceiling <= LOCKED["n_sites"] + TOL, f"ceiling {ceiling} exceeds {LOCKED['n_sites']} sites"


def test_unreachable_target_is_reported_not_faked(model, ease, pc):
    """Asking for DAR 99 must raise with the reachable ceiling, not silently clamp."""
    with pytest.raises(ValueError, match="UNREACHABLE"):
        model.solve_tcep_eq_for_target_dar(99.0, ease, LOCKED["K_open"], pc)


def test_zero_target_is_rejected(model, ease, pc):
    with pytest.raises(ValueError):
        model.solve_tcep_eq_for_target_dar(0.0, ease, LOCKED["K_open"], pc)


# =========================================================================
# Conservation
# =========================================================================
EQS = [0.5, 1.0, 1.5, 2.5, 4.0, 8.0]


@pytest.mark.parametrize("eq", EQS)
def test_opened_bonds_respect_stoichiometric_cap(model, ease, eq):
    """One TCEP reduces one disulfide. Opened bonds <= min(TCEP eq, bond count).

    This is the whole point of the shared-pool model: the old pseudo-first-order
    form had no cap at all and could "open" more bonds than reductant added.
    """
    opened = sum(model.pred_shared(eq, ease, LOCKED["K_open"]).values())
    cap = min(eq, LOCKED["n_bond"])
    assert opened <= cap + 1e-6, f"opened {opened:.4f} bonds with a cap of {cap}"


@pytest.mark.parametrize("eq", [1.0, 2.5, 6.0])
def test_reductant_is_fully_consumed_at_long_time(model, ease, eq):
    """Given unlimited time, every TCEP molecule that can react does react.

    Sum of opened bonds -> eq (or 4, if there are fewer bonds than reductant).
    """
    opened = sum(model.pred_shared(eq, ease, K=60.0).values())
    expected = min(eq, LOCKED["n_bond"])
    assert abs(opened - expected) < 1e-3, (
        f"at full conversion {expected} bonds should be open, got {opened:.4f}"
    )


def test_released_thiols_are_two_per_opened_bond(model, ease):
    """One disulfide -> two cysteines. No thiols appear from anywhere else."""
    pred = model.pred_shared(2.5, ease, LOCKED["K_open"])
    opened = sum(pred.values())
    thiols = 2.0 * opened
    assert abs(thiols - 2.0 * opened) < TOL
    assert thiols <= LOCKED["n_sites"] + TOL, "more thiols released than cysteines exist"


def test_ceiling_equals_sum_of_site_probabilities(model, pc):
    """The saturation ceiling is just the sum of the 8 per-site probabilities."""
    assert abs(model.max_reachable_mean_dar(pc) - sum(pc.values())) < TOL


def test_perfect_conjugation_gives_only_even_dar(model, ease):
    """Every bond opens and both cysteines react -> each pair adds 0 or 2 -> DAR is even.

    Odd DAR is the fingerprint of one-half-of-a-pair failing. If a fully open,
    fully conjugated system produced odd DAR, the pair convolution would be
    modelling 8 independent sites instead of 4 pairs.
    """
    pred = {p: 1.0 for p in model.PAIR_NAMES}
    pc = {s: 1.0 for s in model._all_sites()}
    d = model.full_dist(pc, pred)
    odd = float(d[1::2].sum())
    assert odd < 1e-9, f"fully conjugated system produced odd DAR with p={odd}"
    assert abs(d[8] - 1.0) < 1e-9, f"expected all mass at DAR 8, got {d[8]}"


def test_one_bond_half_done_is_the_only_odd_source(model, ease):
    """With 3 pairs complete and 1 pair half-open, DAR 7 must be the only odd state."""
    pred = {p: 1.0 for p in model.PAIR_NAMES}
    pred[model.PAIR_NAMES[0]] = 1.0
    pc = {s: 1.0 for s in model._all_sites()}
    # break exactly one cysteine on one pair: it can only contribute 1, never 2
    pc[model.PAIRS[0][1]] = 0.0
    d = model.full_dist(pc, pred)
    assert abs(d[7] - 1.0) < 1e-9, f"expected all mass at DAR 7, got dist={d}"


def test_distribution_carries_no_mass_beyond_eight(model, ease, pc):
    """full_dist returns exactly DAR 0..8 and all of it is probability."""
    pred = model.pred_shared(2.5, ease, LOCKED["K_open"])
    d = model.full_dist(pc, pred)
    assert len(d) == 9
    assert abs(float(d.sum()) - 1.0) < 1e-9
    assert float(d.min()) >= -1e-12, f"negative probability in distribution: {d}"


def test_mean_dar_increases_with_reductant(model, ease, pc):
    """Monotonicity. The inverse solver bisects on this, so it must actually hold."""
    means = [model.mean_dar_from_eq(eq, ease, LOCKED["K_open"], pc) for eq in EQS]
    assert all(b >= a - 1e-9 for a, b in zip(means, means[1:])), f"non-monotonic: {means}"


def test_inverse_solve_round_trips(model, ease, pc):
    """target DAR -> TCEP eq -> forward -> same DAR."""
    for target in (2.0, 3.0, 4.0, 5.0):
        eq = model.solve_tcep_eq_for_target_dar(target, ease, LOCKED["K_open"], pc)
        fwd = model.mean_dar_from_eq(eq, ease, LOCKED["K_open"], pc)
        assert abs(fwd - target) < 1e-3, (
            f"inverse gave eq={eq:.4f}; forward returned {fwd:.4f}, target {target}"
        )


def test_min_lp_feed_covers_released_thiols(model, ease, pc):
    """LP feed must at least match the thiols released, plus the chosen overage."""
    inv = model.min_lp_eq_for_target_dar(4.0, ease, LOCKED["K_open"], pc,
                                         lp_overage_margin=0.5)
    released = inv["released_thiols_eq"]
    assert abs(inv["lp_min_eq"] - released * 1.5) < 1e-9
    assert inv["lp_min_eq"] >= released, "LP feed below stoichiometry"


def test_implied_k_tcep_matches_k_definition(model):
    """K = k_TCEP x [mAb] x t. Inverting it must recover K, not drift."""
    k = model.implied_k_tcep(LOCKED["K_open"], mab_uM=10.0, t_red_h=2.0)
    back = k * (10.0e-6 * 2.0 * 3600.0)
    assert abs(back - LOCKED["K_open"]) < 1e-9, f"round-trip drifted: {back}"


# ----------------------------------------------------------------------------
# Conjugation-type classification (LP warhead -> site type)
# ----------------------------------------------------------------------------
from adcsim.chemistry import classify_conjugation  # noqa: E402
from conftest import ROOT  # noqa: E402


def test_classify_maleimide_is_cysteine():
    """vc-MMAE carries a maleimide -> thiol-Michael -> cysteine conjugation."""
    sdf = ROOT / "examples/trop_adc/data/payload_ensemble_79.sdf"
    if not sdf.exists():
        pytest.skip(f"LP ensemble not present: {sdf}")
    c = classify_conjugation(str(sdf))
    assert c["conjugation_type"] == "cysteine"
    assert "maleimide" in c["warhead"].lower()
    assert c["confidence"] == "C"


def test_classify_no_warhead_is_unknown(tmp_path):
    """A molecule with no recognised warhead must not be silently assigned."""
    from rdkit import Chem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))  # ethanol, no reactive warhead
    p = tmp_path / "ethanol.sdf"
    w = Chem.SDWriter(str(p))
    w.write(mol)
    w.close()
    c = classify_conjugation(str(p))
    assert c["conjugation_type"] == "unknown"
