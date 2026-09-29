"""Input structure QC: does the pipeline reject a structure it cannot use?

The failures these tests encode are not hypothetical. Both happened:

  * A Boltz model with no C2 symmetry (two half-antibodies 24 A apart) was fed
    to the model for weeks. Nothing flagged it. Rosetta refinement did not fix
    it either (19.30 -> 19.46 A).
  * The AlphaFold Server's top-ranked model has no hinge disulfide, so only 2
    of the 4 interchain bonds were detected and the model ran on a wrong site
    set -- silently.

So the point of this file is not coverage, it is making sure those two
structures can never get through again.
"""

import os
import re

import pytest

from conftest import ROOT

FOLD_DIR = ROOT / "examples/trop_adc/data/fold_2026_09_24_18_06"
AF_CIF = {i: FOLD_DIR / f"fold_2026_09_24_18_06_model_{i}.cif" for i in range(5)}
BOLTZ_REPAIRED = ROOT / "examples/trop_adc/data/B220_235_disulfide_repaired.pdb"
BOLTZ_RAW = ROOT / "examples/trop_adc/data/trastuzumab_boltz_model_0.pdb"
AF_PDB = ROOT / "examples/trop_adc/data/trastuzumab_alphafold.pdb"

pytestmark = pytest.mark.skipif(
    not AF_CIF[2].exists(),
    reason="AlphaFold Server output not present",
)


@pytest.fixture(scope="module")
def qc():
    from adcsim import structure_qc

    return structure_qc


@pytest.fixture(scope="module")
def af2(qc):
    return qc.assess(str(AF_CIF[2]))


@pytest.fixture(scope="module")
def boltz(qc):
    if not BOLTZ_REPAIRED.exists():
        pytest.skip("Boltz repaired structure not present")
    return qc.assess(str(BOLTZ_REPAIRED))


# =========================================================================
# The accepted structure
# =========================================================================
def test_the_accepted_alphafold_model_passes_every_check(af2):
    assert af2["verdict"] == "PASS", f"rejected: {af2['reasons']}"


def test_accepted_model_has_four_interchain_disulfides(af2):
    assert af2["disulfides"]["n_interchain"] == 4


def test_accepted_model_is_c2_symmetric(af2):
    assert af2["symmetry"]["worst_A"] <= af2["symmetry"]["limit_A"]


def test_accepted_model_has_no_atomic_overlaps(af2):
    assert af2["clashes"]["n_clash"] == 0


def test_accepted_model_has_two_heavy_and_two_light_chains(af2):
    assert len(af2["composition"]["heavy"]) == 2
    assert len(af2["composition"]["light"]) == 2
    assert af2["composition"]["lengths_match"]


# =========================================================================
# Structures that must be rejected
# =========================================================================
def test_the_structure_the_model_was_built_on_is_rejected(boltz):
    """The symmetry failure that went unnoticed.

    Boltz (raw and Rosetta-refined alike) places the two half-antibodies about
    24 A apart. It is not a subtle error, and nothing in the old pipeline
    looked for it.
    """
    assert boltz["verdict"] == "FAIL"
    assert any("symmetry" in r for r in boltz["reasons"]), boltz["reasons"]
    assert boltz["symmetry"]["worst_A"] > 10.0


def test_the_default_ranked_alphafold_model_is_rejected(qc):
    """AlphaFold Server's top-ranked model never forms the hinge disulfide."""
    r = qc.assess(str(AF_CIF[0]), with_exposure=False)
    assert r["verdict"] == "FAIL"
    assert r["disulfides"]["n_interchain"] < 4, (
        "model_0 was expected to miss at least one interchain bond"
    )


@pytest.mark.parametrize("model", [1, 4])
def test_other_rejected_alphafold_models(qc, model):
    r = qc.assess(str(AF_CIF[model]), with_exposure=False)
    assert r["verdict"] == "FAIL"


# =========================================================================
# Internal consistency of the verdict
# =========================================================================
def test_verdict_agrees_with_the_reason_list(af2, boltz):
    for r in (af2, boltz):
        assert r["ok"] == (len(r["reasons"]) == 0)
        assert r["verdict"] == ("PASS" if r["ok"] else "FAIL")


def test_self_superposition_is_zero(qc):
    """Sanity check on the symmetry measure itself: a chain matches itself."""
    model = qc.load_model(str(AF_CIF[2]))
    ca = qc._ca_map(model, "A")
    assert qc._rmsd(ca, ca) < 1e-6


def test_clash_count_is_deterministic(qc):
    a = qc.check_clashes(qc.load_model(str(AF_CIF[2])))
    b = qc.check_clashes(qc.load_model(str(AF_CIF[2])))
    assert a["n_clash"] == b["n_clash"]


# =========================================================================
# The cutoff disagreement bug
# =========================================================================
def test_disulfide_cutoff_matches_the_downstream_script():
    """Pin the two cutoffs together so they cannot drift apart again.

    seqqc used 3.0 A while ss_reduction_profile had 2.5 A hardcoded. On the
    AlphaFold model_2 structure that made seqqc find 4 interchain bonds and the
    reduction profile find 3 -- the A-Cys229--B-Cys229 pair at 2.60 A was
    silently dropped, and the reduction profile ran on 3 bonds instead of 4.
    """
    from adcsim import seqqc
    from adcsim import structure_qc

    script = ROOT / "examples/trop_adc/kaggle_md_8site/ss_reduction_profile.py"
    if not script.exists():
        pytest.skip("reduction profile script not present")
    src = script.read_text()
    m = re.search(r"^SS_CUTOFF\s*=\s*([0-9.]+)", src, re.M)
    assert m, "ss_reduction_profile.py no longer defines SS_CUTOFF"
    downstream = float(m.group(1))
    assert downstream == seqqc.SS_BOND_INTER, (
        f"cutoffs disagree: ss_reduction_profile {downstream} A vs "
        f"seqqc {seqqc.SS_BOND_INTER} A"
    )
    assert structure_qc.SS_CUTOFF == seqqc.SS_BOND_INTER


def test_accepted_model_detects_the_same_bonds_as_the_reduction_profile():
    """End-to-end form of the same check: both stages must see 4 bonds."""
    import json

    prof = ROOT / "examples/trop_adc/results_hinge/ss_reduction_profile_af.json"
    if not prof.exists():
        pytest.skip("reduction profile not run on the AlphaFold model")
    n = json.loads(prof.read_text())["n_interchain_ss"]
    assert n == 4, f"reduction profile found {n} interchain bonds on the accepted model"
