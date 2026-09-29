"""Sequence QC: does the model read the conjugation topology off a structure correctly?

This is the first stage of the pipeline. If it mis-reads how many interchain
disulfides exist, every downstream stage runs on the wrong site set, so these
assertions are the cheapest guard available.
"""

import pytest

from conftest import LOCKED


def test_disulfides_are_typed(seqqc_result):
    """Every disulfide must be classified as either interchain or intrachain."""
    kinds = {p["kind"] for p in seqqc_result["disulfides"]}
    assert kinds <= {"inter", "intra"}, f"unexpected bond kinds: {kinds}"
    assert len(seqqc_result["disulfides"]) > 0, "no disulfide found at all"


def test_interchain_bond_count_is_four(seqqc_result):
    """IgG1 has 4 interchain disulfides -> 8 conjugatable thiols."""
    inter = [p for p in seqqc_result["disulfides"] if p["kind"] == "inter"]
    assert len(inter) == LOCKED["n_bond"], (
        f"expected {LOCKED['n_bond']} interchain disulfides, found {len(inter)}"
    )


def test_interchain_bonds_actually_cross_chains(seqqc_result):
    """A bond labelled 'inter' must join two different chains (label self-consistency)."""
    for p in seqqc_result["disulfides"]:
        if p["kind"] != "inter":
            continue
        ca, cb = p["a"].split("-")[0], p["b"].split("-")[0]
        assert ca != cb, f"bond labelled interchain but both ends on chain {ca}: {p}"


def test_conjugatable_site_count(seqqc_result):
    """4 interchain bonds give 8 conjugatable cysteines."""
    n = seqqc_result["cys_audit"]["n_conjugatable"]
    assert n == LOCKED["n_sites"], f"expected {LOCKED['n_sites']} sites, got {n}"


def test_no_unpaired_cysteine(seqqc_result):
    """An unpaired Cys would be labelled directly by maleimide -- an unintended site."""
    assert seqqc_result["cys_audit"]["n_free_cys"] == 0, (
        "unpaired cysteine found; it changes the site set and must be handled explicitly"
    )


def test_isotype_is_igg1(seqqc_result):
    """The model scope is IgG1 only; anything else needs a different topology."""
    assert seqqc_result["isotype"]["call"] == "IgG1"


def test_one_cysteine_never_bonded_twice(seqqc_result):
    """Greedy exclusive matching: a Cys appears in at most one disulfide."""
    seen = []
    for p in seqqc_result["disulfides"]:
        seen.extend([p["a"], p["b"]])
    assert len(seen) == len(set(seen)), "a cysteine appears in more than one bond"


@pytest.mark.parametrize("key", ["chain", "site"])
def test_conjugation_sites_are_structured(seqqc_result, key):
    """Downstream stages index sites by chain/residue; the fields must exist."""
    sites = seqqc_result["conjugation_sites"]
    assert len(sites) == LOCKED["n_sites"]
    for s in sites:
        assert key in s, f"site record missing '{key}': {s}"


# ----------------------------------------------------------------------------
# Conjugation-type routing: judge chemistry -> find matching sites
# ----------------------------------------------------------------------------
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from rdkit import Chem  # noqa: E402
from adcsim.io import load_antibody, load_md_frames  # noqa: E402
from adcsim.sites import surface_lysines, resolve_conjugation_sites  # noqa: E402
from adcsim.clash import hydrogen_false_collision_count  # noqa: E402
from conftest import ROOT  # noqa: E402

CHAIN_TYPES = {"A": "Heavy", "B": "Heavy", "C": "Light", "D": "Light"}
DISULF = {"intra_cutoff": 2.5, "inter_cutoff": 3.0}


def test_surface_lysines_finds_sites(pdb_path):
    """Amine-reactive LP needs exposed lysines; they must be findable."""
    model = load_antibody(pdb_path)
    lys = surface_lysines(model)
    assert len(lys) > 0, "no lysines found in antibody"
    assert len([d for d in lys if d["accessible"]]) > 0
    for d in lys[:3]:
        assert {"chain", "residue_number", "site", "nz_sasa", "accessible"} <= d.keys()


def test_resolve_cysteine_gives_eight_sites_and_reduction_profile(pdb_path):
    """Cysteine conjugation: 8 interchain sites + a per-bond reduction profile."""
    model = load_antibody(pdb_path)
    resolved = resolve_conjugation_sites(model, "cysteine", CHAIN_TYPES, DISULF)
    assert resolved["type"] == "cysteine"
    assert len(resolved["sites"]) == LOCKED["n_sites"]
    assert resolved["reduction_profile"] is not None
    for r in resolved["reduction_profile"]:
        assert "reduction_difficulty" in r and r["type"] == "interchain"


def test_resolve_lysine_gives_lysine_sites(pdb_path):
    """Lysine conjugation: exposed lysines, and no disulfide reduction step."""
    model = load_antibody(pdb_path)
    resolved = resolve_conjugation_sites(model, "lysine", CHAIN_TYPES, DISULF)
    assert resolved["type"] == "lysine"
    assert resolved["reduction_profile"] is None
    assert len(resolved["sites"]) > 0


def test_load_md_frames_converts_nm_to_angstrom():
    """MD frames are stored in nm; the loader must return Angstrom."""
    npz = ROOT / "examples/trop_adc/data/md_af_nolock/allcut_r2.npz"
    if not npz.exists():
        pytest.skip(f"MD frames not present: {npz}")
    fr = load_md_frames(str(npz))
    probe = fr[0]
    nn = cKDTree(probe).query(probe, k=2)[0][:, 1]
    med = float(np.median(nn))
    # In Angstrom a protein's nearest-neighbour spacing is ~1.4 A; in nm it would be ~0.14.
    assert 1.0 < med < 2.0, f"frames look like they are still in nm (median nn={med})"


def test_hydrogen_false_collision_count():
    """Hydrogens attached to a heavy atom near the protein are 'false' clashes."""
    mol = Chem.AddHs(Chem.MolFromSmiles("C"))  # methane: 1 heavy C + 4 H
    heavy_xyz = np.array([[0.0, 0.0, 0.0]])     # the single carbon at origin
    protein_near = np.array([[0.5, 0.0, 0.0]])    # within cutoff + h_ext of the C
    protein_far = np.array([[5.0, 0.0, 0.0]])
    assert hydrogen_false_collision_count(protein_near, heavy_xyz, mol, cutoff=2.0, h_ext=1.1) == 4
    assert hydrogen_false_collision_count(protein_far, heavy_xyz, mol, cutoff=2.0, h_ext=1.1) == 0
