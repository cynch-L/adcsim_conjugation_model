"""Site accessibility: is the SASA engine correct, and is the MD input sane?

Everything downstream (effective exposure time -> conjugation probability ->
DAR distribution) is driven by one number per site: solvent accessibility.
If the SASA engine is wrong, no amount of downstream rigour saves the result,
and the error is invisible -- the numbers still look plausible.

Two independent lines of defence:
  1. closed-form geometry -- a lone sphere, two spheres at a known separation.
     These have exact analytic answers, so they catch a broken engine directly.
  2. cross-implementation check -- our Shrake-Rupley against Biopython's,
     same radii, same probe, same point count. This is *implementation*
     validation, not physical validation; it proves the code, not the physics.
     Reference behaviour for this style of check: PDV benchmarks against
     FreeSASA at Pearson r ~ 0.997-0.998.
"""

import json
import math

import numpy as np
import pytest

from conftest import LOCKED, ROOT

PROBE = 1.4
N_POINTS = 960
RTOL = 0.03          # 960-point sphere sampling gives ~1-2% quantisation noise


# =========================================================================
# Closed-form geometry
# =========================================================================
def _sphere_area(r):
    return 4.0 * math.pi * (r + PROBE) ** 2


def test_lone_atom_gives_full_sphere():
    from adcsim.sasa import shrake_rupley

    r = 1.70
    got = float(shrake_rupley(np.zeros((1, 3)), np.array([r]),
                              probe=PROBE, n_points=N_POINTS)[0])
    assert abs(got - _sphere_area(r)) / _sphere_area(r) < RTOL, (
        f"lone atom: expected {_sphere_area(r):.2f} A^2, got {got:.2f}"
    )


def test_two_distant_atoms_do_not_shade_each_other():
    from adcsim.sasa import shrake_rupley

    r = 1.70
    coords = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    got = shrake_rupley(coords, np.array([r, r]), probe=PROBE, n_points=N_POINTS)
    for v in got:
        assert abs(v - _sphere_area(r)) / _sphere_area(r) < RTOL


def test_two_overlapping_spheres_match_the_analytic_cap():
    """Exposed area of each of two equal spheres at separation d:  2*pi*R^2 + pi*R*d

    with R = r + probe. This is the exact Shrake-Rupley occlusion geometry, so
    it pins down the neighbour test -- the part most likely to be subtly wrong
    (off-by-one distance, wrong effective radius).
    """
    from adcsim.sasa import shrake_rupley

    r, R = 1.70, 1.70 + PROBE
    for d in (R, 1.5 * R):
        coords = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
        got = shrake_rupley(coords, np.array([r, r]), probe=PROBE, n_points=N_POINTS)
        want = 2.0 * math.pi * R ** 2 + math.pi * R * d
        for v in got:
            assert abs(v - want) / want < RTOL, (
                f"d={d:.2f}: expected {want:.2f} A^2, got {v:.2f}"
            )


def test_occlusion_increases_as_atoms_approach():
    from adcsim.sasa import shrake_rupley

    r, R = 1.70, 1.70 + PROBE
    areas = []
    for d in (2.0 * R, 1.5 * R, 1.0 * R, 0.5 * R):
        coords = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
        areas.append(float(shrake_rupley(coords, np.array([r, r]),
                                         probe=PROBE, n_points=N_POINTS)[0]))
    assert all(b < a for a, b in zip(areas, areas[1:])), f"not monotonically shaded: {areas}"


def test_fully_surrounded_atom_is_buried():
    """A probe-sized shell of neighbours around a central atom drives SASA to ~0."""
    from adcsim.sasa import shrake_rupley

    r = 1.70
    R = r + PROBE
    shell = 2.6          # < R, so every test point of the centre atom is covered
    n = 60
    # Fibonacci sphere of neighbours
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = math.pi * (1 + 5 ** 0.5) * i
    pts = np.stack([np.cos(theta) * np.sin(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(phi)], axis=1) * shell
    coords = np.vstack([np.zeros(3), pts])
    radii = np.full(len(coords), r)
    got = float(shrake_rupley(coords, radii, probe=PROBE, n_points=N_POINTS)[0])
    assert got < 1e-6, f"surrounded atom should be fully buried, got {got:.4f} A^2"


# =========================================================================
# Cross-implementation check
# =========================================================================
@pytest.fixture(scope="module")
def sasa_crosscheck(pdb_path):
    """Our production core vs Biopython ShrakeRupley on the real antibody."""
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley

    from adcsim.sasa import OONS_RADII, shrake_rupley

    struct = PDBParser(QUIET=True).get_structure("mab", pdb_path)
    model = struct[0]
    for res in model.get_residues():
        for atom in list(res):
            if atom.element == "H":
                res.detach_child(atom.get_id())

    atoms = [a for a in model.get_atoms()]
    coords = np.array([a.coord for a in atoms], float)
    radii = np.array([OONS_RADII.get(a.element, 1.70) for a in atoms], float)

    ours = shrake_rupley(coords, radii, probe=PROBE, n_points=N_POINTS)
    ShrakeRupley(probe_radius=PROBE, n_points=N_POINTS).compute(model, level="A")
    ref = np.array([a.sasa for a in atoms], float)
    return ours, ref


def test_sasa_matches_biopython(sasa_crosscheck):
    """Pearson r >= 0.99 against an independent implementation.

    Same radii, same probe, same point count, so any gap is an implementation
    difference -- exactly the thing this test exists to catch.
    """
    ours, ref = sasa_crosscheck
    r = float(np.corrcoef(ours, ref)[0, 1])
    assert r >= 0.99, f"Pearson r = {r:.4f} -- below 0.99 means a real implementation bug"


def test_sasa_has_no_systematic_bias(sasa_crosscheck):
    """No global offset: our numbers must be unbiased, not merely correlated."""
    ours, ref = sasa_crosscheck
    mad = float(np.mean(np.abs(ours - ref)))
    bias = float(np.mean(ours - ref))
    assert mad < 1.0, f"mean absolute deviation {mad:.3f} A^2 is too large"
    assert abs(bias) < 0.5, f"systematic bias {bias:+.3f} A^2"


def test_saved_sasa_calibration_still_passes():
    """Drift guard on the recorded calibration run."""
    p = ROOT / "examples/trop_adc/results_hinge/sasa_validation.json"
    if not p.exists():
        pytest.skip("no recorded SASA calibration run")
    d = json.loads(p.read_text())
    assert d["pearson_r"] >= 0.99, f"recorded Pearson r dropped to {d['pearson_r']}"


# =========================================================================
# MD input sanity -- the accessibility stage is only as good as its frames
# =========================================================================
def test_measured_sasa_frames_are_consistent():
    """All four measured sites must come from the same sampling scheme.

    Mixing a 100-frame site with a 200-frame site would silently weight one
    site's effective exposure time differently from the others.
    """
    import glob
    import os

    files = sorted(glob.glob(str(ROOT / "examples/trop_adc/results_hinge/*_v5_sasa.npy")))
    assert len(files) == 4, f"expected 4 measured sites, found {len(files)}"
    shapes = {np.load(f).shape for f in files}
    assert len(shapes) == 1, f"inconsistent frame counts across sites: {shapes}"


def test_measured_sasa_is_physical():
    """Per-frame SASA is an area: non-negative and below a single exposed cysteine."""
    import glob

    for f in glob.glob(str(ROOT / "examples/trop_adc/results_hinge/*_v5_sasa.npy")):
        a = np.load(f)
        assert a.min() >= 0.0, f"negative SASA in {f}"
        assert a.max() < 200.0, f"implausible SASA {a.max():.1f} A^2 in {f}"


def test_effective_exposure_time_is_within_the_process_window(model):
    """t_eff is a subset of the conjugation duration; it cannot exceed it."""
    import glob
    import os

    total = model.T_H * 3600.0
    for f in glob.glob(str(ROOT / "examples/trop_adc/results_hinge/*_v5_sasa.npy")):
        for row in np.load(f):
            te = model.t_eff_from_frames(row)
            assert 0.0 <= te <= total + 1e-6, (
                f"t_eff {te:.1f}s outside [0, {total:.0f}s] for {os.path.basename(f)}"
            )


def test_buried_site_is_less_accessible_than_exposed_site(model):
    """MD-measured ordering, not an assumption.

    C-Cys214 (the light-chain terminal cysteine) is the poorly exposed site in
    this antibody; A-Cys223 is well exposed. If accessibility were not reading
    the frames at all, all four sites would come out the same.
    """
    c214 = np.load(ROOT / "examples/trop_adc/results_hinge/C-Cys214_v5_sasa.npy")
    a223 = np.load(ROOT / "examples/trop_adc/results_hinge/A-Cys223_v5_sasa.npy")
    te_c = model.t_eff_from_frames(c214.ravel())
    te_a = model.t_eff_from_frames(a223.ravel())
    assert te_a > te_c, (
        f"expected A-Cys223 ({te_a:.0f}s) to be exposed longer than "
        f"C-Cys214 ({te_c:.0f}s)"
    )


def test_mirrored_sites_match_their_source(dar_result):
    """Sites with no MD data are mirrored by symmetry -- p_c must equal the donor's.

    This is a stated approximation, so it must be visible and exact: if the
    mirror is silently dropped, four of the eight sites become unfounded.
    """
    by_source = {}
    for site, rec in dar_result["sites"].items():
        by_source.setdefault(rec["source"], []).append((site, rec["p_c"]))
    for src, members in by_source.items():
        values = {p for _, p in members}
        assert len(values) == 1, (
            f"sites sharing source '{src}' disagree on p_c: {members}"
        )
