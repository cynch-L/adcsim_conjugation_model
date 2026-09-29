"""Input antibody structure QC.

Answers one question: **is this structure allowed into the DAR model?**

Why this module exists
----------------------
Every downstream step assumes the input structure is a physically valid IgG1.
That assumption has been silently violated twice in this project:

  1. A Boltz model whose two Fab arms were placed 19.5 A apart relative to each
     other -- no C2 symmetry at all -- was used for months. Rosetta refinement
     did not fix it (19.30 -> 19.46 A).
  2. The default top-ranked AlphaFold model has no hinge disulfide formed
     (SG-SG 5.44 A), so the pipeline read only 2 of the 4 interchain bonds and
     ran on a wrong site set -- without raising an error.

Both were invisible because nothing checked the input. This module is that
check. It measures coordinates only, so it runs before anything is calibrated
or fitted, and can reject a structure before a single downstream number exists.

The four checks and the failure each one catches
------------------------------------------------
composition  two heavy + two light chains, matching lengths
             -> catches truncated or mis-assembled models
disulfide    every interchain disulfide within the detection cutoff
             -> catches bonds the predictor never formed
symmetry     the two heavy chains must superpose
             -> catches a wrong quaternary arrangement (failure 1 above)
clash        no close atomic contacts other than real bonds
             -> catches models needing a minimisation pass before use

A fifth item is measured but only warns:
exposure     how many of the 8 conjugatable cysteines fall below the
             accessibility floor in a single static conformer
             -> tells you whether MD is going to be necessary (it usually is)

Every numeric limit below states where it comes from. Limits with no external
source are marked provisional and must be calibrated against accepted
structures before anyone defends them.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

from Bio.PDB import MMCIFParser, PDBParser, Superimposer
from Bio.PDB.Polypeptide import is_aa

# ---------------------------------------------------------------- limits ----
# Disulfide detection cutoff. Real S-S bond length is ~2.05 A (Cremlyn 1996);
# this is a tolerance for predicted coordinates, not a bond length. Kept equal
# to seqqc.SS_BOND_INTER so the two modules cannot disagree again -- they once
# did, and the same antibody gave different site counts as a result.
SS_CUTOFF = 3.0

# Symmetry limit. An IgG1 is a C2-symmetric homodimer of two identical heavy
# chains, so a correct model must superpose them. The reported number is the
# worst of (heavy-chain superposition, half-antibody superposition).
# Measured on trastuzumab, the values fall into two clearly separated clusters:
#     accepted AlphaFold models : 1.21, 0.41, 0.22 A
#     rejected models           : 4.47, 24.08, 24.16, 37.20 A
# The limit sits in the empty gap between 1.21 and 4.47. It is a separation
# between observed clusters, not a physical constant -- recalibrate it if a
# structure ever lands inside the gap.
SYMMETRY_MAX_A = 2.0

# Clash limits: distance below which two non-bonded heavy atoms count as
# overlapping, and how many are tolerated.
# PROVISIONAL -- no external source. Calibrate against a set of accepted
# experimental Fab/Fc structures before defending this number.
CLASH_DIST_A = 2.2
CLASH_MAX = 5

# Accessibility floor, reused from config.yaml (site accessibility, sasa_lo)
# so the QC and the model agree.
SASA_FLOOR_A2 = 5.0
SASA_PROBE = 1.4
SASA_N_POINTS = 960

MIN_HEAVY_RES = 300


# ------------------------------------------------------------- loading ------
def load_model(path: str):
    """Read a structure file (PDB or mmCIF) and return the first model."""
    parser = MMCIFParser(QUIET=True) if path.lower().endswith((".cif", ".mmcif")) \
        else PDBParser(QUIET=True)
    name = os.path.basename(path).rsplit(".", 1)[0]
    return parser.get_structure(name, path)[0]


def _confidence_of(res) -> Optional[float]:
    """Median per-residue confidence, if the file carries it.

    AlphaFold and Boltz both write pLDDT into the B-factor column (0-100).
    A genuine B-factor column can hold almost any positive number, so this is
    treated as confidence only when the whole column lands inside 0-100.
    """
    vals = [a.bfactor for a in res]
    if not vals:
        return None
    m = float(np.median(vals))
    return m if 0.0 <= m <= 100.0 else None


# ------------------------------------------------------------- checks ------
def check_composition(model, chains: List[str]) -> Dict[str, Any]:
    """Two heavy and two light chains, with matching partner lengths."""
    counts = {c: sum(1 for r in model[c] if is_aa(r, standard=True)) for c in chains}
    heavy = [c for c in chains if counts[c] >= MIN_HEAVY_RES]
    light = [c for c in chains if counts[c] < MIN_HEAVY_RES]
    lengths_match = (
        len(heavy) == 2 and counts[heavy[0]] == counts[heavy[1]]
        and len(light) == 2 and counts[light[0]] == counts[light[1]]
    )
    return dict(ok=bool(len(heavy) == 2 and len(light) == 2 and lengths_match),
                heavy=heavy, light=light, n_residues=counts,
                lengths_match=bool(lengths_match))


def check_disulfides(topology: Dict[str, Any]) -> Dict[str, Any]:
    """Every detected interchain disulfide must sit inside the cutoff."""
    inter = [p for p in topology.get("disulfides", []) if p.get("kind") == "inter"]
    dists = [float(p["dist"]) for p in inter]
    return dict(n_interchain=len(inter), max_dist_A=max(dists) if dists else None,
                cutoff_A=SS_CUTOFF, pairs=[f"{p['a']}--{p['b']}" for p in inter],
                ok=bool(dists) and max(dists) <= SS_CUTOFF)


def _ca_map(model, cid, lo=None, hi=None) -> Dict[int, Any]:
    out = {}
    for r in model[cid]:
        if not is_aa(r, standard=True) or "CA" not in r:
            continue
        i = r.id[1]
        if lo is not None and not (lo <= i <= hi):
            continue
        out[i] = r["CA"]
    return out


def _rmsd(a: Dict[int, Any], b: Dict[int, Any]) -> Optional[float]:
    common = sorted(set(a) & set(b))
    if len(common) < 20:
        return None
    sup = Superimposer()
    sup.set_atoms([a[i] for i in common], [b[i] for i in common])
    return float(sup.rms)


def check_symmetry(model, composition: Dict[str, Any]) -> Dict[str, Any]:
    """The two half-antibodies must be related by C2 symmetry.

    IgG1 pairs two identical heavy chains and two identical light chains, so a
    correct model superposes them almost exactly. Reported separately for the
    heavy chains, the light chains and the assembled half, so a failure can be
    localised: if the domains superpose but the half does not, the problem is
    domain placement rather than the domains themselves.
    """
    heavy, light = composition["heavy"], composition["light"]
    if len(heavy) != 2 or len(light) != 2:
        return dict(ok=False, reason="not exactly two heavy and two light chains")

    r_h = _rmsd(_ca_map(model, heavy[0]), _ca_map(model, heavy[1]))
    r_l = _rmsd(_ca_map(model, light[0]), _ca_map(model, light[1]))

    ha, hb = _ca_map(model, heavy[0]), _ca_map(model, heavy[1])
    la, lb = _ca_map(model, light[0]), _ca_map(model, light[1])
    common_h, common_l = sorted(set(ha) & set(hb)), sorted(set(la) & set(lb))
    if common_h and common_l:
        sup = Superimposer()
        sup.set_atoms([ha[i] for i in common_h], [hb[i] for i in common_h])
        rot, tran = sup.rotran
        half = float(np.sqrt(np.mean([
            float(np.sum((np.asarray(lb[i].coord) @ rot + tran
                          - np.asarray(la[i].coord)) ** 2))
            for i in common_l])))
    else:
        half = None

    values = [v for v in (r_h, r_l, half) if v is not None]
    worst = max(values) if values else None
    return dict(heavy_rmsd_A=round(r_h, 3) if r_h is not None else None,
                light_rmsd_A=round(r_l, 3) if r_l is not None else None,
                half_rmsd_A=round(half, 3) if half is not None else None,
                worst_A=round(worst, 3) if worst is not None else None,
                limit_A=SYMMETRY_MAX_A,
                ok=bool(worst is not None and worst <= SYMMETRY_MAX_A))


def check_clashes(model) -> Dict[str, Any]:
    """Count non-bonded heavy-atom contacts closer than CLASH_DIST_A.

    Excluded: atoms in the same residue or in sequence-adjacent residues
    (bonded by construction), and sulfur-sulfur contacts, which are the real
    disulfide bonds being detected rather than an error.
    """
    from scipy.spatial import cKDTree

    atoms = []
    for ch in model:
        for r in ch:
            if not is_aa(r, standard=True):
                continue
            conf = _confidence_of(r)
            for a in r:
                if a.element == "H":
                    continue
                atoms.append((a.coord, ch.id, r.id[1], a.element, conf))
    if not atoms:
        return dict(n_clash=0, ok=False, reason="no heavy atoms found")

    X = np.array([t[0] for t in atoms], float)
    pairs = cKDTree(X).query_pairs(CLASH_DIST_A, output_type="ndarray")

    high_conf = low_conf = 0
    for i, j in pairs:
        ci, ri, ei = atoms[i][1], atoms[i][2], atoms[i][3]
        cj, rj, ej = atoms[j][1], atoms[j][2], atoms[j][3]
        if ci == cj and abs(ri - rj) <= 1:
            continue
        if ei == "S" and ej == "S":
            continue
        confs = [c for c in (atoms[i][4], atoms[j][4]) if c is not None]
        if not confs or min(confs) >= 70.0:
            high_conf += 1
        else:
            low_conf += 1

    n = high_conf + low_conf
    return dict(n_clash=n, in_high_confidence=high_conf,
                in_low_confidence=low_conf, dist_A=CLASH_DIST_A,
                limit=CLASH_MAX, ok=bool(n <= CLASH_MAX))


def check_static_exposure(model, topology: Dict[str, Any]) -> Dict[str, Any]:
    """How many conjugatable cysteines look buried in this single conformer.

    Advisory only, deliberately. A static conformer is one snapshot, and this
    project's own evidence shows a bonded MD run rescues sites judged buried
    here -- the structure the model is calibrated on has one such site. A
    non-zero count is normal; what matters is which sites are affected.
    """
    from .sasa import OONS_RADII, shrake_rupley

    atoms, index = [], {}
    for ch in model:
        for r in ch:
            if not is_aa(r, standard=True):
                continue
            for a in r:
                if a.element == "H":
                    continue
                index[(ch.id, r.id[1], a.get_id())] = len(atoms)
                atoms.append(a)
    coords = np.array([a.coord for a in atoms], float)
    radii = np.array([OONS_RADII.get(a.element, 1.70) for a in atoms], float)
    sasa = shrake_rupley(coords, radii, probe=SASA_PROBE, n_points=SASA_N_POINTS)

    per_site, buried = {}, []
    for s in topology.get("conjugation_sites", []):
        key = (s["chain"], s["res_id"], "SG")
        if key not in index:
            continue
        v = float(sasa[index[key]])
        per_site[s["site"]] = round(v, 2)
        if v < SASA_FLOOR_A2:
            buried.append(s["site"])
    return dict(per_site_A2=per_site, floor_A2=SASA_FLOOR_A2,
                n_below_floor=len(buried), buried_sites=buried)


# ------------------------------------------------------------- driver ------
def _as_pdb(path: str) -> Tuple[str, Optional[str]]:
    """Return a path seqqc can read, plus a temp file to clean up.

    seqqc reads PDB text directly, so an mmCIF input has to be converted first.
    AlphaFold Server and AlphaFold DB both emit mmCIF, so this is the normal
    case rather than an edge case.
    """
    if not path.lower().endswith((".cif", ".mmcif")):
        return path, None
    import tempfile

    from Bio.PDB import PDBIO

    io = PDBIO()
    io.set_structure(load_model(path).get_parent())
    fd, tmp = tempfile.mkstemp(suffix=".pdb")
    os.close(fd)
    io.save(tmp)
    return tmp, tmp


def assess(path: str, with_exposure: bool = True) -> Dict[str, Any]:
    """Run every check on one structure file and return the verdict."""
    readable, tmp = _as_pdb(path)
    try:
        from . import seqqc
        topology = seqqc.assess(readable)
    except Exception as exc:                                   # noqa: BLE001
        return dict(path=path, file=os.path.basename(path), ok=False,
                    verdict="FAIL", fatal=f"topology read failed: {exc}",
                    reasons=[f"topology read failed: {exc}"])
    try:
        return _assess_loaded(path, topology, with_exposure)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def _assess_loaded(path: str, topology: Dict[str, Any],
                   with_exposure: bool) -> Dict[str, Any]:
    model = load_model(path)
    chains = sorted(c.id for c in model)

    composition = check_composition(model, chains)
    disulfides = check_disulfides(topology)
    symmetry = check_symmetry(model, composition)
    clashes = check_clashes(model)

    result: Dict[str, Any] = dict(
        path=path, file=os.path.basename(path),
        n_atoms=sum(1 for ch in model for r in ch for a in r if a.element != "H"),
        chains=chains, composition=composition, disulfides=disulfides,
        symmetry=symmetry, clashes=clashes)

    reasons = []
    if not composition["ok"]:
        reasons.append("composition: expected 2 heavy + 2 light chains of equal length")
    if not disulfides["ok"]:
        reasons.append(f"disulfide: no interchain bond found, or one beyond the "
                       f"{SS_CUTOFF} A cutoff")
    if not symmetry["ok"]:
        reasons.append(f"symmetry: half-antibody superposition "
                       f"{symmetry.get('worst_A')} A exceeds {SYMMETRY_MAX_A} A")
    if not clashes["ok"]:
        reasons.append(f"clash: {clashes['n_clash']} non-bonded contacts closer than "
                       f"{CLASH_DIST_A} A (limit {CLASH_MAX})")

    if with_exposure:
        exposure = check_static_exposure(model, topology)
        result["exposure"] = exposure
        if exposure["n_below_floor"]:
            result["warnings"] = [
                f"exposure: {exposure['n_below_floor']} site(s) below the "
                f"{SASA_FLOOR_A2} A^2 floor in this single conformer: "
                f"{', '.join(exposure['buried_sites'])}. Confirm with MD before use."]

    result["ok"] = not reasons
    result["reasons"] = reasons
    result["verdict"] = "PASS" if not reasons else "FAIL"
    return result


def report(results: List[Dict[str, Any]]) -> str:
    """Human-readable comparison table."""
    head = (f"{'structure':<38}{'verdict':>8}{'half symmetry':>16}"
            f"{'interchain bonds':>18}{'clashes':>9}")
    lines = [head, "-" * len(head)]
    for r in results:
        if r.get("fatal"):
            lines.append(f"{r.get('file', '?'):<38}{'FAIL':>8}   {r['fatal']}")
            continue
        worst = r["symmetry"].get("worst_A")
        worst = f"{worst:.2f} A" if worst is not None else "n/a"
        lines.append(f"{r['file']:<38}{r['verdict']:>8}{worst:>16}"
                     f"{r['disulfides']['n_interchain']:>18}"
                     f"{r['clashes']['n_clash']:>9}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="QC an antibody structure before it enters the DAR model.")
    ap.add_argument("files", nargs="+", help="PDB or mmCIF structure files")
    ap.add_argument("--json", dest="json_out", help="write the full result here")
    ap.add_argument("--no-exposure", action="store_true",
                    help="skip the per-site accessibility measurement")
    args = ap.parse_args(argv)

    results = [assess(p, with_exposure=not args.no_exposure) for p in args.files]
    print(report(results))
    for r in results:
        if r["verdict"] == "FAIL":
            print(f"\n{r['file']}: rejected")
            for reason in r["reasons"]:
                print(f"  - {reason}")
        for w in r.get("warnings", []):
            print(f"\n{r['file']}: warning\n  - {w}")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print(f"\nwritten -> {args.json_out}")
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
