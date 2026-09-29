"""Solvation-driven fold — explicit hydrophobic collapse proxy.

In solution, a free MMAE collapses due to hydrophobic effect (clogP ~3–4).
This module uses RDKit FreeSASA to quantify non-polar surface area and
drives torsional scans toward minimal exposed hydrophobic area, constrained
by protein/LP clash cutoffs.

Used in polish.py as the first fold step: let each payload "feel" the solvent
BEFORE considering inter-LP crowding.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdFreeSASA
from rdkit.Chem.Lipinski import RotatableBondSmarts

# These three molecular-graph helpers used to be duplicated verbatim in this
# module and in fold.py.  Now imported from geometry; the local copies are gone
# and every call site keeps its original name.
from .geometry import _dist_from, _graph, _side_atoms

# Shared physical constants.  payload_qc.py used to carry its own copies
# (RGAS_R / UM2_PER_CM2) — same values, two places to update.  This is canonical.
RGAS_KCAL_MOL_K = 1.98720425e-3     # kcal/(mol·K) — same source as site_env.R_KCAL
UM2_TO_CM2 = 1e-8                   # 1 µm² = 1e-8 cm²


_OONS_RADII = {6: 1.70, 7: 1.55, 8: 1.52, 1: 1.20, 16: 1.80, 9: 1.47, 17: 1.75, 35: 1.85, 53: 1.98}


def _classify_by_element(mol_h):
    """Assign SASAClassName based on atomic number (OONS-style) — bypasses
    rdFreeSASA.classifyAtoms which fails on LINK-record molecules."""
    for atom in mol_h.GetAtoms():
        num = atom.GetAtomicNum()
        if num == 1:
            atom.SetProp("SASAClassName", "H")
        elif num == 6:
            atom.SetProp("SASAClassName", "C")
        elif num == 7:
            atom.SetProp("SASAClassName", "N")
        elif num == 8:
            atom.SetProp("SASAClassName", "O")
        elif num == 16:
            atom.SetProp("SASAClassName", "S")
        elif num == 9:
            atom.SetProp("SASAClassName", "F")
        elif num == 17:
            atom.SetProp("SASAClassName", "Cl")
        else:
            atom.SetProp("SASAClassName", "C")  # fallback


def sasa(mol_h, radii: dict | None = None) -> tuple[float, float, float]:
    """Return (total, polar, non-polar) SASA in Å².

    .. deprecated::
        The unified SASA engine is :mod:`adcsim.sasa`. New callers (scorecard)
        should use ``adcsim.sasa.sasa_of_rdkit``. This function is retained only
        as the engine behind ``polish.py`` (out of scope for the unification
        refactor); it still uses rdFreeSASA with OONS radii.

    OONS radii, element classification and polar/non-polar convention: defined
    here and in ``adcsim.sasa`` from the same source so the two sides cannot
    drift. Probe radius 1.4 Å (== Bio.PDB ShrakeRupley default).

    NOTE: radii are OONS *atomic-number* keyed, applied per atom index, because
    rdFreeSASA.classifyAtoms() fails on molecules built from LINK records.
    """
    if radii is None:
        radii = {
            atom.GetIdx(): _OONS_RADII.get(atom.GetAtomicNum(), 1.70)
            for atom in mol_h.GetAtoms()
        }
    _classify_by_element(mol_h)
    rdFreeSASA.CalcSASA(mol_h, radii)
    total = polar = npolar = 0.0
    for atom in mol_h.GetAtoms():
        s_i = atom.GetDoubleProp("SASA") if atom.HasProp("SASA") else 0.0
        total += s_i
        cls = atom.GetProp("SASAClassName") if atom.HasProp("SASAClassName") else "C"
        if cls in ("O", "N"):
            polar += s_i
        else:
            npolar += s_i
    return total, polar, npolar


def sasa_metric(xyz: np.ndarray, mol_h) -> float:
    """Non-polar SASA per heavy atom — lower = more hydrophobically collapsed."""
    conf = mol_h.GetConformer()
    for i in range(mol_h.GetNumAtoms()):
        conf.SetAtomPosition(i, (float(xyz[i][0]), float(xyz[i][1]), float(xyz[i][2])))
    _, _, npolar = sasa(mol_h)
    heavy = sum(1 for atom in mol_h.GetAtoms() if atom.GetAtomicNum() > 1)
    return npolar / max(heavy, 1)


def rotate_xyz(
    xyz: np.ndarray,
    g: dict[int, list[int]],
    proximal: int,
    distal: int,
    angle_deg: float,
) -> np.ndarray:
    """Rotate atoms on the distal side of bond proximal→distal."""
    moved = _side_atoms(g, proximal, distal)
    if not moved:
        return xyz
    origin = xyz[proximal]
    axis = xyz[distal] - origin
    nrm = np.linalg.norm(axis)
    if nrm < 1e-8:
        return xyz
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_rotvec(np.deg2rad(angle_deg) * (axis / nrm))
    out = xyz.copy()
    out[moved] = rot.apply(out[moved] - origin) + origin
    return out


def _rotatable_bonds(mol_h, c91_idx: int, min_dist: int = 3, max_dist: int = 20):
    g = _graph(mol_h)
    dist = _dist_from(g, c91_idx)
    matches = mol_h.GetSubstructMatches(RotatableBondSmarts)
    bonds = []
    for a, b in matches:
        md = min(dist.get(a, 99), dist.get(b, 99))
        if min_dist <= md <= max_dist:
            if dist.get(a, 99) < dist.get(b, 99):
                bonds.append((a, b))
            else:
                bonds.append((b, a))
    seen = set()
    uniq = []
    for a, b in sorted(bonds, key=lambda x: dist.get(x[0], 99)):
        key = (a, b)
        if key not in seen:
            seen.add(key)
            uniq.append(key)
    return uniq


def solvation_collapse(
    xyz: np.ndarray,
    mol_h,
    c91_idx: int,
    anchor_idx: int | None = None,
    clash_fn=None,
    n_random: int = 48,
    angles: tuple[float, ...] = np.arange(-150, 180, 30).tolist(),
) -> tuple[np.ndarray, float, float]:
    """Drive one payload toward minimal non-polar SASA.

    Parameters
    ----------
    xyz : (Natoms, 3) starting coords
    mol_h : RDKit Mol with Hs
    c91_idx : index of C91 (will be pinned)
    anchor_idx : optional index to anchor (default = c91_idx)
    clash_fn : callable(xyz) -> bool, returns True if clash-free
    n_random : number of random multi-bond walks
    angles : torsion angles to scan (deg)

    Returns
    -------
    best_xyz, sasa_before, sasa_after
    """
    if anchor_idx is None:
        anchor_idx = c91_idx
    bonds = _rotatable_bonds(mol_h, c91_idx)
    if not bonds:
        return xyz, sasa_metric(xyz, mol_h), sasa_metric(xyz, mol_h)

    g = _graph(mol_h)
    sasa0 = sasa_metric(xyz, mol_h)
    best_xyz = xyz
    best_sasa = sasa0
    rng = np.random.default_rng(91)

    # ---- full-angle brute-force scan per bond (systematic cover) ----
    for (a, b) in bonds:
        for ang in np.arange(-180, 180, 15):
            trial = rotate_xyz(xyz, g, a, b, float(ang))
            if clash_fn is not None and not clash_fn(trial):
                continue
            sasa = sasa_metric(trial, mol_h)
            if sasa < best_sasa - 0.2:
                best_xyz, best_sasa = trial, sasa

    # ---- random multi-bond walks ----
    n_bonds = len(bonds)
    for _ in range(n_random):
        trial = xyz.copy()
        n_move = int(rng.integers(3, max(4, min(12, n_bonds + 1))))
        chosen = rng.choice(n_bonds, size=min(n_move, n_bonds), replace=False)
        for i in chosen:
            a, b = bonds[int(i)]
            ang = float(rng.choice(angles))
            trial = rotate_xyz(trial, g, a, b, ang)
        if clash_fn is not None and not clash_fn(trial):
            continue
        sasa = sasa_metric(trial, mol_h)
        if sasa < best_sasa - 0.2:
            best_xyz, best_sasa = trial, sasa

    # ---- re-clamp C91 to its original position ----
    best_xyz[c91_idx] = xyz[c91_idx]

    return best_xyz, sasa0, best_sasa