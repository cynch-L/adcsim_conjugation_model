"""Pin C91 and collapse a placed payload onto the antibody surface.

This is NOT a new vacuum ETKDG library. Seeds are already-placed DAR1 poses.
Only linker torsions distal to the maleimide are moved; the reactive carbon
stays at the covalent S–C position.

Compactness is a continuous range (extended → folded), not a discrete count
of extra conformers.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from rdkit import Chem
from rdkit.Chem.Lipinski import RotatableBondSmarts
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .clash import protein_clash_ok
from .geometry import _dist_from, _graph, _side_atoms
from .io import get_sg, heavy_atoms, parse_site


def linker_rotatable_bonds(mol, reactive_atom: int, min_dist: int = 3, max_dist: int = 14) -> list[tuple[int, int]]:
    """Rotatable bonds on the linker, far enough from C91 that rotating them
    cannot move the covalent carbon."""
    g = _graph(mol)
    dist = _dist_from(g, reactive_atom)
    pairs = mol.GetSubstructMatches(RotatableBondSmarts)
    out = []
    for a, b in pairs:
        md = min(dist.get(a, 99), dist.get(b, 99))
        if min_dist <= md <= max_dist:
            # orient so `b` is the distal atom
            if dist.get(a, 99) < dist.get(b, 99):
                out.append((a, b))
            else:
                out.append((b, a))
    # unique, proximal-first
    seen = set()
    uniq = []
    for a, b in sorted(out, key=lambda x: min(dist.get(x[0], 99), dist.get(x[1], 99))):
        key = (a, b)
        if key not in seen:
            seen.add(key)
            uniq.append(key)
    return uniq


# The cache key must be derived from molecular *content*, never from id(mol):
# once a mol is garbage-collected its id can be reused by a new object, and the
# cache would then hand back another molecule's pair list — silently wrong.
_SELF_PAIRS: dict[tuple, list[tuple[int, int]]] = {}


def _self_pairs(mol, heavy: list[int]) -> list[tuple[int, int]]:
    try:
        from rdkit.Chem import MolToSmiles
        key: tuple = (MolToSmiles(mol), len(heavy))
    except Exception:
        key = None          # no fingerprint -> recompute; slow beats wrong
    if key is not None and key in _SELF_PAIRS:
        return _SELF_PAIRS[key]
    g = _graph(mol)
    idx = {atom: i for i, atom in enumerate(heavy)}
    pairs: list[tuple[int, int]] = []
    for i, ai in enumerate(heavy):
        dist = _dist_from(g, ai)
        for aj, dj in dist.items():
            if dj < 4 or aj not in idx:
                continue
            j = idx[aj]
            if j <= i:
                continue
            pairs.append((i, j))
    if key is not None:
        _SELF_PAIRS[key] = pairs
    return pairs


def _self_clash_ok(xyz: np.ndarray, heavy: list[int], mol, min_dist: float = 1.6) -> bool:
    """Reject folded poses that thread the payload through itself."""
    pairs = _self_pairs(mol, heavy)
    if not pairs:
        return True
    ij = np.asarray(pairs, dtype=int)
    d = np.linalg.norm(xyz[ij[:, 0]] - xyz[ij[:, 1]], axis=1)
    return bool(d.min() >= min_dist)


def rotate_heavy(
    xyz: np.ndarray,
    heavy: list[int],
    mol,
    proximal: int,
    distal: int,
    angle_deg: float,
) -> np.ndarray:
    """Rotate heavy atoms on the distal side around the proximal→distal bond."""
    idx = {atom: i for i, atom in enumerate(heavy)}
    if proximal not in idx or distal not in idx:
        return xyz
    g = _graph(mol)
    moved = [a for a in _side_atoms(g, proximal, distal) if a in idx]
    if not moved:
        return xyz
    origin = xyz[idx[proximal]]
    axis = xyz[idx[distal]] - origin
    nrm = np.linalg.norm(axis)
    if nrm < 1e-8:
        return xyz
    rot = Rotation.from_rotvec(np.deg2rad(angle_deg) * (axis / nrm))
    out = xyz.copy()
    rows = [idx[a] for a in moved]
    out[rows] = rot.apply(out[rows] - origin) + origin
    return out


def compactness(xyz: np.ndarray, anchor: np.ndarray) -> dict:
    com = xyz.mean(axis=0)
    rg = float(np.sqrt(((xyz - com) ** 2).sum(axis=1).mean()))
    span = float(np.max(np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)))
    reach = float(np.linalg.norm(xyz - anchor, axis=1).max())
    return {"rg": rg, "span": span, "reach": reach, "com": com}


def surface_contact(xyz: np.ndarray, protein_tree: cKDTree, lo: float = 2.4, hi: float = 4.5) -> int:
    d, _ = protein_tree.query(xyz, k=1)
    return int(np.sum((d >= lo) & (d <= hi)))


def collapse_pose(
    pose: dict,
    protein_xyz: np.ndarray,
    sg: np.ndarray,
    chemistry: dict,
    qc: dict,
    n_random: int = 24,
    angles: tuple[float, ...] = (-120.0, -90.0, -60.0, -30.0, 30.0, 60.0, 90.0, 120.0),
) -> list[dict]:
    """Greedy + random torsion walks that keep C91 fixed and stay clash-free.

    Returns a list of folded poses including the original (compactness_bin=0).
    """
    mol = pose["payload"]
    heavy = list(pose["heavy_atom_indices"])
    xyz0 = np.asarray(pose["coords"], dtype=float)
    reactive = int(chemistry["reactive_atom"])
    cutoff = float(qc["protein_lp_min"])
    tree = cKDTree(protein_xyz)
    bonds = linker_rotatable_bonds(mol, reactive)[:18]
    try:
        anchor_row = heavy.index(reactive)
    except ValueError:
        return []

    def pack(xyz, source: str, bin_: int) -> dict | None:
        ok, mind = protein_clash_ok(xyz, protein_xyz, cutoff)
        if not ok:
            return None
        if not _self_clash_ok(xyz, heavy, mol):
            return None
        # C91 must not have drifted
        if float(np.linalg.norm(xyz[anchor_row] - xyz0[anchor_row])) > 0.05:
            return None
        c = compactness(xyz, sg)
        contact = surface_contact(xyz, tree)
        out = dict(pose)
        out["coords"] = xyz
        out["min_dist"] = float(mind)
        out["clashes"] = 0
        out["fold_source"] = source
        out["compactness_bin"] = int(bin_)
        out["rg"] = c["rg"]
        out["span"] = c["span"]
        out["reach"] = c["reach"]
        out["surface_contact"] = int(contact)
        # higher = more folded onto the Fc without clashing
        out["fold_score"] = float(-c["reach"] + 0.15 * contact + 0.5 * mind)
        return out

    orig_c = compactness(xyz0, sg)
    seed = pack(xyz0, "extended", 0)
    if seed is None:
        return []
    found = [seed]
    seen: list[np.ndarray] = [xyz0]

    def novel(xyz: np.ndarray, tol: float = 1.2) -> bool:
        for old in seen:
            if float(np.sqrt(((xyz - old) ** 2).sum(axis=1).mean())) < tol:
                return False
        return True

    # greedy sequential collapse: at each linker bond pick the angle that
        # shortens reach the most, if still clash-free
    xyz = xyz0.copy()
    current_reach = compactness(xyz, sg)["reach"]
    for step, (a, b) in enumerate(bonds, start=1):
        best = None
        for ang in angles:
            trial = rotate_heavy(xyz, heavy, mol, a, b, ang)
            item = pack(trial, f"greedy_{step}", min(4, 1 + step // 2))
            if item is None:
                continue
            if item["reach"] < current_reach - 0.3:
                if best is None or item["reach"] < best["reach"]:
                    best = item
        if best is not None:
            if novel(best["coords"]):
                found.append(best)
                seen.append(best["coords"])
            xyz = best["coords"]
            current_reach = best["reach"]

    rng = np.random.default_rng(91)
    n_bonds = len(bonds)
    for k in range(n_random):
        xyz = xyz0.copy()
        # Bias toward the first 8 (near-C91) bonds so the chain can U-turn.
        n_move = int(rng.integers(4, max(5, min(10, n_bonds))))
        weights = np.array([3.0 if i < 8 else 1.0 for i in range(n_bonds)], dtype=float)
        weights /= weights.sum()
        chosen_idx = rng.choice(n_bonds, size=min(n_move, n_bonds), replace=False, p=weights)
        for i in chosen_idx:
            a, b = bonds[int(i)]
            ang = float(rng.choice(angles))
            xyz = rotate_heavy(xyz, heavy, mol, a, b, ang)
        item = pack(xyz, f"random_{k}", 3)
        if item is None or not novel(item["coords"]):
            continue
        found.append(item)
        seen.append(item["coords"])

    found.sort(key=lambda p: p["fold_score"], reverse=True)
    return found


def collapse_library(
    library: dict,
    model,
    chemistry: dict,
    qc: dict,
    max_seeds_per_site: int | None = 40,
    n_random: int = 16,
) -> dict:
    folded: dict[str, list] = {}
    for site, poses in library.items():
        chain, res = parse_site(site)
        sg = get_sg(model, site)
        _, prot = heavy_atoms(model, exclude_sg=(chain, res))
        seeds = sorted(poses, key=lambda p: p.get("min_dist", 0), reverse=True)
        if max_seeds_per_site:
            seeds = seeds[: max_seeds_per_site]
        acc = []
        for pose in seeds:
            acc.extend(
                collapse_pose(
                    pose,
                    prot,
                    sg,
                    chemistry,
                    qc,
                    n_random=n_random,
                )
            )
        # keep extended + best folded, cap
        acc.sort(key=lambda p: (p.get("compactness_bin", 0) == 0, p.get("fold_score", 0)), reverse=True)
        folded[site] = acc
    return folded
