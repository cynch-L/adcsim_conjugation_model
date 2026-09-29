from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def min_distance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    tree = cKDTree(b)
    d, _ = tree.query(a, k=1)
    return float(np.min(d))


def protein_clash_ok(
    lp_xyz: np.ndarray,
    protein_xyz: np.ndarray,
    cutoff: float,
) -> tuple[bool, float]:
    """LP vs protein heavy atoms. protein_xyz should already exclude the
    covalent Cys-SG (1,2 neighbour of the new C–S bond).
    Returns (pass, min_distance).
    """
    d = min_distance(lp_xyz, protein_xyz)
    return d >= cutoff, d


def lp_lp_ok(a: np.ndarray, b: np.ndarray, cutoff: float) -> tuple[bool, float]:
    d = min_distance(a, b)
    return d >= cutoff, d


def covalent_neighbours_1_2_1_3(mol, reactive_atom: int) -> set[int]:
    """Heavy-atom indices that are 1,2 or 1,3 neighbours of the reactive carbon.

    These contacts to Cys CB/CA/N/C after placement are chemically expected
    and must not be counted as clashes. The SG itself is already excluded
    from protein_xyz by the caller.
    """
    atom = mol.GetAtomWithIdx(reactive_atom)
    n1 = {n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() > 1}
    n2 = set()
    for idx in n1:
        for n in mol.GetAtomWithIdx(idx).GetNeighbors():
            if n.GetAtomicNum() > 1 and n.GetIdx() != reactive_atom:
                n2.add(n.GetIdx())
    return n1 | n2


def hydrogen_false_collision_count(
    protein_xyz: np.ndarray,
    lp_heavy_xyz: np.ndarray,
    mol,
    cutoff: float,
    h_ext: float = 1.1,
) -> int:
    """Count LP hydrogen atoms that would *falsely* register as a clash.

    Collision detection uses only heavy atoms on purpose: a hydrogen rotates
    freely, so counting it would flag a perfectly fine pose as "clashing". This
    function reports, separately, how many LP hydrogens WOULD fall inside
    `cutoff` of the protein if H were counted.

    A hydrogen can reach at most ~`h_ext` Angstrom closer than the heavy atom it
    is bonded to, so any LP heavy atom within (cutoff + h_ext) of the protein
    drags its attached hydrogens into the false-collision set. We count unique H
    atoms. This never changes the heavy-atom verdict; it only quantifies why H is
    excluded.

    `mol` is the LP RDKit molecule (with explicit H) used to walk heavy->H bonds;
    `lp_heavy_xyz` must be ordered by the molecule's heavy-atom indices.
    """
    if len(lp_heavy_xyz) == 0 or len(protein_xyz) == 0:
        return 0
    heavy_pos = {a.GetIdx(): i for i, a in enumerate(mol.GetAtoms())
                 if a.GetAtomicNum() > 1}
    tree = cKDTree(protein_xyz)
    d, _ = tree.query(lp_heavy_xyz, k=1)
    near = d < (cutoff + h_ext)
    seen, count = set(), 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        parent = atom.GetNeighbors()[0]
        pi = heavy_pos.get(parent.GetIdx())
        if pi is None or not near[pi]:
            continue
        if atom.GetIdx() not in seen:
            seen.add(atom.GetIdx())
            count += 1
    return count
