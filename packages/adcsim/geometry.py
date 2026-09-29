from __future__ import annotations

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation


# --------------------------------------------------------------------------
# Molecular-graph helpers shared by fold.py (surface folding) and solvation.py
# (hydrophobic collapse): adjacency list + BFS topological distance + the atom
# set on the distal side of a bond.  Both modules used to carry their own
# byte-identical copy; changing one and missing the other produced different
# rotatable-bond sets.  Single definition lives here.
# --------------------------------------------------------------------------

def _graph(mol) -> dict[int, list[int]]:
    """Atom-index adjacency list of an RDKit molecule."""
    g: dict[int, list[int]] = {a.GetIdx(): [] for a in mol.GetAtoms()}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        g[i].append(j)
        g[j].append(i)
    return g


def _dist_from(g: dict[int, list[int]], src: int) -> dict[int, int]:
    """BFS topological distance (in bonds) from `src` to every reachable atom."""
    dist = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in g[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _side_atoms(g: dict[int, list[int]], proximal: int, distal: int) -> list[int]:
    """Atoms on the distal side of bond proximal–distal (excluding proximal)."""
    seen = {proximal}
    out: list[int] = []
    q = deque([distal])
    seen.add(distal)
    while q:
        u = q.popleft()
        out.append(u)
        for v in g[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return out


def sphere_grid(step_deg: float = 15.0) -> np.ndarray:
    directions = []
    for theta in np.arange(0, 181, step_deg):
        for phi in np.arange(0, 360, step_deg):
            t = np.deg2rad(theta)
            p = np.deg2rad(phi)
            directions.append(
                [
                    np.sin(t) * np.cos(p),
                    np.sin(t) * np.sin(p),
                    np.cos(t),
                ]
            )
    vecs = np.asarray(directions, dtype=float)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-12, None)


def antibody_front_normal(ca_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = ca_coords.mean(axis=0)
    centered = ca_coords - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[0]
    normal = normal / np.linalg.norm(normal)
    return center, normal


def hemisphere(directions: np.ndarray, normal: np.ndarray, sign: float = 1.0) -> np.ndarray:
    dots = directions @ (normal / np.linalg.norm(normal))
    return directions[dots * sign > 0]


def align_and_place(
    xyz: np.ndarray,
    anchor_idx: int,
    exit_idx: int,
    sg: np.ndarray,
    direction: np.ndarray,
    bond: float,
    roll_deg: float,
) -> np.ndarray:
    """Place LP so that atom `anchor_idx` sits at SG + bond * direction,
    and the exit vector (exit - anchor) is aligned to `direction`, then roll."""
    centered = xyz - xyz[anchor_idx]
    exit_vec = xyz[exit_idx] - xyz[anchor_idx]
    n = np.linalg.norm(exit_vec)
    if n < 1e-8:
        raise ValueError("degenerate exit vector")
    exit_vec = exit_vec / n
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    align = Rotation.align_vectors([direction], [exit_vec])[0]
    base = align.apply(centered)
    rolled = Rotation.from_rotvec(np.deg2rad(roll_deg) * direction).apply(base)
    return rolled + (sg + bond * direction)
