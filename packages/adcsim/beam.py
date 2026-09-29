from __future__ import annotations

import numpy as np

from .clash import lp_lp_ok
from .io import get_sg


def bond_distance(sg: np.ndarray, pose: dict, reactive_atom: int) -> float:
    idx = pose["heavy_atom_indices"].index(reactive_atom)
    return float(np.linalg.norm(pose["coords"][idx] - sg))


def expand(
    states: list[dict],
    library: dict,
    model,
    chemistry: dict,
    qc: dict,
    max_states: int,
) -> list[dict]:
    reactive = int(chemistry["reactive_atom"])
    bond_min = float(chemistry["bond_min"])
    bond_max = float(chemistry["bond_max"])
    lp_cut = float(qc["lp_lp_min"])
    new = []
    for state in states:
        used = set(state["sites"])
        for site, poses in library.items():
            if site in used:
                continue
            sg = get_sg(model, site)
            for pose in poses:
                if pose.get("clashes", 0) != 0:
                    continue
                try:
                    bd = bond_distance(sg, pose, reactive)
                except ValueError:
                    continue
                if bd < bond_min or bd > bond_max:
                    continue
                lp_ok = True
                min_lp = float("inf")
                for old in state["poses"]:
                    ok, d = lp_lp_ok(pose["coords"], old["coords"], lp_cut)
                    min_lp = min(min_lp, d)
                    if not ok:
                        lp_ok = False
                        break
                if not lp_ok:
                    continue
                score = min(float(state["score"]), float(pose["min_dist"]), float(min_lp))
                new.append(
                    {
                        "sites": state["sites"] + [site],
                        "poses": state["poses"] + [pose],
                        "score": score,
                        "lp_min": float(min_lp),
                    }
                )
    new.sort(key=lambda x: x["score"], reverse=True)
    if max_states:
        new = new[:max_states]
    return new


def seed_dar1(library: dict) -> list[dict]:
    states = []
    for site, poses in library.items():
        for pose in poses:
            if pose.get("clashes", 0) != 0:
                continue
            states.append(
                {
                    "sites": [site],
                    "poses": [pose],
                    "score": float(pose["min_dist"]),
                    "lp_min": float("inf"),
                }
            )
    states.sort(key=lambda x: x["score"], reverse=True)
    return states


def funnel(
    library: dict,
    model,
    chemistry: dict,
    qc: dict,
    search: dict,
    max_dar: int,
) -> dict[int, list[dict]]:
    """Expand DAR1 → max_dar. Stop when a level has zero valid states."""
    max_states = int(search.get("max_states", 80))
    max_dar1 = int(search.get("max_dar1_seeds", 200))
    dar = {1: seed_dar1(library)[:max_dar1]}
    for n in range(2, max_dar + 1):
        if not dar.get(n - 1):
            dar[n] = []
            break
        dar[n] = expand(dar[n - 1], library, model, chemistry, qc, max_states)
        if not dar[n]:
            break
    return dar
