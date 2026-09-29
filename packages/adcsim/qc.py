from __future__ import annotations

import numpy as np

from .clash import min_distance
from .io import get_sg, heavy_atoms, parse_site


def measure_state(state: dict, model, chemistry: dict, qc: dict) -> dict:
    reactive = int(chemistry["reactive_atom"])
    rows = []
    protein_mins = []
    for site, pose in zip(state["sites"], state["poses"]):
        chain, res = parse_site(site)
        sg = get_sg(model, site)
        idx = pose["heavy_atom_indices"].index(reactive)
        bond = float(np.linalg.norm(pose["coords"][idx] - sg))
        _, prot = heavy_atoms(model, exclude_sg=(chain, res))
        pmin = min_distance(pose["coords"], prot)
        protein_mins.append(pmin)
        rows.append(
            {
                "site": site,
                "side": pose.get("side", ""),
                "bond": bond,
                "protein_min": pmin,
                "roll": pose.get("roll"),
            }
        )
    lp_mins = []
    poses = state["poses"]
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            lp_mins.append(min_distance(poses[i]["coords"], poses[j]["coords"]))
    lp_min = min(lp_mins) if lp_mins else float("inf")
    prot_min = min(protein_mins) if protein_mins else float("inf")
    bond_ok = all(
        chemistry["bond_min"] <= r["bond"] <= chemistry["bond_max"] for r in rows
    )
    clash_ok = prot_min >= qc["protein_lp_min"] and lp_min >= qc["lp_lp_min"]
    return {
        "sites": list(state["sites"]),
        "n_lp": len(state["sites"]),
        "score": float(state["score"]),
        "protein_min": float(prot_min),
        "lp_min": float(lp_min) if np.isfinite(lp_min) else None,
        "bond_ok": bond_ok,
        "clash_ok": clash_ok,
        "pass": bond_ok and clash_ok,
        "per_site": rows,
    }
