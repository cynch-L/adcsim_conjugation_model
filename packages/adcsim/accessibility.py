"""Chemical accessibility gate for inter-chain Cys sites.

This is a cheap pre-filter that runs after site definition and before the
DAR1 library. It answers: can a maleimide physically approach this thiol?

Geometry, not kinetics:
- residue / SG SASA (Shrake–Rupley)
- solvent rays from SG (same probe as open-space sampling)
- CB–SG vs ray angle as a Michael-addition approach proxy

A FAIL site is dropped from sampling. WARN sites stay in the pool.
SG SASA = 0 is common for hinge Cys and is NOT a fail by itself.
"""

from __future__ import annotations

import numpy as np
from Bio.PDB import ShrakeRupley  # deprecated: SASA now via adcsim.sasa.compute_sasa

from .geometry import sphere_grid
from .io import get_sg, heavy_atoms, parse_site
from .sampling import open_directions
from .sasa import compute_sasa  # unified SASA core (Biopython adapter)


DEFAULTS = {
    "min_cys_sasa": 5.0,
    # "min_sg_sasa" was here and in config.yaml but nothing ever read it — a knob
    # that looks tunable and is not. Removed from both places (2026-09-22).
    "min_open_rays": 4,
    "attack_angle_min_deg": 70.0,
    "attack_angle_max_deg": 140.0,
    "drop_failed": True,
}


def _residue(model, site: str):
    chain, resnum = parse_site(site)
    return model[chain][(" ", resnum, " ")]


def _cb_sg_vector(residue, sg: np.ndarray) -> np.ndarray | None:
    if "CB" not in residue:
        return None
    cb = np.asarray(residue["CB"].coord, dtype=float)
    vec = sg - cb
    n = np.linalg.norm(vec)
    if n < 1e-8:
        return None
    return vec / n


def _angles_deg(ref: np.ndarray, rays: np.ndarray) -> np.ndarray:
    if len(rays) == 0:
        return np.zeros(0, dtype=float)
    dots = np.clip(rays @ ref, -1.0, 1.0)
    return np.rad2deg(np.arccos(dots))


def score_site(
    model,
    site: str,
    sampling: dict,
    qc: dict,
    access: dict,
    sasa_row: dict | None = None,
) -> dict:
    residue = _residue(model, site)
    chain, res = parse_site(site)
    sg = get_sg(model, site)
    _, xyz_ex = heavy_atoms(model, exclude_sg=(chain, res))
    grid = sphere_grid(sampling.get("direction_step_deg", 15))
    rays = open_directions(
        sg,
        xyz_ex,
        grid,
        sampling.get("ray_start", 2.0),
        sampling.get("ray_end", 10.0),
        sampling.get("ray_step", 0.5),
        qc.get("probe_cutoff", 2.0),
    )
    n_rays = int(len(rays))

    if sasa_row is None:
        compute_sasa(model)  # unified core: writes atom.sasa on every Atom
        cys_sasa = float(sum(getattr(atom, "sasa", 0.0) for atom in residue))
        sg_sasa = float(getattr(residue["SG"], "sasa", 0.0) or 0.0)
        neighbors_4A = None
    else:
        cys_sasa = float(sasa_row["cys_sasa"])
        sg_sasa = float(sasa_row["sg_sasa"])
        neighbors_4A = int(sasa_row.get("neighbors_4A", 0))

    cb_sg = _cb_sg_vector(residue, sg)
    amin = float(access.get("attack_angle_min_deg", DEFAULTS["attack_angle_min_deg"]))
    amax = float(access.get("attack_angle_max_deg", DEFAULTS["attack_angle_max_deg"]))
    if cb_sg is None or n_rays == 0:
        n_attack = 0
        best_angle = None
    else:
        angs = _angles_deg(cb_sg, rays)
        mask = (angs >= amin) & (angs <= amax)
        n_attack = int(np.count_nonzero(mask))
        best_angle = float(angs[mask].min()) if n_attack else float(angs.min())

    min_cys = float(access.get("min_cys_sasa", DEFAULTS["min_cys_sasa"]))
    min_rays = int(access.get("min_open_rays", DEFAULTS["min_open_rays"]))

    buried = cys_sasa < min_cys and n_rays == 0
    poor_approach = n_rays > 0 and n_attack == 0
    if buried:
        verdict = "FAIL"
        reason = "no solvent ray and residue SASA below cutoff"
    elif n_rays < min_rays and cys_sasa < min_cys:
        verdict = "FAIL"
        reason = f"only {n_rays} open rays and Cys-SASA {cys_sasa:.1f} Å²"
    elif poor_approach:
        verdict = "WARN"
        reason = "open rays exist but none sit in the CB–SG attack cone"
    elif n_rays < min_rays:
        verdict = "WARN"
        reason = f"few open rays ({n_rays})"
    else:
        verdict = "PASS"
        reason = "solvent-accessible thiol with an open approach cone"

    return {
        "site": site,
        "cys_sasa": cys_sasa,
        "sg_sasa": sg_sasa,
        "neighbors_4A": neighbors_4A,
        "n_open_rays": n_rays,
        "n_attack_rays": n_attack,
        "best_attack_angle_deg": best_angle,
        "verdict": verdict,
        "reason": reason,
    }


def score_sites(
    model,
    sites: list[str],
    sampling: dict,
    qc: dict,
    access: dict | None = None,
    sasa_rows: list[dict] | None = None,
) -> list[dict]:
    access = {**DEFAULTS, **(access or {})}
    sasa_map = {r["site"]: r for r in (sasa_rows or [])}
    return [
        score_site(model, site, sampling, qc, access, sasa_map.get(site))
        for site in sites
    ]


def filter_sites(rows: list[dict], drop_failed: bool = True) -> tuple[list[str], list[str]]:
    keep, drop = [], []
    for r in rows:
        if drop_failed and r["verdict"] == "FAIL":
            drop.append(r["site"])
        else:
            keep.append(r["site"])
    return keep, drop
