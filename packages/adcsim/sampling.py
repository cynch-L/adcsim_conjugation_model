from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .clash import protein_clash_ok
from .geometry import align_and_place, hemisphere, sphere_grid
from .io import ca_coords, get_sg, heavy_atoms, mol_heavy_xyz, parse_site
from .geometry import antibody_front_normal


def open_directions(
    sg: np.ndarray,
    protein_xyz: np.ndarray,
    directions: np.ndarray,
    ray_start: float,
    ray_end: float,
    ray_step: float,
    probe_cutoff: float,
) -> np.ndarray:
    tree = cKDTree(protein_xyz)
    free = []
    rs = np.arange(ray_start, ray_end + 1e-6, ray_step)
    for d in directions:
        ok = True
        for r in rs:
            dist, _ = tree.query(sg + r * d, k=1)
            if dist < probe_cutoff:
                ok = False
                break
        if ok:
            free.append(d)
    return np.asarray(free, dtype=float) if free else np.zeros((0, 3))


def build_open_space(model, sites: list[str], sampling: dict, qc: dict) -> dict:
    _, protein_xyz_all = heavy_atoms(model)
    center, front_normal = antibody_front_normal(ca_coords(model))
    grid = sphere_grid(sampling["direction_step_deg"])
    front_dirs = hemisphere(grid, front_normal, +1.0)
    back_dirs = hemisphere(grid, front_normal, -1.0)

    out = {}
    for site in sites:
        chain, res = parse_site(site)
        sg = get_sg(model, site)
        _, xyz_ex = heavy_atoms(model, exclude_sg=(chain, res))
        front = open_directions(
            sg,
            xyz_ex,
            front_dirs,
            sampling["ray_start"],
            sampling["ray_end"],
            sampling["ray_step"],
            qc["probe_cutoff"],
        )
        back = open_directions(
            sg,
            xyz_ex,
            back_dirs,
            sampling["ray_start"],
            sampling["ray_end"],
            sampling["ray_step"],
            qc["probe_cutoff"],
        )
        out[site] = {
            "sg": sg,
            "front": front,
            "back": back,
            "front_normal": front_normal,
        }
    return out


def sample_site_library(
    model,
    mols: list,
    site: str,
    directions: np.ndarray,
    side: str,
    chemistry: dict,
    sampling: dict,
    qc: dict,
    max_poses: int | None = None,
) -> list[dict]:
    chain, res = parse_site(site)
    sg = get_sg(model, site)
    _, protein_xyz = heavy_atoms(model, exclude_sg=(chain, res))
    rolls = np.arange(0, 360, sampling["roll_step_deg"])
    if chemistry.get("bond_values"):
        bonds = np.asarray(chemistry["bond_values"], dtype=float)
    else:
        bonds = np.arange(
            chemistry["bond_min"],
            chemistry["bond_max"] + 1e-9,
            chemistry["bond_step"],
        )
    reactive = int(chemistry["reactive_atom"])
    exit_atom = int(chemistry["exit_atom"])
    cutoff = float(qc["protein_lp_min"])

    poses = []
    for mol in mols:
        heavy, xyz = mol_heavy_xyz(mol)
        try:
            anchor_pos = heavy.index(reactive)
            exit_pos = heavy.index(exit_atom)
        except ValueError:
            continue
        for d in directions:
            nrm = np.linalg.norm(d)
            if nrm < 1e-8:
                continue
            d = d / nrm
            for bond in bonds:
                for roll in rolls:
                    placed = align_and_place(
                        xyz, anchor_pos, exit_pos, sg, d, float(bond), float(roll)
                    )
                    ok, mind = protein_clash_ok(placed, protein_xyz, cutoff)
                    if not ok:
                        continue
                    poses.append(
                        {
                            "payload": mol,
                            "direction": np.asarray(d, dtype=float),
                            "roll": int(roll),
                            "bond": float(bond),
                            "coords": placed,
                            "heavy_atom_indices": heavy,
                            "min_dist": float(mind),
                            "clashes": 0,
                            "side": side,
                            "site": site,
                        }
                    )
                    if max_poses is not None and len(poses) >= max_poses:
                        return poses
    return poses


def build_dar1_library(
    model,
    mols: list,
    open_space: dict,
    chemistry: dict,
    sampling: dict,
    qc: dict,
    max_poses_per_site_side: int | None = None,
) -> dict:
    library = {}
    for site, info in open_space.items():
        library[site] = []
        for side, dirs in (("front", info["front"]), ("back", info["back"])):
            if len(dirs) == 0:
                continue
            poses = sample_site_library(
                model,
                mols,
                site,
                dirs,
                side,
                chemistry,
                sampling,
                qc,
                max_poses=max_poses_per_site_side,
            )
            library[site].extend(poses)
    return library
