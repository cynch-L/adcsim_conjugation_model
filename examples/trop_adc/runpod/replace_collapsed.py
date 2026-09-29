#!/usr/bin/env python3
"""Replace extended LP conformers with collapsed (balled-up) conformers.

The existing ADC_DAR*.pdb carry LP payloads that are extended rods
(span 17-40 A). The solution-realistic ensemble (payload_ensemble_100.sdf)
is 60% collapsed (span <= 20 A, Rg ~6.8 A). This script:
  1. reads an ADC PDB and groups the HETATM LPP atoms per (chain, resSeq);
  2. for every LP instance, finds its C91 (reactive maleimide carbon) and
     C85 (linker exit atom) PDB coordinates;
  3. picks a collapsed conformer (span <= 20 A) from the ensemble whose
     C91-C85 vector is closest to the PDB LP's C91->C85 direction;
  4. rigid-aligns the conformer so C91 lands on the PDB C91 and the
     C85 direction matches (rotate around the SG-C91 axis);
  5. checks clashes against the antibody protein heavy atoms
     (probe 2.0 A, excluding the own Cys SG and its bonded neighbors);
  6. rewrites the PDB with the replaced LP + correct element symbols.

C91-C85 have no hydrogen assignment in the SDF; the atom names in output
are regenerated from RDKit order (C0, C1, ...) as the pipeline does.

Output: <stem>_collapsed.pdb next to source, plus a JSON report.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from rdkit import Chem

# --------------------------------------------------------------------------- #
# conformer metrics
# --------------------------------------------------------------------------- #
def _heavy_xyz(mol, cid: int = 0) -> np.ndarray:
    conf = mol.GetConformer(int(cid))
    heavy = [i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
    return np.asarray([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                        conf.GetAtomPosition(i).z] for i in heavy], dtype=float)


def _span(xyz: np.ndarray) -> float:
    d = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1))
    return float(d.max())


def build_collapsed_pool(sdf_path: str, max_span: float = 20.0):
    """Build span-tiered conformer pools.

    Tiers (each list: (conformer_idx, mol, heavy_xyz, c85_xyz, c91_xyz)):
      collapsed : span <= max_span (default 20 A)  — fully balled up
      mid       : max_span..25 A                    — partially folded
      partly    : 25..30 A                          — linker tucked, MMAE open
      plus      : 30..36 A                          — lightly folded
    The caller tries collapsed first, then falls through to looser tiers so
    that genuinely buried sites (A:CYS223 / D:CYS214, with no room for a
    rigid Rg~6.8 A globule) still get meaningfully folded, not left at
    37-41 A fully-extended.
    """
    mols = [m for m in Chem.SDMolSupplier(sdf_path, removeHs=False) if m is not None]
    tiers = {"collapsed": [], "mid": [], "partly": [], "plus": []}

    def bucket(sp: float) -> str | None:
        if sp <= max_span:
            return "collapsed"
        if sp <= 25.0:
            return "mid"
        if sp <= 30.0:
            return "partly"
        if sp <= 36.0:
            return "plus"
        return None

    for mi, m in enumerate(mols):
        x = _heavy_xyz(m, 0)
        sp = _span(x)
        b = bucket(sp)
        if b is None:
            continue
        c85 = np.asarray(m.GetConformer().GetAtomPosition(85), dtype=float)
        c91 = np.asarray(m.GetConformer().GetAtomPosition(91), dtype=float)
        tiers[b].append((mi, m, x, c85, c91))
    return [tiers["collapsed"], tiers["mid"], tiers["partly"], tiers["plus"]]


def align_to_anchor(c85: np.ndarray, c91: np.ndarray, p85: np.ndarray, p91: np.ndarray) -> Rotation:
    """Rotation aligning conformer vector (c91->c85) onto PDB vector (p91->p85),
    keeping C91 fixed (translation is applied by the caller)."""
    u = c91 - c85
    v = p91 - p85
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    return Rotation.align_vectors(np.atleast_2d(v), np.atleast_2d(u))[0]


def coordinates_for_conformer(m: Chem.Mol, rot: Rotation, c85: np.ndarray,
                              p91: np.ndarray) -> np.ndarray:
    """Apply rotation + translation so C91 maps to p91. Return heavy xyz."""
    heavy = [i for i in range(m.GetNumAtoms()) if m.GetAtomWithIdx(i).GetAtomicNum() > 1]
    x = _heavy_xyz(m, 0)
    c91 = np.asarray(m.GetConformer().GetAtomPosition(91), dtype=float)
    xr = rot.apply(x - c91) + p91
    return xr


def check_clash(xyz: np.ndarray, protein: np.ndarray, cutoff: float = 2.0,
                exclude_serial: set[int] | None = None) -> tuple[bool, float]:
    """Clash check vs protein heavy atoms. Returns (ok, min_dist)."""
    tree = cKDTree(protein)
    d, _ = tree.query(xyz, k=1)
    return bool((d >= cutoff).all()), float(d.min())


# --------------------------------------------------------------------------- #
# PDB parsing / writing
# --------------------------------------------------------------------------- #
def parse_hetatm_lines(pdb_path: Path):
    """Return (lines, total_serial) minus CONECT/END."""
    lines = []
    for ln in pdb_path.read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            lines.append(ln)
    serials = [int(ln[6:11]) for ln in lines if ln.startswith(("ATOM", "HETATM"))]
    max_serial = max(serials)
    return lines, max_serial


def _atom_element(ln: str) -> str:
    return ln[76:78].strip() or (ln[12:16].strip()[0] if ln[12:16].strip() else "C")


def group_lpp(lines):
    """Group HETATM LPP records by (chain, resSeq): list of (line, coord)."""
    groups = defaultdict(list)
    for ln in lines:
        if not ln.startswith("HETATM"):
            continue
        if ln[17:20].strip() != "LPP":
            continue
        ch = ln[21]
        seq = ln[22:26].strip()
        x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
        groups[(ch, seq)].append((ln, np.array([x, y, z])))
    return groups


def sorted_heavy(ln: str) -> str:
    el = _atom_element(ln)
    name = ln[12:16].strip()
    return f"{el}_{name}"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def process_pdb(pdb_path: Path, pools, protein_cutoff: float = 1.7,
                extra_cutoff: float = 1.7, max_tries: int = 60,
                n_roll: int = 12, n_exit_dirs: int = 18,
                tier_names=("collapsed", "mid", "partly", "plus")) -> dict:
    lines, _ = parse_hetatm_lines(pdb_path)

    # --- all Cys SG (chain, resnum) -> coord, plus full residue atom coords ---
    cys_residues = {}   # (chain, resnum) -> list of atom coords
    sg_xyz = {}         # (chain, resnum) -> SG coord
    for ln in lines:
        if not ln.startswith("ATOM"):
            continue
        if ln[17:20].strip() != "CYS":
            continue
        key = (ln[21], int(ln[22:26].strip()))
        coord = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        cys_residues.setdefault(key, []).append(coord)
        if ln[12:16].strip() == "SG":
            sg_xyz[key] = coord
    sg_keys = list(sg_xyz.keys())

    groups = group_lpp(lines)
    report = {}
    next_serial = max(int(ln[6:11]) for ln in lines) + 1

    # heavy order in SDF == heavy order in PDB (ascending idx)
    pool_mol = next((p[0][1] for p in pools if p), None)
    if pool_mol is None:
        raise SystemExit("no conformers at all")
    heavy = [i for i in range(pool_mol.GetNumAtoms()) if pool_mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
    heavy_idx85 = heavy.index(85)
    heavy_idx91 = heavy.index(91)

    for (ch, seq), recs in sorted(groups.items()):
        # heavy coords of this LP instance, in SDF-heavy order
        lpp_heavy = [c for _, c in recs]   # group_lpp returns in line order = heavy order
        if len(lpp_heavy) < 94:
            report[seq] = {"ok": False, "reason": f"only {len(lpp_heavy)} heavy atoms"}
            continue
        # C91 / C85 by heavy order (not by guessing nearest/farthest carbon)
        p91 = np.asarray(lpp_heavy[heavy_idx91], dtype=float)
        p85 = np.asarray(lpp_heavy[heavy_idx85], dtype=float)

        # site = Cys SG closest to C91 (must be < 2.5 A, thioether bond)
        dvec = [np.linalg.norm(sg_xyz[k] - p91) for k in sg_keys]
        site_key = sg_keys[int(np.argmin(dvec))]
        d_sg = float(min(dvec))
        if d_sg > 2.5:
            report[seq] = {"ok": False, "reason": f"no SG within 2.5A (d={d_sg:.2f})"}
            continue
        sg = sg_xyz[site_key]

        # excluded residues: own Cys + the Cys whose SG is within 3.5A (its
        # disulfide/reduction partner). Their atoms sit 1.8-3.5A from SG and
        # are the covalent bonding environment, NOT clashes.
        excl_keys = {site_key}
        for k in sg_keys:
            if k == site_key:
                continue
            if np.linalg.norm(sg_xyz[k] - sg) < 3.5:
                excl_keys.add(k)
        # Clash criterion is heavy-atom only, matching the project QC standard
        # (qc.protein_lp_min = 2.0 A in config.yaml). Hydrogens are excluded on
        # purpose: a payload sitting 1.7 A from a hydrogen is ~2.7 A from the
        # heavy atom it is bonded to, which is not a clash. Counting hydrogens
        # made the test far too strict and rejected nearly every placement.
        prot_xyz = np.array([
            [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
            for ln in lines if ln.startswith("ATOM")
            and (ln[21], int(ln[22:26].strip())) not in excl_keys
            and _atom_element(ln).upper() not in ("H", "D")
        ], dtype=float)

        # direction to match: PDB C91 -> C85
        u_pdb = (p91 - p85) / max(np.linalg.norm(p91 - p85), 1e-9)

        # other-LP atoms (unchanged coords; old conformers)
        other_lps = []
        for (och, oseq), orecs in groups.items():
            if (och, oseq) == (ch, seq):
                continue
            other_lps.append(np.asarray([c for _, c in orecs], dtype=float))
        other = np.concatenate(other_lps, axis=0) if other_lps else np.zeros((0, 3))

        # try conformers tier by tier (collapsed -> mid -> partly -> plus),
        # with random roll (rotation around the C91->C85 axis). Buried sites
        # (A:CYS223 / D:CYS214) physically cannot fit a rigid globule, and
        # their original exit direction is often the ONLY blocked one. So if
        # tier search fails, re-scan the exit direction: keep C91 pinned but
        # let C85 point along any of n_exit_dirs uniformly sampled on the
        # sphere (a flexible linker can turn) — pick the one that fits a
        # collapsed conformer clash-free.
        rng = np.random.default_rng(91)
        roll_angles = np.linspace(0, 360, n_roll, endpoint=False)  # degrees
        placed = False
        tier = None
        span_old = _span(np.asarray(lpp_heavy, dtype=float))
        for pool_label, pool in zip(tier_names, pools):
            if not pool:
                continue
            cands = []
            for pool_idx, (mi, m, x, c85, c91) in enumerate(pool):
                u = (c91 - c85) / max(np.linalg.norm(c91 - c85), 1e-9)
                cands.append((float(np.dot(u, u_pdb)), pool_idx))
            cands.sort(key=lambda t: -t[0])
            for cos, pool_idx in cands[:max_tries]:
                mi, m, x, c85, c91 = pool[pool_idx]
                rot = align_to_anchor(c85, c91, p85, p91)
                base_xyz = coordinates_for_conformer(m, rot, c85, p91)
                u_axis = (p91 - p85) / max(np.linalg.norm(p91 - p85), 1e-9)
                for roll_deg in roll_angles:
                    r = Rotation.from_rotvec(np.deg2rad(roll_deg) * u_axis)
                    new_xyz = r.apply(base_xyz - p91) + p91
                    ok_p, dmin_p = check_clash(new_xyz, prot_xyz, protein_cutoff)
                    if not ok_p:
                        continue
                    if len(other):
                        d, _ = cKDTree(other).query(new_xyz, k=1)
                        if (d < extra_cutoff).any():
                            continue
                    placed = True
                    tier = pool_label
                    break
                if placed:
                    break
            if placed:
                break

        # exit-direction re-scan: fallback for genuinely blocked original dir.
        exit_scan_used = False
        if not placed:
            cands_col = []
            for pool_idx, (mi, m, x, c85, c91) in enumerate(pools[0]):  # collapsed only
                cands_col.append((float(np.linalg.norm(c91 - c85)), pool_idx))
            cands_col.sort(key=lambda t: t[0])
            # uniform exit directions on sphere (fibonacci)
            n_d = n_exit_dirs
            idxs = np.arange(n_d) + 0.5
            phi = np.arccos(1 - 2 * idxs / n_d)
            theta = np.pi * (1 + 5 ** 0.5) * idxs
            exit_dirs = np.stack([np.sin(phi) * np.cos(theta),
                                  np.sin(phi) * np.sin(theta),
                                  np.cos(phi)], axis=1)
            for _, pool_idx in cands_col[: min(6, max_tries)]:
                mi, m, x, c85, c91 = pools[0][pool_idx]
                for ed in exit_dirs:
                    # target exit direction: C85 sits at p91 - ed
                    rot = align_to_anchor(c85, c91, p91 - ed, p91)
                    base_xyz = coordinates_for_conformer(m, rot, c85, p91)
                    for roll_deg in roll_angles:
                        r = Rotation.from_rotvec(np.deg2rad(roll_deg) * ed)
                        new_xyz = r.apply(base_xyz - p91) + p91
                        ok_p, dmin_p = check_clash(new_xyz, prot_xyz, protein_cutoff)
                        if not ok_p:
                            continue
                        if len(other):
                            d, _ = cKDTree(other).query(new_xyz, k=1)
                            if (d < extra_cutoff).any():
                                continue
                        placed = True
                        tier = "collapsed_exit"
                        break
                    if placed:
                        break
                if placed:
                    break
        if placed and tier == "collapsed_exit":
            exit_scan_used = True

        report[seq] = {
            "site": f"{site_key[0]}:CYS{site_key[1]}", "ok": bool(placed),
            "tier": tier, "exit_scan": exit_scan_used,
            "min_dist_protein": round(dmin_p, 3),
            "conformer_idx": mi if placed else None,
            "cos": round(cos, 3) if placed else None,
            "span_new": round(_span(new_xyz), 1) if placed else None,
            "span_old": round(span_old, 1),
            "roll_deg": round(float(roll_deg), 1) if placed else None,
            "reason": None if placed else "no clash-free conformer in any tier or exit dir",
        }

        if placed:
            new_lines = []
            for pos in range(len(heavy)):
                atom = m.GetAtomWithIdx(int(heavy[pos]))
                x, y, z = new_xyz[pos]
                name = f"{atom.GetSymbol()}{heavy[pos]}"[:4]
                element = atom.GetSymbol()
                new_lines.append(
                    f"HETATM{next_serial:5d} {name:<4s} LPP {ch}{int(seq):4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
                )
                next_serial += 1
            for idx, (ln, _) in enumerate(recs):
                lines[lines.index(ln)] = new_lines[idx]

    return lines, report


def sg_serial_by_coord(lines, sg: np.ndarray) -> int:
    for ln in lines:
        if ln.startswith("ATOM") and ln[12:16].strip() == "SG":
            c = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
            if np.linalg.norm(c - sg) < 0.5:
                return int(ln[6:11])
    return -1


# --------------------------------------------------------------------------- #
# MD-frame mode: place collapsed conformers on real MD snapshots
#
# The old mode replaces payloads on ONE static antibody structure. If the
# side chains around a site do not leave room, there is nothing to do but
# fall back to a looser (more extended) conformer.
#
# This mode instead walks the 60 stored MD snapshots of the antibody. The
# hinge is genuinely mobile (SG wanders 3.7-11.3 A; the number of protein
# heavy atoms within 6 A of the attachment point swings between 10 and 54),
# so some frames simply have more room than others. Placing on the roomiest
# frame uses conformations the simulation actually sampled, which is closer
# to reality than an optimiser pushing side chains out of the way.
# --------------------------------------------------------------------------- #
def load_md_context(ref_pdb: str, npz_path: str):
    """Map MD frame atom order onto a reference PDB.

    Two flavours of npz are accepted:

      allcut  (data/md_af_nolock/allcut_r{0..3}.npz) -- the current MD.
        4 replicas x 2000 frames, mode "allcut" = ALL FOUR interchain
        disulfides reduced, so all 8 conjugation sites are free thiols and a
        multi-site DAR can be built from one self-consistent snapshot.
        sg_i is an array of the 8 site atoms; chains/resids give their names.

      per-site (results_hinge/*-Cys*.npz) -- the older MD.
        60 frames, ONE interchain disulfide reduced per file. Kept only for
        comparison: a multi-site DAR built from these is missing the reduction
        at the other pairs.

    Frames are stored in NANOMETRES; atom order equals the heavy-atom order of
    ref_pdb. Returns frames in Angstrom, per-atom metadata (so a residue can be
    excluded from the clash test by chain+resSeq) and the site SG indices.
    """
    rows = [ln for ln in Path(ref_pdb).read_text().splitlines()
            if ln.startswith("ATOM")]
    meta, heavy_rows = [], []
    for ln in rows:
        if _atom_element(ln).upper() in ("H", "D"):
            continue
        meta.append(dict(chain=ln[21], resSeq=int(ln[22:26]),
                         resName=ln[17:20].strip(),
                         atomName=ln[12:16].strip(),
                         element=_atom_element(ln)))
        heavy_rows.append(ln)
    d = np.load(npz_path, allow_pickle=True)
    raw = d["frames"]
    # Unit handling must match adcsim.io.load_md_frames (this script is standalone
    # and cannot import the package, so the logic is inlined here on purpose).
    # Frames are stored in NANOMETRES; convert to Angstrom by x10 unless an
    # explicit `units` key says otherwise, or auto-detect from the median
    # nearest-neighbour spacing (protein frames are ~0.14 nm = 1.4 A).
    if "units" in d.files:
        units = str(d["units"]).strip().lower()
        scale = 1.0 if units.startswith("a") else 10.0
    else:
        probe = raw[0].astype(np.float64)
        nn = cKDTree(probe).query(probe, k=2)[0][:, 1]
        scale = 10.0 if np.median(nn) < 0.5 else 1.0
    # float32 keeps a 2000-frame trajectory at ~245 MB instead of ~490 MB
    frames = (raw * np.float32(scale)).astype(np.float32)
    if frames.shape[1] != len(meta):
        raise SystemExit(
            f"MD frame has {frames.shape[1]} heavy atoms but {ref_pdb} has {len(meta)}")
    if d["sg_i"].ndim == 0:                       # old per-site flavour
        sg_idx = {(m["chain"], m["resSeq"]): i
                  for i, m in enumerate(meta)
                  if m["resName"] == "CYS" and m["atomName"] == "SG"}
    else:                                          # allcut flavour
        sg_idx = {(str(c), int(r)): int(i) for c, r, i
                  in zip(d["chains"], d["resids"], d["sg_i"])}
    return frames, meta, heavy_rows, sg_idx


def residue_indices(meta, chain, resSeq) -> list[int]:
    return [i for i, m in enumerate(meta)
            if m["chain"] == chain and m["resSeq"] == resSeq]


def hemisphere_dirs(axis: np.ndarray, n: int) -> np.ndarray:
    """~n unit vectors spread over the hemisphere centred on `axis`.

    Only outward directions are sampled: a payload tethered to a surface
    cysteine cannot point into the protein core.
    """
    m = max(8, 2 * n)
    k = np.arange(m) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / m)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * k
    d = np.stack([np.sin(phi) * np.cos(theta),
                  np.sin(phi) * np.sin(theta),
                  np.cos(phi)], axis=1)
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    return d[d @ axis > 0.0]


# Conformer tiers, by end-to-end span. The solution ensemble is 60% collapsed
# (<=20 A), 25% half-collapsed (20-25), 10% half-extended (25-30) and 5%
# extended (>30). Collapsed is the dominant form in buffer, so it is always
# tried first, but the hinge cysteine pairs sit only 5-9 A apart while a
# collapsed globule is ~12-14 A across -- several sites cannot hold one and
# must fall back to a looser conformer. That is a real geometric limit of a
# high-DAR ADC, not a failure of the search.
TIERS = [("collapsed", 0.0, 20.0),
         ("half_collapsed", 20.0, 25.0),
         ("half_extended", 25.0, 30.0),
         ("extended", 30.0, 1e9)]


def build_rg_pool(sdf_path: str, max_span: float) -> list[dict]:
    """Collapsed conformers (span <= max_span) sorted compact-first."""
    return build_tier_pools(sdf_path)[0][1]


def build_tier_pools(sdf_path: str) -> list[tuple[str, list[dict]]]:
    """Conformers grouped into TIERS, each group sorted compact-first."""
    mols = [m for m in Chem.SDMolSupplier(sdf_path, removeHs=False) if m is not None]
    recs = []
    for mi, m in enumerate(mols):
        heavy = [i for i in range(m.GetNumAtoms())
                 if m.GetAtomWithIdx(i).GetAtomicNum() > 1]
        if 91 not in heavy:
            continue
        x = _heavy_xyz(m, 0)
        span = _span(x)
        c = m.GetConformer()
        cen = x.mean(0)
        recs.append(dict(mi=mi, mol=m, xyz=x, span=span,
                         c85=np.asarray(c.GetAtomPosition(85), dtype=float),
                         c91=np.asarray(c.GetAtomPosition(91), dtype=float),
                         rg=float(np.sqrt(((x - cen) ** 2).sum(1).mean())),
                         heavy=heavy, i91=heavy.index(91)))
    out = []
    for name, lo, hi in TIERS:
        grp = [r for r in recs if lo <= r["span"] < hi]
        grp.sort(key=lambda r: r["rg"])
        if grp:
            out.append((name, grp))
    return out


def place_on_site(sg: np.ndarray, prot_xyz: np.ndarray, pools,
                  dirs: np.ndarray, n_roll: int, cutoff: float,
                  obstacles, bond: float = 1.82, bond_range=None):
    """Fit the most compact conformer that clears protein and other payloads.

    `pools` is a list of (tier_name, conformers) ordered tightest first (see
    TIERS). Within a tier conformers are tried compact-first, so the first hit
    IS the most collapsed pose that fits -- no need to score every pose. Only
    if a whole tier fails do we loosen to the next one.

    If nothing reaches `cutoff` the best pose seen is returned anyway with
    tight=False and its real dmin, rather than None: some sites (C214 in
    particular) are too crowded for a clean fit in every snapshot, and the
    caller needs to see how tight it is instead of silently losing a payload.

    `bond` is the S-C thioether bond length (default 1.82 A, the standard
    Cys SG - maleimide C bond). Real S-C bonds span ~1.7-2.4 A; when
    `bond_range=(lo, hi, n)` is given, the C91 anchor is placed at every
    sampled bond length in that interval and the clash test is re-run, so a
    site that cannot fit a payload at 1.82 A may still fit at a slightly
    longer (or shorter) bond with less steric strain. The returned dict
    carries the chosen `bond_len`.
    """
    tree = cKDTree(prot_xyz)
    obs_tree = cKDTree(obstacles) if obstacles is not None and len(obstacles) else None
    rolls = np.linspace(0.0, 360.0, n_roll, endpoint=False)
    n_at = None
    best = None
    # bond lengths to try: single value by default, or a sampled range
    if bond_range:
        lo, hi, n_b = bond_range
        bonds = np.linspace(lo, hi, int(n_b))
    else:
        bonds = [bond]
    for tier_name, pool in pools:
        for rec in pool:
            src = rec["c85"] - rec["c91"]
            nv = np.linalg.norm(src)
            if nv < 1e-6:
                continue
            src = src / nv
            body = rec["xyz"] - rec["c91"]
            n_at = body.shape[0]
            for dirv in dirs:
                rot = Rotation.align_vectors(np.atleast_2d(dirv),
                                             np.atleast_2d(src))[0]
                for bd in bonds:
                    p91 = sg + bd * dirv
                    base = rot.apply(body)                      # (n_at, 3)
                    # every rotation about the tether axis at once: one batched
                    # rotation per (roll, atom) pair instead of a Python loop
                    rv = np.repeat(np.deg2rad(rolls)[:, None] * dirv[None, :],
                                   n_at, axis=0)
                    stack = Rotation.from_rotvec(rv).apply(
                        np.tile(base, (len(rolls), 1))) + p91
                    d = tree.query(stack, k=1)[0].reshape(len(rolls), n_at).min(axis=1)
                    if obs_tree is not None:
                        do = obs_tree.query(stack, k=1)[0].reshape(len(rolls), n_at)
                        d = np.minimum(d, do.min(axis=1))
                    k = int(np.argmax(d))
                    if d[k] >= cutoff:
                        new = stack[k * n_at:(k + 1) * n_at]
                        return dict(rec=rec, new=new, dmin=float(d[k]),
                                    tier=tier_name, tight=True,
                                    dirv=dirv.copy(), deg=float(rolls[k]),
                                    p91=p91.copy(), bond_len=float(bd))
                    if best is None or d[k] > best["dmin"]:
                        new = stack[k * n_at:(k + 1) * n_at]
                        best = dict(rec=rec, new=new, dmin=float(d[k]),
                                    tier=tier_name, tight=False,
                                    dirv=dirv.copy(), deg=float(rolls[k]),
                                    p91=p91.copy(), bond_len=float(bd))
    return best


def read_sites_from_pdbs(d: str) -> dict[str, list[tuple[str, int]]]:
    """Which (chain, resSeq) each existing DAR structure actually uses."""
    out = {}
    for p in sorted(Path(d).glob("ADC_DAR*.pdb")):
        text = Path(p).read_text().splitlines()
        sg = [(ln[21], int(ln[22:26]),
               np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]))
              for ln in text if ln.startswith("ATOM")
              and ln[17:20].strip() == "CYS" and ln[12:16].strip() == "SG"]
        groups = defaultdict(list)
        for ln in text:
            if ln.startswith("HETATM") and ln[17:20].strip() == "LPP":
                groups[int(ln[22:26])].append(
                    np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]))
        if not groups:
            continue
        sites = []
        for ri, coords in groups.items():
            _, s = min(((np.linalg.norm(c - t[2]), t) for c in coords for t in sg),
                       key=lambda z: z[0])
            sites.append((s[0], s[1]))
        parts = p.stem.split("_")
        dar = parts[1]
        if len(parts) > 2 and parts[2] not in ("collapsed", "min"):
            dar = f"{parts[1]}_{parts[2]}"
        out[dar] = sorted(set(sites))
    return out


def write_md_pdb(path: Path, heavy_rows, xyz, placed) -> None:
    """Write one ADC structure: MD-frame protein + payloads + real bonds."""
    lines = []
    serial = 0
    serial_of_heavy = {}
    for i, ln in enumerate(heavy_rows):
        serial += 1
        serial_of_heavy[i] = serial
        x, y, z = xyz[i]
        lines.append(f"ATOM  {serial:5d}{ln[11:30]}{x:8.3f}{y:8.3f}{z:8.3f}{ln[54:]}")
    conect = []
    for k, ((ch, resi), sgi, r) in enumerate(placed):
        rec, mol, new, heavy = r["rec"], r["rec"]["mol"], r["new"], r["rec"]["heavy"]
        s_of = {}
        for pos, ai in enumerate(heavy):
            serial += 1
            s_of[ai] = serial
            atom = mol.GetAtomWithIdx(int(ai))
            # Keep the two atoms the rest of the pipeline looks up by name:
            # C91 is the maleimide carbon bonded to Cys SG, C85 the linker exit.
            name = {91: "C91", 85: "C85"}.get(int(ai),
                                              f"{atom.GetSymbol()}{pos}"[:4])
            x, y, z = new[pos]
            lines.append(f"HETATM{serial:5d} {name:<4s} LPP {ch}{900 + k + 1:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom.GetSymbol():>2s}")
        for b in mol.GetBonds():
            a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if a1 in s_of and a2 in s_of:
                conect.append((s_of[a1], s_of[a2]))
        conect.append((serial_of_heavy[sgi], s_of[91]))
    for a, b in conect:
        lines.append(f"CONECT{a:5d}{b:5d}")
        lines.append(f"CONECT{b:5d}{a:5d}")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def sasa_candidates(sasa_path, labels, sites, n_frames, n_keep):
    """Rank (replica, frame) by how exposed the requested sites are.

    sasa_path holds sasa_A2 of shape (n_rep, n_frames, 8) -- the solvent
    exposure of each conjugation-site thiol, one entry per stored frame. The
    bottleneck of the search is a crowded site, so rank a snapshot by its
    WORST requested site and keep the roomiest. This replaces a blind scan of
    8000 snapshots with a few dozen targeted ones.
    """
    if not sasa_path or not Path(sasa_path).exists():
        return None
    d = np.load(sasa_path, allow_pickle=True)
    if "sasa_A2" not in d:
        return None
    sasa = d["sasa_A2"]
    lab = [str(x) for x in d["labels"]]
    cols = [lab.index(f"{c}-Cys{r}") for c, r in sites]
    score = sasa[:, :, cols].min(axis=2)          # (n_rep, n_frames)
    flat = [(score[rep, t], rep, t)
            for rep in range(score.shape[0])
            for t in range(min(n_frames, score.shape[1]))]
    flat.sort(reverse=True)
    return [(rep, t) for _, rep, t in flat[:n_keep]]


def scan_snapshots(frames, meta, sg_idx, sites, pools, cutoff, n_roll, n_dirs,
                   stride=1, frame_list=None, bond_range=None):
    """Try stored frames; rank by (sites meeting cutoff, compact, clearance)."""
    centroid = frames.mean(axis=1)
    out = []
    for t in (frame_list if frame_list is not None
              else range(0, frames.shape[0], stride)):
        xyz = frames[t]
        placed, obstacles = [], []
        for (ch, resi) in sites:
            sgi = sg_idx[(ch, resi)]
            sg = xyz[sgi]
            excl = set(residue_indices(meta, ch, resi))
            # Drop any conjugation-site thiol sitting right on top of this one.
            # In the allcut MD every pair is already reduced (closest two site
            # SGs are 5.1 A apart) so this never fires; it only matters for the
            # older per-site frames, where the un-reduced half of a disulfide
            # would otherwise block a site that is free in the real reaction.
            # A site that is NOT in this DAR keeps its thiol and stays an
            # obstacle -- that sulfur really is there.
            for (c2, r2), j in sg_idx.items():
                if (c2, r2) == (ch, resi):
                    continue
                if np.linalg.norm(xyz[j] - sg) < 3.5:
                    excl |= set(residue_indices(meta, c2, r2))
            prot = np.delete(xyz, list(excl), axis=0)
            out_axis = sg - centroid[t]
            if np.linalg.norm(out_axis) < 1e-6:
                out_axis = sg - xyz.mean(0)
            r = place_on_site(sg, prot, pools,
                              hemisphere_dirs(out_axis, n_dirs),
                              n_roll, cutoff,
                              np.concatenate(obstacles) if obstacles else None,
                              bond_range=bond_range)
            if r is None:
                break
            placed.append(((ch, resi), sgi, r))
            obstacles.append(r["new"])
        if not placed:
            out.append(dict(t=t, n=0, n_ok=0, rg=99.0, dmin=0.0,
                            short=99.0, n_collapsed=0, tiers=[], placed=[]))
            continue
        tiers = [p[2]["tier"] for p in placed]
        dmins = [p[2]["dmin"] for p in placed]
        out.append(dict(
            t=t, n=len(placed),
            n_ok=sum(1 for p in placed if p[2]["tight"]),
            rg=float(np.mean([p[2]["rec"]["rg"] for p in placed])),
            dmin=float(min(dmins)),
            short=float(sum(max(0.0, cutoff - d) for d in dmins)),
            n_collapsed=sum(1 for x in tiers if x == "collapsed"),
            tiers=tiers, placed=placed))
    out.sort(key=lambda r: (-r["n"], -r["n_ok"], -r["n_collapsed"], r["short"]))
    return out


def run_md_mode(ref_pdb, npz_paths, sdf, sites_from, out_dir, dar_spec,
                cutoff, n_roll, n_dirs, max_span, top_k=1, stride=1,
                sasa_path=None, n_cand=40, bond_range=None):
    """Place payloads on real MD snapshots, searching every trajectory.

    Trajectories are never spliced into a chimera: each one differs from the
    others by up to 80 A, and mixing them would produce a structure the
    simulation never visited. Every (trajectory, frame) pair is instead scored
    as a whole self-consistent antibody, and the roomiest wins. The side chains
    then sit in poses the simulation actually sampled.

    Preferred source is data/md_af_nolock/allcut_r{0..3}.npz: 4 replicas x
    2000 frames with all four interchain disulfides reduced, so a DAR of any
    size comes from one physically coherent snapshot.
    """
    pools = build_tier_pools(sdf)
    if not pools:
        raise SystemExit("no conformers")
    print("conformer tiers: " + "  ".join(
        f"{name}={len(g)}" for name, g in pools), flush=True)

    known = read_sites_from_pdbs(sites_from)
    targets = dict(known)
    for dar, site_list in dar_spec.items():
        targets[dar] = [tuple(s) for s in site_list]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    for dar in sorted(targets, key=lambda s: int(s.split("_")[0][3:])):
        sites = targets[dar]
        cand = sasa_candidates(sasa_path, None, sites, 10 ** 9, n_cand)
        by_rep = defaultdict(list)
        if cand:
            for rep, t in cand:
                if rep < len(npz_paths):
                    by_rep[rep].append(t)
        best = None
        for ri, npz_path in enumerate(npz_paths):
            frames, meta, heavy_rows, sg_idx = load_md_context(ref_pdb, npz_path)
            missing = [s for s in sites if s not in sg_idx]
            if missing:
                continue
            flist = sorted(by_rep[ri]) if by_rep else None
            ranked = scan_snapshots(frames, meta, sg_idx, sites,
                                    pools, cutoff, n_roll, n_dirs, stride,
                                    frame_list=flist, bond_range=bond_range)
            if not ranked:
                continue
            top = ranked[0]
            # prefer: most payloads, most of them clean, most of them collapsed
            nm = Path(npz_path).name
            if "allcut" in nm:
                own = len(sites)          # every pair reduced already
            else:
                own = sum(1 for s in sites
                          if s[0] == nm[0]
                          and s[1] == int(nm.split("Cys")[1][:3]))
            key = (top["n"], top["n_ok"], top["n_collapsed"], own, -top["short"])
            if best is None or key > best["key"]:
                best = dict(key=key, npz=npz_path, ranked=ranked,
                            frames=frames, heavy_rows=heavy_rows,
                            n_ok=sum(1 for r in ranked
                                     if r["n_ok"] == len(sites)),
                            n_frames=len(ranked))
        if best is None:
            print(f"{dar}: 跳过，没有可用轨迹")
            continue
        ranked = best["ranked"]
        npz_name = Path(best["npz"]).name
        for rank in range(top_k):
            r = ranked[rank] if rank < len(ranked) else ranked[0]
            tag = dar if rank == 0 else f"{dar}_alt{rank}"
            write_md_pdb(out_dir / f"ADC_{tag}_md.pdb", best["heavy_rows"],
                         best["frames"][r["t"]], r["placed"])
        b = ranked[0]
        report[dar] = dict(
            sites=[f"{c}{s}" for c, s in sites],
            n_requested=len(sites), n_placed=b["n"],
            complete=b["n"] == len(sites),
            source_npz=npz_name, best_frame=int(b["t"]),
            frames_fully_placed=best["n_ok"], n_frames_scanned=len(ranked),
            mean_rg=round(b["rg"], 2), min_dist=round(b["dmin"], 2),
            n_collapsed=b["n_collapsed"],
            tiers={f"{c}{s}": t for (c, s), t in zip(sites, b["tiers"])})
        print(f"{dar}: 位点{len(sites)} 放下{b['n']} "
              f"(其中蜷缩{b['n_collapsed']})  "
              f"来自 {npz_name} 帧#{b['t']}  平均Rg {b['rg']:.2f} A  "
              f"最小间距 {b['dmin']:.2f} A  "
              f"(扫{len(ranked)}帧, {best['n_ok']}帧可全放下)", flush=True)

    (out_dir / "md_placement_report.json").write_text(
        json.dumps(report, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_dir", required=True)
    ap.add_argument("--sdf", required=True,
                    help="path to payload_ensemble_100.sdf")
    ap.add_argument("--max-span", type=float, default=20.0)
    ap.add_argument("--out-dir", default=None)
    # Heavy-atom cutoffs, same 2.0 A standard as qc.protein_lp_min /
    # qc.lp_lp_min in config.yaml. The old 1.7 default also counted
    # hydrogens, which is why the previous run both rejected good poses and
    # let real overlaps (C90---1HG at 1.18 A) through.
    ap.add_argument("--protein-cutoff", type=float, default=2.0,
                    help="min payload-to-protein HEAVY-atom distance (A)")
    ap.add_argument("--extra-cutoff", type=float, default=2.0,
                    help="min payload-to-payload heavy-atom distance (A)")
    ap.add_argument("--n-roll", type=int, default=24,
                    help="rotations tried around the C91->C85 axis")

    # --- MD-frame mode ---------------------------------------------------- #
    # Instead of placing on one static antibody, walk the stored MD snapshots
    # and place on the frame that has room. Side chains then do not have to be
    # pushed aside by an optimiser: they are already out of the way because the
    # simulation actually sampled that pose.
    ap.add_argument("--md-mode", action="store_true",
                    help="place collapsed conformers on real MD snapshots")
    ap.add_argument("--ref-pdb", default=None,
                    help="[md-mode] PDB whose heavy-atom order matches the npz")
    ap.add_argument("--npz", nargs="+", default=None,
                    help="[md-mode] MD frame npz, glob allowed "
                         "(default: data/md_af_nolock/allcut_r*.npz)")
    ap.add_argument("--sites-from", default=None,
                    help="[md-mode] dir of ADC_DAR*.pdb to read site combos from")
    ap.add_argument("--n-dirs", type=int, default=40,
                    help="[md-mode] outward directions sampled per site")
    ap.add_argument("--top-k", type=int, default=1,
                    help="[md-mode] how many of the best frames to write")
    ap.add_argument("--stride", type=int, default=1,
                    help="[md-mode] use every Nth frame (speeds up the scan)")
    ap.add_argument("--sasa", default=None,
                    help="[md-mode] npz with sasa_A2 used to pre-rank frames")
    ap.add_argument("--n-cand", type=int, default=40,
                    help="[md-mode] how many pre-ranked frames to actually try")
    ap.add_argument("--add-dar8", action="store_true",
                    help="[md-mode] add DAR8 (all 8 sites) to the target list")
    ap.add_argument("--bond-range", type=float, nargs=3, default=None,
                    metavar=("LO", "HI", "N"),
                    help="[md-mode] sample S-C thioether bond lengths in [LO,HI] with N steps "
                         "(default 1.82 A fixed; e.g. 1.70 2.40 7)")
    args = ap.parse_args()

    if args.md_mode:
        if not args.ref_pdb:
            raise SystemExit("--md-mode needs --ref-pdb")
        npz_paths = []
        for pat in (args.npz or ["data/md_af_nolock/allcut_r*.npz"]):
            hits = sorted(glob.glob(pat))
            if not hits:
                raise SystemExit(f"no npz matched {pat}")
            npz_paths.extend(hits)
        print(f"MD 快照来源 ({len(npz_paths)} 条轨迹): "
              + ", ".join(Path(p).name for p in npz_paths), flush=True)
        dar_spec = {}
        if args.add_dar8:
            dar_spec["DAR8"] = [("A", 223), ("A", 229), ("A", 232),
                                ("B", 223), ("B", 229), ("B", 232),
                                ("C", 214), ("D", 214)]
        run_md_mode(
            ref_pdb=args.ref_pdb,
            npz_paths=npz_paths,
            sdf=args.sdf,
            sites_from=args.sites_from or str(args.pdb_dir),
            out_dir=args.out_dir or str(Path(args.pdb_dir) / "md_placed"),
            dar_spec=dar_spec,
            cutoff=args.protein_cutoff,
            n_roll=args.n_roll,
            n_dirs=args.n_dirs,
            max_span=args.max_span,
            top_k=args.top_k,
            stride=args.stride,
            sasa_path=args.sasa or "results_hinge/md_af_allcut_4rep_series.npz",
            n_cand=args.n_cand,
            bond_range=tuple(args.bond_range) if args.bond_range else None,
        )
        return

    pools = build_collapsed_pool(args.sdf, args.max_span)
    counts = [len(p) for p in pools]
    print(f"pools [collapsed/mid/partly/plus]: {counts}", flush=True)
    if not any(pools):
        raise SystemExit("no conformers in any tier")

    pdb_dir = Path(args.pdb_dir)
    out_dir = Path(args.out_dir) if args.out_dir else pdb_dir / "polished_collapsed"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_reports = {}
    for pdb in sorted(pdb_dir.glob("ADC_DAR*.pdb")):
        print(f"=== {pdb.name} ===", flush=True)
        lines, report = process_pdb(
            pdb, pools,
            protein_cutoff=args.protein_cutoff,
            extra_cutoff=args.extra_cutoff,
            n_roll=args.n_roll,
        )
        all_reports[pdb.name] = report
        out = out_dir / f"{pdb.stem}_collapsed.pdb"
        # append CONECT/END from original
        origen = pdb.read_text()
        tail = ""
        for ln in origen.splitlines():
            if ln.startswith(("CONECT", "END")):
                tail += ln + "\n"
        out.write_text("\n".join(lines) + "\n" + tail)
        for seq, r in report.items():
            print(f"  LP{seq}: {r}", flush=True)

    (out_dir / "replace_report.json").write_text(
        json.dumps(all_reports, indent=2, default=str))
    print(f"\nWrote {out_dir}/*.pdb + replace_report.json")


if __name__ == "__main__":
    main()