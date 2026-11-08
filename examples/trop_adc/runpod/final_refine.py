#!/usr/bin/env python3
"""Final refinement of the placed ADC structures (DAR 1-7).

Where this step sits
--------------------
    replace_collapsed.py --md-mode   ->  results_hinge/md_placed_raw/*.pdb
    this script                     ->  results_hinge/md_placed/*.pdb
    PyMOL figures                   ->  results_hinge/pymol_images/*.png

`md_placed_raw/` is the untouched output of the previous placement search and
is shipped with the repository as the input to this step. It was produced with
a 7-step 1.70-2.40 A SG-C91 ladder, one MD frame per DAR and the best
direction / roll / compact conformer per site. That ladder is wider than a real
thioether bond, so this step is the last optimisation on top of it: it keeps
the MD frame, the site list, the direction and all protein coordinates, and
re-optimises together

  * bond length : each site's winning ladder value mapped into the measured
                  C-S single-bond range 1.78-1.87 A (Cox & Jeffrey,
                  Proc. R. Soc. A 1951; CSD mean 1.807 A over 546 structures),
                  with a +/-0.01 A micro-adjustment;
  * roll        : full 360 degrees about the SG-C91 tether;
  * axis tilt   : a local cone around the previous optimum direction;
  * clash test  : 2.0 A heavy-atom cutoff, with the same whole-residue
                  exclusion rule as scan_snapshots() in replace_collapsed.py.

Every site therefore keeps its own traceable bond length. If the original
conformer cannot clear 2.0 A at a real bond length, the search falls back to
other conformers of the SAME compactness tier (compact-first), which is what
place_on_site() itself does. In this dataset that happens at exactly one site:
DAR5 A229. All other 27 sites keep the conformer chosen by the placement step.

No energy minimisation or binding-affinity scoring is performed, and there is
no random number anywhere: the output is deterministic.

DAR7 never went through the 7-step ladder search, so its per-site target
lengths are inherited from the same sites in DAR6 (D214 from DAR4).

Running it
----------
    pip install -e .            # provides numpy, scipy and rdkit
    python examples/trop_adc/runpod/final_refine.py
    python examples/trop_adc/runpod/final_refine.py --check

Default mode rewrites the seven PDBs in results_hinge/md_placed/. `--check`
regenerates them in memory and compares byte-for-byte against the files on
disk, without writing anything: it exits 0 only when the committed structures
are exactly what this script produces. A clean checkout should pass --check
and leave `git status` clean after a default run.

Inputs, all inside the repository:
  results_hinge/md_placed_raw/ADC_DAR*_md.pdb   preceding placement output
  results_hinge/md_placed/md_placement_report.json  per-site conformer tiers
  data/payload_ensemble_100.sdf                  vc-MMAE conformer ensemble
  runpod/replace_collapsed.py                    build_tier_pools()

Only the payload atom coordinates change; protein coordinates and every CONECT
record are preserved.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
EX = HERE.parent
sys.path.insert(0, str(HERE))
from replace_collapsed import build_tier_pools  # noqa: E402

SRC = EX / "results_hinge" / "md_placed_raw"
DST = EX / "results_hinge" / "md_placed"
SDF = EX / "data" / "payload_ensemble_100.sdf"
PLACEMENT_REPORT = DST / "md_placement_report.json"

FILES = {n: "ADC_DAR%d_md.pdb" % n for n in range(1, 7)}
FILES[7] = "ADC_DAR7_omit_md.pdb"
REPORT_KEY = {n: "DAR%d" % n for n in range(1, 7)}
REPORT_KEY[7] = "DAR7_omit"

OLD_LO, OLD_HI = 1.70, 2.40          # ladder used by the previous step
NEW_LO, NEW_HI = 1.78, 1.87          # measured C-S single-bond range
CUTOFF = 2.0
SWEEPS = 2

# stage 1: tight cone, smallest departure from the previous optimum
FAST_TILTS = (0.0, 7.5, 15.0, 22.5, 30.0)
FAST_AZ_STEP = 30.0
FAST_ROLL_STEP = 5.0
FAST_OFFSETS = (-0.01, 0.0, 0.01)
# stage 2: wider cone and finer sampling, still centred on the previous optimum
WIDE_TILTS = tuple(np.arange(0.0, 45.1, 5.0))
WIDE_AZ_STEP = 20.0
WIDE_ROLL_STEP = 3.0
WIDE_OFFSETS = (-0.02, -0.01, 0.0, 0.01, 0.02)
MAX_TIER_TRIES = 24                  # compact-first conformers to fall back on

POOLS = dict(build_tier_pools(str(SDF)))
PREV = json.loads(PLACEMENT_REPORT.read_text())


def remap(d):
    return NEW_LO + (d - OLD_LO) * (NEW_HI - NEW_LO) / (OLD_HI - OLD_LO)


def parse(path):
    lines = path.read_text().splitlines()
    atoms = []
    for i, ln in enumerate(lines):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        atoms.append(dict(
            i=i, het=ln.startswith("HETATM"),
            name=ln[12:16].strip(), resn=ln[17:20].strip(),
            chain=ln[21], resi=ln[22:26].strip(),
            elem=ln[76:78].strip().upper(),
            xyz=np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]),
        ))
    return lines, atoms


def render(lines, atoms):
    """Return the PDB text. Only the atom coordinate columns are rewritten."""
    out = list(lines)
    for a in atoms:
        ln = out[a["i"]]
        x, y, z = a["xyz"]
        out[a["i"]] = ln[:30] + "%8.3f%8.3f%8.3f" % (x, y, z) + ln[54:]
    return "\n".join(out) + "\n"


def site_lengths(path):
    """(chain, resi) -> SG-C91 distance in the input structure."""
    _, atoms = parse(path)
    pay = defaultdict(list)
    for a in atoms:
        if a["het"] and a["resn"] == "LPP":
            pay[(a["chain"], a["resi"])].append(a)
    sg = [a for a in atoms if not a["het"] and a["resn"] == "CYS"
          and a["name"] == "SG"]
    tree = cKDTree(np.array([a["xyz"] for a in sg]))
    out = {}
    for key, group in pay.items():
        c91 = next(a for a in group if a["name"] == "C91")
        d, idx = tree.query(c91["xyz"])
        s = sg[int(idx)]
        out[(s["chain"], int(s["resi"]))] = float(d)
    return out


def cone_dirs(u0, tilts, az_step):
    seed = (np.array([1.0, 0.0, 0.0]) if abs(u0[0]) < 0.9
            else np.array([0.0, 1.0, 0.0]))
    e1 = np.cross(u0, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u0, e1)
    dirs = [(0.0, u0)]
    for td in tilts:
        if td == 0.0:
            continue
        a = math.radians(td)
        for az in np.arange(0.0, 360.0, az_step):
            z = math.radians(az)
            v = u0 + math.tan(a) * (math.cos(z) * e1 + math.sin(z) * e2)
            v /= np.linalg.norm(v)
            dirs.append((td, v))
    return dirs


def search_one(sg_xyz, body, src, u0, d_new, obs_tree,
               dirs, roll_step, offsets, label):
    """Joint length x roll x tilt search for one conformer, seeded at u0."""
    lens = np.clip(d_new + np.array(offsets), NEW_LO, NEW_HI)
    roll_degs = np.arange(0.0, 360.0, roll_step)
    rolls = np.deg2rad(roll_degs)
    c = np.cos(rolls)[:, None, None]
    s = np.sin(rolls)[:, None, None]
    roll_dep = np.minimum(roll_degs, 360.0 - roll_degs)[None, :]
    length_pen = np.abs(lens - d_new)[:, None]
    best = None
    for tilt_deg, un in dirs:
        aligned = (body if np.allclose(un, u0, atol=1e-9) else
                   Rotation.align_vectors(np.atleast_2d(un),
                                          np.atleast_2d(src))[0].apply(body))
        dot = np.einsum("ijk,k->ij", aligned[None, :, :], un)
        rot_all = (aligned[None, :, :] * c
                   + np.cross(un, aligned)[None, :, :] * s
                   + dot[:, :, None] * un[None, None, :] * (1.0 - c))
        cand = (sg_xyz + lens[:, None, None, None] * un
                + rot_all[None, :, :, :])
        # Nearest-neighbour distances are exact and independent of the worker
        # count, so a different machine produces the same picked pose.
        dm = obs_tree.query(cand.reshape(-1, 3), k=1, workers=-1)[0].reshape(
            len(lens), len(roll_degs), -1).min(axis=2)
        score = dm - 0.004 * tilt_deg - 0.0003 * roll_dep - 1.5 * length_pen
        # any pose clearing 2.0 A beats every pose that does not
        score = np.where(dm >= CUTOFF, score, -1.0e3 + dm)
        li, ri = np.unravel_index(int(np.argmax(score)), score.shape)
        rec = dict(score=float(score[li, ri]), dmin=float(dm[li, ri]),
                   length=float(lens[li]), roll_deg=float(roll_degs[ri]),
                   tilt_deg=float(tilt_deg), xyz=cand[li, ri].copy(),
                   conformer=label)
        if best is None or rec["score"] > best["score"]:
            best = rec
    return best


def tier_bodies(tier, cur_body, cur_src):
    """Previous optimum first, then the other conformers of the SAME tier."""
    out = [("kept", cur_body, cur_src)]
    n = cur_body.shape[0]
    for j, rec in enumerate(POOLS.get(tier, [])):
        b = rec["xyz"] - rec["c91"]
        if b.shape[0] != n:
            continue
        src = rec["c85"] - rec["c91"]
        nv = np.linalg.norm(src)
        if nv < 1e-6:
            continue
        out.append(("%s#%d" % (tier, j), b, src / nv))
    return out


def refine(path, target_by_site, original_by_site, dar):
    lines, atoms = parse(path)
    prot = [a for a in atoms if not a["het"] and a["elem"] not in ("H", "D")]
    pay = defaultdict(list)
    for a in atoms:
        if a["het"] and a["resn"] == "LPP":
            pay[(a["chain"], a["resi"])].append(a)

    sg_all = [a for a in prot if a["resn"] == "CYS" and a["name"] == "SG"]
    sg_tree = cKDTree(np.array([a["xyz"] for a in sg_all]))
    tiers = PREV.get(REPORT_KEY[dar], {}).get("tiers", {})

    jobs = []
    for key, group in sorted(pay.items()):
        c91 = next(a for a in group if a["name"] == "C91")
        d, idx = sg_tree.query(c91["xyz"])
        s = sg_all[int(idx)]
        if d > 2.6:
            raise RuntimeError("no bonded SG for %s in %s" % (key, path))
        jobs.append((key, group, c91, s, (s["chain"], int(s["resi"]))))

    fallback_used = set()
    for sweep in range(SWEEPS):
        report = []
        for key, group, c91, sg, site in jobs:
            sg_xyz = sg["xyz"]
            c91_xyz = c91["xyz"].copy()
            d_old = float(np.linalg.norm(c91_xyz - sg_xyz))
            u0 = (c91_xyz - sg_xyz) / d_old
            d_new = float(np.clip(target_by_site[site], NEW_LO, NEW_HI))

            # same exclusion rule as scan_snapshots(): drop the whole cysteine
            # residue that owns this SG (its backbone O is part of the bond)
            obs = [a["xyz"] for a in prot
                   if not (a["chain"] == sg["chain"]
                           and a["resi"] == sg["resi"])]
            for k in pay:
                if k != key:
                    obs.extend(a["xyz"] for a in pay[k])
            tree = cKDTree(np.array(obs))

            body = np.array([a["xyz"] - c91_xyz for a in group])
            i85 = next(i for i, a in enumerate(group) if a["name"] == "C85")
            cur_src = body[i85] / np.linalg.norm(body[i85])
            tier = tiers.get("%s%d" % site, "collapsed")
            bodies = tier_bodies(tier, body, cur_src)

            fast = cone_dirs(u0, FAST_TILTS, FAST_AZ_STEP)
            wide = cone_dirs(u0, WIDE_TILTS, WIDE_AZ_STEP)
            pick, stage = None, None
            # 1) keep the previous conformer, stay as close as possible to it
            pick = search_one(sg_xyz, body, cur_src, u0, d_new, tree,
                              fast, FAST_ROLL_STEP, FAST_OFFSETS, "kept")
            stage = "local"
            # 2) same conformer, slightly wider local cone + finer sampling
            if pick["dmin"] < CUTOFF:
                p2 = search_one(sg_xyz, body, cur_src, u0, d_new, tree,
                                wide, WIDE_ROLL_STEP, WIDE_OFFSETS, "kept")
                if p2["dmin"] > pick["dmin"]:
                    pick, stage = p2, "wide"
                if pick["dmin"] < CUTOFF:
                    # 3) same tier, compact-first alternate conformers
                    for label, b, src in bodies[1:MAX_TIER_TRIES + 1]:
                        p3 = search_one(sg_xyz, b, src, u0, d_new, tree,
                                        fast, FAST_ROLL_STEP, FAST_OFFSETS,
                                        label)
                        if p3["dmin"] > pick["dmin"]:
                            pick, stage = p3, "tier"
                            fallback_used.add("%s%d" % site)
                        if pick["dmin"] >= CUTOFF:
                            break
            # A site may still be tight after the first sweep: the other
            # payloads have not moved out of its way yet. The second sweep
            # re-optimises it against their new positions. main() enforces
            # the cutoff on the final report instead of stopping mid-sweep.

            for a_, xyz in zip(group, pick["xyz"]):
                a_["xyz"] = xyz
            report.append(dict(
                site="%s%d" % site,
                previous_optimum_length=round(original_by_site[site], 3),
                remapped_target=round(d_new, 4),
                final_length=round(pick["length"], 4),
                conformer=pick["conformer"], stage=stage,
                axis_tilt_deg=round(pick["tilt_deg"], 1),
                roll_deg=round(pick["roll_deg"], 1),
                min_heavy_dist=round(pick["dmin"], 3)))
    return render(lines, atoms), report, sorted(fallback_used)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="compare regenerated PDBs with the files on disk; "
                         "do not write anything")
    args = ap.parse_args()

    for label, p in [("input structures", SRC), ("conformer ensemble", SDF),
                     ("placement report", PLACEMENT_REPORT)]:
        if not p.exists():
            raise SystemExit("missing %s: %s" % (label, p))
    missing = [f for f in FILES.values() if not (SRC / f).is_file()]
    if missing:
        raise SystemExit("missing input PDBs in %s: %s"
                         % (SRC, ", ".join(missing)))

    lengths = {n: site_lengths(SRC / FILES[n]) for n in FILES}
    d7 = dict(lengths[6])
    d7[("D", 214)] = lengths[4][("D", 214)]
    lengths[7] = {s: d7.get(s, 1.82) for s in lengths[7]}

    ok = True
    for n in sorted(FILES):
        targets = {s: remap(d) for s, d in lengths[n].items()}
        text, rep, fallback = refine(SRC / FILES[n], targets, lengths[n], n)
        dest = DST / FILES[n]
        if args.check:
            same = dest.is_file() and dest.read_text() == text
            ok &= same
            print("DAR%d  %-6s  lengths=%s  dmin=%s  fallback=%s" %
                  (n, "PASS" if same else "FAIL",
                   [r["final_length"] for r in rep],
                   [r["min_heavy_dist"] for r in rep], fallback), flush=True)
        else:
            dest.write_text(text)
            print("DAR%d  wrote %s  lengths=%s  dmin=%s  fallback=%s" %
                  (n, dest.name, [r["final_length"] for r in rep],
                   [r["min_heavy_dist"] for r in rep], fallback), flush=True)

        bad = [r for r in rep
               if not (NEW_LO - 1e-6 <= r["final_length"] <= NEW_HI + 1e-6)
               or r["min_heavy_dist"] < CUTOFF]
        if bad:
            raise RuntimeError("DAR%d produced invalid sites: %s" % (n, bad))

    if args.check and not ok:
        raise SystemExit("check failed: committed PDBs differ from script output")
    print("ALL %s" % ("CHECKS PASSED" if args.check else "STRUCTURES WRITTEN"),
          flush=True)


if __name__ == "__main__":
    main()
