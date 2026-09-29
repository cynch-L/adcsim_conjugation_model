#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ss_reduction_profile.py  (v2 — physics-corrected version)
==========================================
Compute a "reduction ease" proxy metric for each interchain disulfide bond.

[v2 key correction]
v1 mistakenly used the "reduced-state MD dynamic SASA" as the accessibility when the reductant attacks — this is circular reasoning:
that SASA is measured only after the disulfide has already been broken.
The reductant (TCEP/DTT) attacks the **bonded (oxidized) state** structure, so accessibility must be computed from the
**complete structure, with all disulfides in the bonded state, using static SG SASA**.

Physical basis
--------
The rate of TCEP/DTT disulfide reduction is mainly determined by three factors:
  1. accessibility: solvent-accessible surface area of SG in the bonded state (prerequisite for SN2 attack)
  2. conformational strain: the farther from optimal geometry, the easier to reduce
       - optimal S-S bond length ≈ 2.05 A
       - CB-SG-SG-CB dihedral energy minimum at ±90 deg, secondary minimum at ±180 deg (circular)
       - CB-SG-SG bond angle ≈ 103 deg
  3. local electrostatics: nearby positive charges (Lys/Arg/His) stabilize the thiolate transition state,
     lowering the local pKa and accelerating the reaction (only a qualitative counting proxy, not a PB calculation)

strain proxy (demo-level, not rigorous energy):
    dev_len  = |d(SG-SG) - 2.05| / 0.15
    dev_dih  = circular_min(|dih|, {90, 180}) / 45
    dev_ang  = |angle(CB-SG-SG) - 103| / 15
    strain   = clip(0.40*dev_len + 0.40*dev_dih + 0.20*dev_ang, 0, 2)

reduction_ease (0..1):
    ease = clip(0.55*accessibility + 0.30*(strain/1.5) + 0.15*electrostatic, 0, 1)

Output: results_hinge/ss_reduction_profile.json
"""
import json, os, sys, math
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # .../trop_adc
PDB  = sys.argv[1] if len(sys.argv) > 1 else \
       os.path.join(ROOT, "data", "B220_235_disulfide_repaired.pdb")
RES  = os.path.join(ROOT, "results_hinge")
TAG  = sys.argv[2] if len(sys.argv) > 2 else ""
OUT  = os.path.join(RES, f"ss_reduction_profile{('_' + TAG) if TAG else ''}.json")

# Disulfide detection cutoff. MUST equal adcsim.seqqc.SS_BOND_INTER (3.0 A).
# It was hardcoded at 2.5 here while seqqc used 3.0, so the two modules found a
# different number of interchain bonds on the same structure: on the AlphaFold
# model_2 structure seqqc found 4 bonds and this script found 3, silently
# dropping the A-Cys229--B-Cys229 pair at 2.60 A. A test now pins the two
# together (tests/test_structure_qc.py).
SS_CUTOFF   = 3.0

D_SS_OPT    = 2.05
DIH_TARGETS = (90.0, 180.0)
ANG_OPT     = 103.0
PROBE       = 1.4
NPOINT      = 720
BONDI = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.85, "P": 1.80,
         "H": 1.20, "F": 1.47, "CL": 1.75, "BR": 1.50, "I": 1.70,
         "SE": 1.90, "MG": 1.73, "CA": 2.14, "NA": 2.27, "K": 2.75}

# ------------------------------------------------------------------ parse
def parse_pdb(path):
    atoms = []
    with open(path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            name = line[12:16].strip(); resn = line[17:20].strip()
            ch = line[21].strip(); rid = int(line[22:26])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            el = line[76:78].strip()
            if not el:
                el = name[0] if name else "C"
            if el.upper().startswith("H"):
                continue
            atoms.append(dict(name=name, resn=resn, ch=ch, rid=rid,
                              xyz=np.array([x, y, z]), el=el.upper()))
    return atoms

def get(atoms, ch, rid, name):
    for a in atoms:
        if a["ch"] == ch and a["rid"] == rid and a["name"] == name:
            return a
    return None

def dist(a, b):
    return float(np.linalg.norm(a["xyz"] - b["xyz"]))

def angle(a, b, c):
    v1 = a["xyz"] - b["xyz"]; v2 = c["xyz"] - b["xyz"]
    cs = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return math.degrees(math.acos(max(-1.0, min(1.0, cs))))

def dihedral(p0, p1, p2, p3):
    b0 = p0["xyz"] - p1["xyz"]; b1 = p2["xyz"] - p1["xyz"]; b2 = p3["xyz"] - p2["xyz"]
    b1n = b1 / (np.linalg.norm(b1) + 1e-12)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    return math.degrees(math.atan2(np.dot(np.cross(b1n, v), w), np.dot(v, w)))

def circ_dev(x, targets):
    """Minimum circular deviation of x from targets (0..180)"""
    x = x % 360.0
    return min(min(abs(x - t), 360 - abs(x - t)) for t in targets)

# ------------------------------------------------------------------ static SASA
def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], axis=1)

def static_sasa(atoms, target_idx):
    """Shrake-Rupley SASA (A^2) for a single atom, with all other heavy atoms as occluders"""
    coords = np.array([a["xyz"] for a in atoms]) / 10.0          # nm
    radii = np.array([BONDI.get(a["el"], 1.70) for a in atoms]) / 10.0
    r_x = radii + PROBE / 10.0
    tree = cKDTree(coords)
    i = target_idx
    sg_x = r_x[i]
    cut = sg_x + r_x.max() + 0.05
    nb = np.asarray(tree.query_ball_point(coords[i], cut), dtype=np.int64)
    nb = nb[nb != i]
    DIRS = fib_sphere(NPOINT)
    pts = coords[i] + sg_x * DIRS
    d = np.sqrt(((pts[:, None, :] - coords[nb][None, :, :]) ** 2).sum(-1))
    blocked = (d < r_x[nb][None, :]).any(axis=1)
    exposed = int((~blocked).sum())
    return exposed / NPOINT * 4 * np.pi * sg_x ** 2 * 100        # A^2

# ------------------------------------------------------------------ env
POSITIVE = {"LYS", "ARG", "HIS", "HIP"}


def residue_charge(resn, ph=7.0):
    """Signed side-chain charge of a residue at this pH (formal charges only).

    HIS is partially protonated (pKa ~6.0); CYS is partially deprotonated
    (thiol pKa ~8.5) so it carries a small negative charge even at pH 7.
    These are the two residues whose charge actually moves across the pH
    window an ADC process uses.
    """
    if resn in ("LYS", "ARG"):
        return +1.0
    if resn in ("ASP", "GLU"):
        return -1.0
    if resn in ("HIS", "HIP", "HIE", "HID"):
        return 1.0 / (1.0 + 10.0 ** (ph - 6.0))
    if resn == "CYS":
        return -1.0 / (1.0 + 10.0 ** (8.5 - ph))
    return 0.0


def nearby_positive(atoms, sg, exclude, cutoff=8.0):
    n = 0
    for a in atoms:
        if a["name"] != "CA" or a["resn"] not in POSITIVE:
            continue
        if (a["ch"], a["rid"]) in exclude:
            continue
        if dist(a, sg) <= cutoff:
            n += 1
    return n


def local_net_charge(atoms, sg, exclude, cutoff=8.0, ph=7.0):
    """Signed net charge of all side chains within `cutoff` A of this SG atom.

    Distinct from `nearby_positive` on purpose, and both are kept:
      - nearby_positive feeds the `electrostatic` term INSIDE reduction_ease.
        That term is a PRODUCT-side effect (positive charges stabilise the
        thiolate that reduction leaves behind), and it is already part of the
        ease numbers the model was calibrated on - do not touch it.
      - this function is the REACTANT-side effect: TCEP is an anion at process
        pH (phosphorus pKa 7.6), so a negatively charged patch repels it and
        slows the attack; a positive patch attracts it. That correction is NOT
        in ease and is applied separately by the model, bounded.
    """
    q = 0.0
    for a in atoms:
        if a["name"] != "CA":
            continue
        z = residue_charge(a["resn"], ph)
        if z == 0.0:
            continue
        if (a["ch"], a["rid"]) in exclude:
            continue
        if dist(a, sg) <= cutoff:
            q += z
    return q

# ------------------------------------------------------------------ main
def main():
    atoms = parse_pdb(PDB)
    idx_of = {id(a): i for i, a in enumerate(atoms)}
    cys = [a for a in atoms if a["resn"] == "CYS" and a["name"] == "SG"]
    print(f"parsed {len(atoms)} heavy atoms, {len(cys)} CYS SG")

    pairs, used = [], set()
    for i in range(len(cys)):
        for j in range(i + 1, len(cys)):
            a, b = cys[i], cys[j]
            d = dist(a, b)
            if d < SS_CUTOFF:
                pairs.append((a, b, d)); used.add((a["ch"], a["rid"])); used.add((b["ch"], b["rid"]))
    inter = [(a, b, d) for a, b, d in pairs if a["ch"] != b["ch"]]
    print(f"SS bonds: {len(pairs)} total, {len(inter)} inter-chain")

    rows = []
    for a, b, d_ss in inter:
        cb_a = get(atoms, a["ch"], a["rid"], "CB")
        cb_b = get(atoms, b["ch"], b["rid"], "CB")
        ca_a = get(atoms, a["ch"], a["rid"], "CA")
        ca_b = get(atoms, b["ch"], b["rid"], "CA")

        dih = abs(dihedral(cb_a, a, b, cb_b)) if (cb_a and cb_b) else None
        ang1 = angle(cb_a, a, b) if cb_a else None
        ang2 = angle(cb_b, b, a) if cb_b else None
        ang_mean = float(np.mean([x for x in (ang1, ang2) if x is not None]))

        dev_len = abs(d_ss - D_SS_OPT) / 0.15
        dev_dih = (circ_dev(abs(dih), DIH_TARGETS) / 45.0) if dih is not None else 1.0
        dev_ang = abs(ang_mean - ANG_OPT) / 15.0
        strain = float(np.clip(0.40 * dev_len + 0.40 * dev_dih + 0.20 * dev_ang, 0, 2))

        # ---- v2: bonded-state static SASA (accessibility when the reductant truly attacks) ----
        sa = static_sasa(atoms, idx_of[id(a)])
        sb = static_sasa(atoms, idx_of[id(b)])
        acc = float(np.clip(max(sa, sb) / 60.0, 0.0, 1.0))

        excl = {(a["ch"], a["rid"]), (b["ch"], b["rid"])}
        npos = nearby_positive(atoms, a, excl) + nearby_positive(atoms, b, excl)
        elec = float(np.clip(npos / 3.0, 0.0, 1.0))

        # reactant-side electrostatics: TCEP is an anion at process pH, so a
        # net-negative patch near the bond repels it. Computed for BOTH SG
        # atoms and averaged; the main model reads this field and applies a
        # bounded rate multiplier (charge_factor).
        q_local = (local_net_charge(atoms, a, excl, ph=7.0)
                   + local_net_charge(atoms, b, excl, ph=7.0)) / 2.0

        ease = float(np.clip(0.55 * acc + 0.30 * (strain / 1.5) + 0.15 * elec, 0.0, 1.0))
        cls = "easy" if ease >= 0.60 else ("mid" if ease >= 0.35 else "hard")
        kind = "HH-hinge" if (a["ch"] in "AB" and b["ch"] in "AB") else "HL-interface"

        rows.append(dict(
            site_a=f"{a['ch']}-Cys{a['rid']}", site_b=f"{b['ch']}-Cys{b['rid']}",
            kind=kind, sg_sg_A=round(d_ss, 3),
            ca_ca_A=round(dist(ca_a, ca_b), 3) if (ca_a and ca_b) else None,
            cb_sg_sg_cb_dih_deg=round(dih, 2) if dih is not None else None,
            cb_sg_sg_angle_deg=round(ang_mean, 2),
            dev_len=round(float(dev_len), 3), dev_dih=round(float(dev_dih), 3),
            dev_ang=round(float(dev_ang), 3), strain=round(strain, 3),
            # bonded-state static SASA (key correction)
            sasa_oxidized_a_A2=round(sa, 2), sasa_oxidized_b_A2=round(sb, 2),
            accessibility=round(acc, 3),
            n_positive_8A=npos, electrostatic=round(elec, 3),
            local_net_charge=round(q_local, 2),
            reduction_ease=round(ease, 3), reduction_class=cls,
        ))
        print(f"  {rows[-1]['site_a']}--{rows[-1]['site_b']}: "
              f"static SASA {sa:.1f}/{sb:.1f} A^2, strain {strain:.2f}")

    rows.sort(key=lambda r: -r["reduction_ease"])
    out = dict(
        method="structural reduction-ease proxy v2 (OXIDIZED-state static SASA + geometry strain + local positive charge)",
        pdb=os.path.basename(PDB),
        n_interchain_ss=len(inter),
        note=("v2 fix: accessibility uses oxidized-state static SG SASA (what TCEP actually "
              "attacks), NOT the reduced-state dynamic SASA from MD (which is circular)."),
        formula="ease = clip(0.55*accessibility + 0.30*(strain/1.5) + 0.15*electrostatic, 0, 1)",
        strain_def="clip(0.40*|dSS-2.05|/0.15 + 0.40*circ_dev(dih,{90,180})/45 + 0.20*|ang-103|/15, 0, 2)",
        pairs=rows,
    )
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)

    print("\n" + "=" * 108)
    print(f"{'pair':<24}{'kind':<14}{'dSS':>6}{'dih':>7}{'strain':>7}"
          f"{'SASAx_a':>9}{'SASAx_b':>9}{'acc':>6}{'elec':>6}{'EASE':>7}  class")
    print("-" * 108)
    for r in rows:
        print(f"{r['site_a']+'--'+r['site_b']:<24}{r['kind']:<14}"
              f"{r['sg_sg_A']:>6.2f}{(r['cb_sg_sg_cb_dih_deg'] or 0):>7.1f}{r['strain']:>7.2f}"
              f"{r['sasa_oxidized_a_A2']:>9.1f}{r['sasa_oxidized_b_A2']:>9.1f}"
              f"{r['accessibility']:>6.2f}{r['electrostatic']:>6.2f}"
              f"{r['reduction_ease']:>7.3f}  {r['reduction_class']}")
    print("=" * 108)
    print("saved ->", OUT)

if __name__ == "__main__":
    main()
