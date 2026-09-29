"""Site microenvironments (the first correction term in this package that varies
within a single isotype).

Background
----
Most tier-A correction terms (isotype, kappa/lambda, Cys count, unpaired Cys,
glycosylation, degradation liabilities) barely change across antibodies of the
same isotype — swap in a different IgG1 kappa mAb and the result is identical.
The only term that truly returns different values for different antibodies is the
"charge environment around each conjugation site".

Physics (in one line)
------------------
For a thiol to react with maleimide it must first shed a proton and become the
anionic thiolate (S-). How hard that is is governed by pKa, and pKa is pushed
around by the surrounding charges:
  · nearby positive charges (Lys/Arg) → the anionic S- is stabilized → pKa drops → reacts faster
  · nearby negative charges (Asp/Glu) → S- is repelled → pKa rises → reacts slower

So this step counts the ionizable groups within 10 Å of each conjugation site and
folds them into a pKa shift ΔpKa via Coulomb interactions (with salt screening).

Which step it feeds
------------
It feeds the "thiol activation" step directly (pKa → activation fraction →
reaction probability). Steric hindrance is NOT handled here — that belongs to the
"site accessibility" step's SASA; here we only report packing density as a
qualitative reference and do not use it in any numeric correction.

Honest boundaries (must be stated, no overselling)
------------------------------
1. ε_eff (effective dielectric constant) is a model assumption, not a literature
   value. There is no consensus on what the protein surface should use, so we
   scan three values (20/40/80) and a claim only counts if it holds across all three.
2. ΔpKa is semi-quantitative: the Coulomb model ignores desolvation, hydrogen
   bonding, and conformational relaxation. Its trustworthy use is "ranking the
   eight sites within the same antibody relative to one another", not an absolute pKa.
3. When propagated to the final DAR, under the current process window this
   correction's effect may be zero — the script computes the "critical time /
   critical molar equivalents" explicitly and lets the numbers speak, rather than
   asserting "there is an effect" without evidence.

Data sources
--------
· Coulomb constant 332.06 kcal·Å/(mol·e²): matches packages/adcsim/electrostatics.py
· Debye screening κ = 0.329 × √I(M) Å⁻¹ (aqueous standard form, 25 °C); I=0.16 M →
  screening length ≈ 7.8 Å
· Ionizable-group pKa (textbook values; essentially fully ionized at pH 7, so the
  error does not affect the conclusions):
  Asp 3.9 / Glu 4.3 / His 6.0 / Cys 8.5 / Tyr 10.1 / Lys 10.5 / Arg 12.5
  N-term 8.0 / C-term 3.1
· Genentech (Junutula 2008, Nat Biotechnol; Shen 2012, Nat Biotechnol) found that
  maleimide on sites with "low solvent exposure + local positive charge" more
  readily undergoes hydrolytic ring-opening, so reverse Michael exchange does not
  occur — this affects only "how stable it is once attached", not DAR, so it is
  listed separately.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .alarms import CRIT, INFO, WARN, make_alarm

# Step name (this project dropped the numbered naming; the bare name says what it answers)
STEP = "thiol activation"

# ------------------------------------------------------------------ Constants

COULOMB = 332.06              # kcal/mol · Å / e²  (must match electrostatics._QC)
R_KCAL = 1.98720425e-3        # kcal/(mol·K)       (must match payload_qc.R_KBT)
EPS_REF = 40.0                # primary ε_eff; EPS_SCAN brackets it
DEFAULT_PH = 7.0
DEFAULT_IONIC_MM = 160.0      # PBS approximation (adc_model_config.yaml)
DEFAULT_TEMP_C = 4.0          # ice-bath conjugation (adc_model_config.yaml)
DEFAULT_CUTOFF = 10.0         # Å, radius for the local-charge census
EPS_SCAN = [20.0, 40.0, 80.0]  # ε_eff scan; a claim must hold across all three
# SG-SG distance below which a cysteine counts as disulfide-bonded.
# Same value as sites.intra_cutoff / config.yaml:disulfide.intra_cutoff.
SS_BOND_CUTOFF = 2.5

# Textbook pKa (neutral side-chain dissociation)
PKA = {
    "ASP": 3.9, "GLU": 4.3, "HIS": 6.0, "CYS": 8.5,
    "TYR": 10.1, "LYS": 10.5, "ARG": 12.5,
    "N_TERM": 8.0, "C_TERM": 3.1,
}
# Acidic groups (negatively charged when deprotonated) get the minus sign; basic groups (positively charged when protonated) get the plus sign
SIGN = {"ASP": -1, "GLU": -1, "TYR": -1, "CYS": -1, "C_TERM": -1,
        "HIS": +1, "LYS": +1, "ARG": +1, "N_TERM": +1}

# Charge-center atoms: take the geometric center of the atoms carrying the most charge in the group
CHARGE_ATOMS = {
    "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "LYS": ("NZ",),
    "ARG": ("CZ",),
    "TYR": ("OH",),
    "CYS": ("SG",),
}
# Bulky side chains (used only as a steric reference; not used in numeric correction)
BULKY = {"TRP", "TYR", "PHE", "ARG", "LYS", "MET", "HIS", "GLU", "LEU"}


# ------------------------------------------------------------------ PDB parsing

def parse_pdb(path: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Read the PDB heavy atoms, returning (atom list, coordinates in Å).

    Each entry in the atom list: chain / resid / resname / atomname / idx

    Backed by the unified :mod:`adcsim.pdbio` engine. The heavy-atom criterion
    (element not in {H, D}) is now the package-wide rule, identical to what this
    function used to apply via ``el == 'H'``.
    """
    import adcsim.pdbio as pdbio
    struct = pdbio.read_pdb(path)
    recs = pdbio.heavy_atoms(struct)
    atoms: List[Dict[str, Any]] = []
    xyz: List[List[float]] = []
    for idx, a in enumerate(recs):
        atoms.append({
            "chain": a.chain,
            "resid": a.resid,
            "resname": a.resname,
            "atomname": a.atomname,
            "idx": idx,
        })
        xyz.append([float(a.xyz[0]), float(a.xyz[1]), float(a.xyz[2])])
    return atoms, np.array(xyz, dtype=float)


def residue_index(atoms: List[Dict[str, Any]]) -> Dict[Tuple[str, int], List[int]]:
    """(chain, resid) -> list of atom indices for that residue."""
    out: Dict[Tuple[str, int], List[int]] = {}
    for a in atoms:
        out.setdefault((a["chain"], a["resid"]), []).append(a["idx"])
    return out


def ionizable_groups(
    atoms: List[Dict[str, Any]],
    coords: np.ndarray,
    ph: float = DEFAULT_PH,
    exclude: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Identify all ionizable groups, giving the charge-center coordinates and the
    actual charge at this pH.

    The actual charge is not an integer: His is only ~9% protonated at pH 7 and Tyr
    is barely ionized at all. Using Henderson-Hasselbalch to compute the fractional
    charge is closer to reality than assigning ±1.
    """
    res = residue_index(atoms)
    name_of = {(a["chain"], a["resid"]): a["resname"] for a in atoms}
    groups: List[Dict[str, Any]] = []
    exclude = exclude or set()

    # First/last residue in a chain, used for terminal charges
    chain_resids: Dict[str, List[int]] = {}
    for (ch, rid) in res:
        chain_resids.setdefault(ch, []).append(rid)
    for ch in chain_resids:
        chain_resids[ch].sort()

    for (ch, rid), idxs in sorted(res.items()):
        if (ch, rid) in exclude:
            continue
        rn = name_of[(ch, rid)]
        by_name = {atoms[i]["atomname"]: i for i in idxs}

        cands: List[Tuple[str, float]] = []   # (kind, pKa)
        if rn in ("ASP", "GLU", "HIS", "LYS", "ARG", "TYR"):
            cands.append((rn, PKA[rn]))
        if rn == "CYS":
            # Count only free thiols. Cys involved in a disulfide carries no charge; skip.
            # Test: another SG within 2.5 Å of this SG → already bonded.
            if "SG" in by_name:
                sg = coords[by_name["SG"]]
                near = [a for a in atoms
                        if a["atomname"] == "SG" and a["resname"] == "CYS"
                        and np.linalg.norm(coords[a["idx"]] - sg) < SS_BOND_CUTOFF
                        and not (a["chain"] == ch and a["resid"] == rid)]
                if not near:
                    cands.append(("CYS", PKA["CYS"]))
        # Termini
        if rid == chain_resids[ch][0] and "N" in by_name:
            cands.append(("N_TERM", PKA["N_TERM"]))
        if rid == chain_resids[ch][-1] and ("OXT" in by_name or "O" in by_name):
            cands.append(("C_TERM", PKA["C_TERM"]))

        for kind, pka in cands:
            s = SIGN[kind]
            # Fractional charge: acidic s=-1 → q = -1/(1+10^(pKa-pH)); basic s=+1 → q = +1/(1+10^(pH-pKa))
            q = s / (1.0 + 10.0 ** (s * (ph - pka))) if s > 0 else s / (1.0 + 10.0 ** (pka - ph))
            # Charge-center atoms
            if kind in CHARGE_ATOMS:
                pick = [by_name[n] for n in CHARGE_ATOMS[kind] if n in by_name]
            elif kind == "N_TERM":
                pick = [by_name["N"]]
            else:
                pick = [by_name["OXT"]] if "OXT" in by_name else [by_name["O"]]
            if not pick:
                continue
            center = coords[pick].mean(axis=0)
            if abs(q) < 0.01:          # groups like Tyr that carry almost no charge at pH 7: skip
                continue
            groups.append({
                "site": f"{ch}-{rn}{rid}",
                "kind": kind,
                "chain": ch,
                "resid": rid,
                "pka": pka,
                "charge": round(float(q), 4),
                "center": center,
                "atom_idx": pick,      # for recomputing the center over MD frames
            })
    return groups


# ------------------------------------------------------------------ Core computation

def debye_kappa(ionic_mM: float, temp_C: float = DEFAULT_TEMP_C) -> float:
    """Debye screening constant κ (Å⁻¹). Aqueous standard form: κ = 0.329 × √I(M).

    0.329 is the constant at 298.15 K, but this process runs at 4 °C (ice bath).
    κ ∝ 1/√(ε·T), so here we scale by √(298.15/T) — at 4 °C κ is ~3.5% larger than
    at room temperature, and the screening length accordingly shrinks from 7.6 Å to
    7.3 Å. Leaving it unscaled is fine at room temperature but systematically
    overestimates the contribution of distant charges at low temperature.
    """
    return 0.329 * np.sqrt(max(ionic_mM, 1e-6) / 1000.0) * np.sqrt(298.15 / (temp_C + 273.15))


def dpka_for_site(
    sg: np.ndarray,
    groups: List[Dict[str, Any]],
    cutoff: float = DEFAULT_CUTOFF,
    eps_eff: float = 40.0,
    ionic_mM: float = DEFAULT_IONIC_MM,
    temp_C: float = DEFAULT_TEMP_C,
) -> Dict[str, Any]:
    """Compute the pKa shift for one site.

    Coulomb interaction energy (with Debye screening):
        ΔG = Σ_i  332.06/ε_eff × q_probe × q_i / r_i × exp(−κ·r_i)   [kcal/mol]
    Probe charge q_probe = −1 (we are computing the product state S⁻; the reactant
    RSH is neutral and unaffected).
    Convert to pKa:
        ΔpKa = ΔG / (2.303·R·T)
    """
    kappa = debye_kappa(ionic_mM, temp_C)
    rt = 2.303 * R_KCAL * (temp_C + 273.15)   # kcal/mol per pKa unit
    dg = 0.0
    near: List[Dict[str, Any]] = []
    for g in groups:
        r = float(np.linalg.norm(g["center"] - sg))
        if r > cutoff or r < 1e-6:
            continue
        contrib = COULOMB / eps_eff * (-1.0) * g["charge"] / r * float(np.exp(-kappa * r))
        dg += contrib
        near.append({
            "site": g.get("site", "?"), "kind": g.get("kind", "?"),
            "charge": g["charge"],
            "dist": round(r, 2), "dpka_contrib": round(contrib / rt, 4),
        })
    near.sort(key=lambda d: d["dist"])
    return {
        "dpka": round(dg / rt, 4),
        "dG_kcal": round(dg, 4),
        "n_groups": len(near),
        "net_charge": round(sum(n["charge"] for n in near), 3),
        "near": near,
    }


def activation_fraction(pka: float, ph: float) -> float:
    """Activation fraction: the proportion of thiols in the S⁻ (thiolate) form."""
    return 1.0 / (1.0 + 10.0 ** (pka - ph))


def critical_condition(
    k2_ref: float, f_ref: float, f_new: float,
    mal_uM: float, p_target: float = 0.99,
    conj_time_h: Optional[float] = None,
) -> Dict[str, float]:
    """Translate this site's change in activation fraction into "will the reaction
    still finish?".

    Covalent-conjugation step probability: P = 1 − exp(−k2_eff · [Mal] · t)
    k2_eff = k2_ref × (f_new / f_ref)   — k2_ref is the apparent rate measured at the
    reference pH, which already includes the activation fraction under those
    conditions, so here we only scale the relative ratio.

    Returns: the time needed to reach P_target, and the actual P at the current
    process time.
    """
    k2_new = k2_ref * (f_new / f_ref)
    c = k2_new * (mal_uM * 1e-6)          # s⁻¹
    t_for_target = -np.log(1.0 - p_target) / c if c > 0 else float("inf")
    out = {
        "k2_eff": round(k2_new, 3),
        "t_for_target_s": round(float(t_for_target), 2),
        "t_for_target_min": round(float(t_for_target) / 60.0, 3),
    }
    # How far it actually gets at the current process time — the docstring promised this number
    if conj_time_h is not None:
        t_proc_s = conj_time_h * 3600.0
        out["p_at_process_time"] = round(float(1.0 - np.exp(-c * t_proc_s)), 6)
        out["process_time_min"] = round(conj_time_h * 60.0, 1)
    return out


# ------------------------------------------------------------------ Main routine

def load_sites(atoms: List[Dict[str, Any]], coords: np.ndarray,
               site_names: List[str]) -> Dict[str, np.ndarray]:
    """Given a name like A-Cys223, find the coordinates of that residue's SG atom."""
    out = {}
    for s in site_names:
        try:
            ch, rest = s.split("-")
            rn, rid = rest[:3].upper(), int(rest[3:])
        except Exception:
            raise SystemExit(f"Bad site name format: {s} (expected e.g. A-Cys223)")
        hit = [a for a in atoms
               if a["chain"] == ch and a["resid"] == rid
               and a["atomname"] == "SG" and a["resname"].upper() == rn]
        if not hit:
            raise SystemExit(f"Could not find the SG atom of {s} in the PDB")
        out[s] = coords[hit[0]["idx"]]
    return out


def md_trajectory(md_dir: str, site: str) -> Optional[np.ndarray]:
    """Read this site's MD frame library, returning the coordinate trajectory (Å).

    Unit conversion (nm stored -> Å returned) is delegated to the single canonical
    loader in adcsim.io so it can never disagree with the structural pipeline.
    """
    if not md_dir:
        return None
    for p in glob.glob(os.path.join(md_dir, f"{site}*.npz")):
        d = np.load(p, allow_pickle=True)
        if "frames" in d.files:
            from .io import load_md_frames
            return load_md_frames(p)
    return None


def check_alignment(atoms: List[Dict[str, Any]], coords: np.ndarray,
                    md_dir: str, sites: List[str]) -> List[Dict[str, Any]]:
    """Check whether the MD frame library's atom numbering lines up with the PDB
    (if it doesn't, everything downstream is wrong).

    Note: the MD system's atom numbering is not a one-to-one match with the PDB —
    the MD has a few more atoms than the PDB, and the offset accumulates per chain
    (measured in this system: A +0, B +1, C +3, D +4). So instead of testing for
    equality we measure each chain's offset here and correct uniformly later.
    """
    rows = []
    by_key = {}
    for a in atoms:
        by_key[(a["chain"], a["resid"], a["atomname"])] = a["idx"]
    import collections
    offsets: Dict[str, List[int]] = collections.defaultdict(list)
    for s in sites:
        ch, rest = s.split("-")
        rid = int(rest[3:])
        pdb_i = by_key.get((ch, rid, "SG"))
        pth = glob.glob(os.path.join(md_dir, f"{s}*.npz"))
        if pdb_i is None or not pth:
            rows.append({"site": s, "chain": ch, "pdb_idx": pdb_i,
                         "md_sg_i": None, "offset": None, "ok": False})
            continue
        d = np.load(pth[0], allow_pickle=True)
        md_i = int(d["sg_i"]) if "sg_i" in d.files else None
        off = None if md_i is None else md_i - pdb_i
        if off is not None:
            offsets[ch].append(off)
        # Cross-check: the PDB CA-SG distance should be ~2.9 Å, and the same pair in the MD frame should be too
        ca_i = by_key.get((ch, rid, "CA"))
        md_ca = int(d["ca_i"]) if "ca_i" in d.files else None
        traj = md_trajectory(md_dir, s)
        ok = False
        if traj is not None and ca_i is not None and md_ca is not None and md_i is not None:
            dp = float(np.linalg.norm(coords[pdb_i] - coords[ca_i]))
            dm = float(np.linalg.norm(traj[0][md_i] - traj[0][md_ca]))
            ok = abs(dp - dm) < 0.3
        rows.append({"site": s, "chain": ch, "pdb_idx": pdb_i,
                     "md_sg_i": md_i, "offset": off, "ok": bool(ok)})

    # Take the mode per chain as that chain's unified offset
    chain_off = {ch: int(np.median(v)) for ch, v in offsets.items()}
    for r in rows:
        r["chain_offset"] = chain_off.get(r["chain"])
        if r["offset"] is not None:
            r["consistent"] = (r["offset"] == chain_off.get(r["chain"]))
        else:
            r["consistent"] = False
    return rows


def assess(
    pdb: str,
    sites: List[str],
    md_dir: str = "",
    ph: float = DEFAULT_PH,
    ionic_mM: float = DEFAULT_IONIC_MM,
    temp_C: float = DEFAULT_TEMP_C,
    cutoff: float = DEFAULT_CUTOFF,
    pka_ref: float = 8.5,
    k2_ref: float = 300.0,
    mal_uM: float = 402.0,
    conj_time_h: float = 2.0,
    partner_pairs: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    atoms, coords = parse_pdb(pdb)
    sg_of = load_sites(atoms, coords, sites)

    # Exclusions: the site itself (its charge is exactly what we are solving for) + its bonded partner Cys
    #   the partner is also a thiol after reduction, but at pH 7 its dissociation is only ~3%; 3% × 3% is negligible
    exclude = set()
    for s in sites:
        ch, rest = s.split("-")
        exclude.add((ch, int(rest[3:])))
    if partner_pairs:
        for a, b in partner_pairs:
            for x in (a, b):
                ch, rest = x.split("-")
                exclude.add((ch, int(rest[3:])))

    groups = ionizable_groups(atoms, coords, ph=ph, exclude=exclude)

    # Steric reference: number of bulky side chains within each site's cutoff
    res = residue_index(atoms)
    name_of = {(a["chain"], a["resid"]): a["resname"] for a in atoms}
    center_of = {k: coords[v].mean(axis=0) for k, v in res.items()}

    f_ref = activation_fraction(pka_ref, ph)
    out_sites: List[Dict[str, Any]] = []
    align = check_alignment(atoms, coords, md_dir, sites) if md_dir else []
    chain_offsets = {r["chain"]: r["chain_offset"] for r in align
                     if r.get("chain_offset") is not None}

    for s in sites:
        sg = sg_of[s]
        # ε_eff three-value scan
        scan = {}
        for eps in EPS_SCAN:
            scan[str(eps)] = dpka_for_site(sg, groups, cutoff, eps, ionic_mM, temp_C)["dpka"]
        main = dpka_for_site(sg, groups, cutoff, 40.0, ionic_mM, temp_C)

        # Steric reference
        bulky = []
        for k, c in center_of.items():
            if k in exclude:
                continue
            if name_of.get(k) in BULKY and float(np.linalg.norm(c - sg)) <= cutoff:
                bulky.append(f"{k[0]}-{name_of[k]}{k[1]}")

        # MD frames: how much ΔpKa jitters under thermal motion (group centers and the site SG are both re-sampled per frame)
        fluct = None
        traj = md_trajectory(md_dir, s)
        if traj is not None:
            sg_i = None
            for r in align:
                if r["site"] == s:
                    sg_i = r.get("md_sg_i")
            fluct = _dpka_over_frames(traj, groups, sg, sg_i, chain_offsets,
                                      cutoff, ionic_mM, temp_C)

        # With MD, use the multi-frame mean — a single structure (especially the
        # hinge, an intrinsically disordered region) can have a charge environment
        # far from the dynamic average; in this system the sign even flipped.
        dpka_used = fluct["mean"] if fluct else main["dpka"]
        pka_new = pka_ref + dpka_used
        f_new = activation_fraction(pka_new, ph)
        crit = critical_condition(k2_ref, f_ref, f_new, mal_uM,
                                  conj_time_h=conj_time_h)
        # Align with alarms.py's convention: it expects thiolate_pct as a percentage, not a 0~1 fraction
        thiolate_pct = f_new * 100.0

        out_sites.append({
            "site": s,
            "dpka": round(dpka_used, 4),
            "dpka_static": main["dpka"],
            "dpka_sd": fluct["sd"] if fluct else None,
            "dpka_scan": scan,
            "dpka_range": round(max(scan.values()) - min(scan.values()), 4),
            "pka_eff": round(pka_new, 3),
            "f_activation_ref": round(f_ref, 5),
            "f_activation": round(f_new, 5),
            "thiolate_pct": round(thiolate_pct, 4),   # percentage convention, read directly by alarms.py
            "f_ratio": round(f_new / f_ref, 3),
            "n_charge_groups": main["n_groups"],
            "net_charge_10A": main["net_charge"],
            "top_groups": main["near"][:6],
            "bulky_neighbors": bulky[:8],
            "n_bulky": len(bulky),
            "critical": crit,
            "dpka_over_frames": fluct,
        })

    vals = [s["dpka"] for s in out_sites]
    if not vals:                      # do not raise ValueError on an empty site list
        raise SystemExit("Site list is empty; cannot compute the spread (check that --sites and the PDB's chain/residue numbering match)")
    return {
        "inputs": {
            "pdb": pdb, "md_dir": md_dir or None, "ph": ph,
            "ionic_mM": ionic_mM, "temp_C": temp_C, "cutoff_A": cutoff,
            "pka_ref": pka_ref, "k2_ref": k2_ref, "mal_uM": mal_uM,
            "conj_time_h": conj_time_h,
            "debye_length_A": round(1.0 / debye_kappa(ionic_mM, temp_C), 2),
        },
        "md_alignment": align,
        "n_ionizable_groups": len(groups),
        "sites": out_sites,
        "spread": {
            "dpka_min": round(min(vals), 3),
            "dpka_max": round(max(vals), 3),
            "dpka_spread": round(max(vals) - min(vals), 3),
            "f_ratio_min": round(min(s["f_ratio"] for s in out_sites), 3),
            "f_ratio_max": round(max(s["f_ratio"] for s in out_sites), 3),
        },
        "alarms": to_alarms(out_sites),
    }


def _dpka_over_frames(traj: np.ndarray, groups: List[Dict[str, Any]],
                      sg0: np.ndarray, sg_i: Optional[int],
                      chain_offsets: Dict[str, int],
                      cutoff: float, ionic_mM: float, temp_C: float,
                      step: int = 5) -> Optional[Dict[str, Any]]:
    """Recompute ΔpKa over the MD frames to see how much it jitters.

    Both the group centers and the site SG are re-sampled per frame (using the
    atom_idx stored by ionizable_groups).
    Stride sampling: adjacent frames are not independent, so taking them
    consecutively would underestimate the jitter.
    """
    if any(g.get("atom_idx") is None for g in groups):
        return None
    vals = []
    for fi in range(0, traj.shape[0], step):
        sub = traj[fi]
        sg = sub[int(sg_i)] if sg_i is not None and sg_i < sub.shape[0] else sg0
        gs = []
        for g in groups:
            off = chain_offsets.get(g["chain"], 0)
            ii = np.array(g["atom_idx"]) + off
            if ii.max() >= sub.shape[0]:
                return None          # index out of range means alignment was never established; better to report nothing
            c = sub[ii].mean(axis=0)
            gs.append({"charge": g["charge"], "center": c})
        vals.append(dpka_for_site(sg, gs, cutoff, 40.0, ionic_mM, temp_C)["dpka"])
    if not vals:
        return None
    v = np.array(vals)
    return {
        "n_frames_used": len(v),
        "mean": round(float(v.mean()), 4),
        "sd": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4),
        "min": round(float(v.min()), 4),
        "max": round(float(v.max()), 4),
    }


def to_alarms(sites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Alarms: report only "sites worth noting", not "definitely a problem".

    Field names go through alarms.make_alarm uniformly, to avoid dropping fields
    when reports are concatenated across layers.
    """
    out = []
    for s in sites:
        if abs(s["dpka"]) >= 0.5:
            meaning = ("Net charge around the site is significant; " +
                       ("positive charge lowers pKa, activation speeds up" if s["dpka"] < 0
                        else "negative charge raises pKa, activation slows down"))
            out.append(make_alarm(
                STEP, f"{s['site']} local charge deviation", s["dpka"], "pKa units", WARN,
                "±0.5", meaning,
                "This site's reactivity will differ from the others; if site-occupancy spectra are measured, prioritize comparing against this one."))
        if s["dpka_range"] >= 0.3:
            out.append(make_alarm(
                STEP, f"{s['site']} ΔpKa sensitive to dielectric constant", s["dpka_range"],
                "pKa units", INFO, "< 0.3",
                "With epsilon_eff ranging from 20 to 80 the conclusion changes, indicating this site sits very close to charged groups",
                "Do not cite its absolute pKa; use it only for relative comparison between sites."))
        if s["critical"]["t_for_target_min"] > 120.0:
            out.append(make_alarm(
                STEP, f"{s['site']} insufficient activation, reaction incomplete",
                s["critical"]["t_for_target_min"], "min", CRIT, "≤ 120",
                f"At the current loading, reaching 99% reaction requires "
                f"{s['critical']['t_for_target_min']:.1f} min",
                "Extend conjugation time or raise pH."))
    return out


# ------------------------------------------------------------------ CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="Site microenvironment: how local charge shifts the thiol pKa")
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--sites", default="A-Cys223,A-Cys229,A-Cys232,B-Cys223,"
                                       "B-Cys229,B-Cys232,C-Cys214,D-Cys214")
    ap.add_argument("--md-dir", default="", help="MD frame-library directory (optional; used to estimate fluctuation)")
    ap.add_argument("--ph", type=float, default=DEFAULT_PH)
    ap.add_argument("--ionic-mM", type=float, default=DEFAULT_IONIC_MM)
    ap.add_argument("--temp-C", type=float, default=DEFAULT_TEMP_C)
    ap.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF)
    ap.add_argument("--pka-ref", type=float, default=8.5)
    ap.add_argument("--k2", type=float, default=300.0)
    ap.add_argument("--mal-uM", type=float, default=402.0)
    ap.add_argument("--conj-time-h", type=float, default=2.0)
    ap.add_argument("--partners", default="A-Cys229:B-Cys229,A-Cys232:B-Cys232,"
                                          "A-Cys223:C-Cys214,B-Cys223:D-Cys214")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    partners = []
    if a.partners:
        for pair in a.partners.split(","):
            x, y = pair.split(":")
            partners.append((x.strip(), y.strip()))

    res = assess(
        pdb=a.pdb, sites=[s.strip() for s in a.sites.split(",")],
        md_dir=a.md_dir, ph=a.ph, ionic_mM=a.ionic_mM, temp_C=a.temp_C,
        cutoff=a.cutoff, pka_ref=a.pka_ref, k2_ref=a.k2, mal_uM=a.mal_uM,
        conj_time_h=a.conj_time_h, partner_pairs=partners,
    )

    sp = res["spread"]
    print("=" * 74)
    print("Site microenvironment: local charge -> thiol pKa shift")
    print("=" * 74)
    print(f"Process conditions  pH {a.ph} | ionic strength {a.ionic_mM} mM (Debye length "
          f"{res['inputs']['debye_length_A']} Å) | {a.temp_C} °C")
    print(f"Reference pKa {a.pka_ref} -> reference activation fraction {activation_fraction(a.pka_ref, a.ph)*100:.2f}%")
    print(f"Total ionizable groups {res['n_ionizable_groups']} (site itself and bonded partners excluded)")
    if res["md_alignment"]:
        ok = sum(1 for r in res["md_alignment"] if r["ok"])
        print(f"MD frame-library check: {ok}/{len(res['md_alignment'])} sites passed bond-length cross-validation; "
              f"chain offsets {dict(sorted({r['chain']: r['chain_offset'] for r in res['md_alignment']}.items()))}")
        print("ΔpKa uses the multi-frame mean (a single structure's charge environment can differ greatly from the dynamic average; this system even showed a sign flip)")
    print()
    print(f"{'Site':<12}{'ΔpKa':>8}{'sd':>8}{'Static':>8}{'pKa':>7}"
          f"{'Act%':>8}{'Ratio':>7}{'net_q':>8}{'groups':>8}")
    print("-" * 74)
    for s in res["sites"]:
        sd = f"{s['dpka_sd']:.2f}" if s["dpka_sd"] is not None else "  -"
        print(f"{s['site']:<12}{s['dpka']:>8.2f}{sd:>8}{s['dpka_static']:>8.2f}"
              f"{s['pka_eff']:>7.2f}{s['f_activation']*100:>8.2f}{s['f_ratio']:>7.2f}"
              f"{s['net_charge_10A']:>8.1f}{s['n_charge_groups']:>8d}")
    print("-" * 74)
    print(f"Inter-site ΔpKa spread {sp['dpka_spread']:.2f} units "
          f"({sp['dpka_min']:.2f} ~ {sp['dpka_max']:.2f})")
    print(f"Activation-fraction ratio range {sp['f_ratio_min']:.2f}x ~ {sp['f_ratio_max']:.2f}x")
    print()
    print("ε_eff sensitivity (three scans 20/40/80; a claim must hold across all three):")
    for s in res["sites"]:
        sc = s["dpka_scan"]
        print(f"  {s['site']:<12} ε20 {sc['20.0']:+.2f}   ε40 {sc['40.0']:+.2f}   "
              f"ε80 {sc['80.0']:+.2f}   spread {s['dpka_range']:.2f}")
    print()
    print("Propagation to covalent conjugation (time to reach P=99%; current process 2 h = 120 min):")
    for s in res["sites"]:
        t = s["critical"]["t_for_target_min"]
        flag = "far exceeds process time -> falls short" if t > 120 else "completes within process time"
        print(f"  {s['site']:<12} k2_eff {s['critical']['k2_eff']:>8.1f} M⁻¹s⁻¹   "
              f"needs {t:>8.3f} min   {flag}")
    tmax = max(s["critical"]["t_for_target_min"] for s in res["sites"])
    tproc = a.conj_time_h * 60.0
    print(f"\n  Slowest site needs {tmax:.2f} min, process allows {tproc:.0f} min "
          f"-> margin {tproc / tmax:.0f}x")
    if tproc / tmax > 10:
        print("  Conclusion: under the current process window, site-charge differences do NOT change the DAR — the reaction time is far in excess,")
        print("        all sites already have a conjugation probability of 1. For this correction to matter, the conjugation time would have to be")
        print(f"        compressed to around {tmax:.1f} min, or the payload loading molar equivalents drastically reduced.")
    else:
        print("  Conclusion: process time is near the critical point, so site-charge differences will genuinely affect each site's conjugation probability.")
    print()
    if res["alarms"]:
        print("Alarms:")
        for al in res["alarms"]:
            print(f"  [{al['level']}] {al['item']}: {al['value']} {al['unit']}")
            print(f"         {al['meaning']}")
    else:
        print("Alarms: none")
    print()
    print("Nearest groups overview (site -> top 3 charged groups):")
    for s in res["sites"]:
        tops = ", ".join(f"{g['site']}({g['charge']:+.2f}, {g['dist']}Å)"
                         for g in s["top_groups"][:3]) or "none"
        print(f"  {s['site']:<12} {tops}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"\nWritten to {a.out}")


if __name__ == "__main__":
    main()
