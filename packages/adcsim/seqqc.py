"""Sequence QC (step 0): before running any simulation, give the antibody
sequence a health check.

Answers two questions:
1. Is the premise correct — the model assumes "4 interchain disulfides = 8
   conjugatable thiols"; does this foundation hold up? Get the foundation wrong
   and all five downstream steps are wasted.
2. Where can it move — delineate the variable and locked regions, so that later
   conformational sampling knows which atoms to perturb. The sequence only
   delineates "where it can move"; how much it moves comes from the MD frames,
   which the sequence alone cannot predict.

All data come from a single PDB (a Boltz prediction model, with pLDDT confidence
stored in the B-factor column). No external database, no network access.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np

from .alarms import make_alarm
from . import pdbio  # unified PDB read engine (backing for read_pdb)

# ---------------------------------------------------------------- Constants and sources

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M",
    "SEC": "U", "PYL": "O",
}

# Disulfide-bond detection thresholds. A real S-S single bond is ~2.05 A, but
# prediction-model coordinates deviate, so we relax the cutoff.
# Both values must stay consistent with sites.py (intra/inter cutoff) and
# config.yaml:disulfide — previously this was a single 3.5 A threshold that
# disagreed with the other modules, so the same antibody yielded different
# conjugatable-site counts in seqqc versus the pipeline.
# Used only to "decide pairing relationships", not for energy calculations.
SS_BOND_INTRA = 2.5   # intrachain
SS_BOND_INTER = 3.0   # interchain (prediction-model coordinates are slightly loose, so wider than intrachain)

# pLDDT tiers follow AlphaFold2's official interpretation
# (Jumper et al. 2021, Nature 596:583):
#   >90 very high confidence (backbone and side chains both reliable);
#   70-90 medium (backbone broadly correct, side chains/loops uncertain);
#   50-70 low (backbone unreliable); <50 very low (often corresponds to a
#   disordered region).
# Boltz writes pLDDT in the same units into the B-factor column.
PLDDT_HIGH = 90.0
PLDDT_MED = 70.0
PLDDT_LOW = 50.0

# Domain boundaries (approximate). These use PDB sequential numbering, not
# EU/Kabat numbering.
# Basis for the split: position of the intrachain disulfide — each immunoglobulin
# domain contains exactly one conserved Cys pair.
#   Heavy chain 22-96 in VH, 147-203 in CH1, 264-324 in CH2, 370-428 in CH3.
# The boundaries themselves are approximate; their only purpose is "which domain
# this segment belongs to", not exact numbering.
HC_DOMAINS = [("VH", 1, 119), ("CH1", 120, 215), ("HINGE", 216, 232),
              ("CH2", 233, 340), ("CH3", 341, 449)]
LC_DOMAINS = [("VL", 1, 107), ("CL", 108, 214)]

# CDR positions (approximate Kabat numbering, used directly as sequential
# numbering, with an error of a few residues; only meant to flag "this segment
# may block the antigen-binding site", not a precise definition).
CDR_HC = [("CDR-H1", 26, 35), ("CDR-H2", 50, 65), ("CDR-H3", 95, 102)]
CDR_LC = [("CDR-L1", 24, 34), ("CDR-L2", 50, 56), ("CDR-L3", 89, 97)]

# Distance threshold between degradation liabilities and conjugatable sites.
# 10 A is a common first-order approximation: beyond this distance, side-chain
# chemistry basically cannot reach the conjugation site itself.
HOTSPOT_PROXIMITY = 10.0

# Hinge characteristic motifs (used for isotype calling)
HINGE_MOTIFS = {
    "IgG1": ["CPPCP", "EPKSC", "DKTHT"],
    "IgG2": ["ERK", "CCVECPPCP"],
    "IgG4": ["ESKYGP", "CPSCP"],
    "IgG3": ["ELKTP", "CPRCP"],
}

# Light-chain constant-region signature peptides (N-terminal fragments, used to distinguish kappa / lambda)
LC_KAPPA_MOTIFS = ["TVAAPSVF", "VVCLLNN", "ADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"]
# Note: the NNFYPREAK stretch also appears in kappa (...VVCLLNNFYPREAK...), so it cannot serve as a lambda signature.
LC_LAMBDA_MOTIFS = ["QPKAAPSV", "LVCLISDF", "KSHRSYSCQVTHE"]


# ---------------------------------------------------------------- PDB reading

def read_pdb(path: str) -> Dict[str, Any]:
    """Read the PDB, extracting each chain's sequence, per-residue pLDDT, and the
    Cys SG coordinates.

    Backed by the unified :mod:`adcsim.pdbio` engine. This is the single PDB
    reader for sequence QC; the heavy-atom criterion is now the package-wide
    explicit rule (element not in {H, D}). On hydrogen-free files the output is
    numerically identical to the old line-by-line parser.

    pLDDT is the mean of the (heavy) atom B-factors for that residue (Boltz
    assigns the same value to every residue).
    """
    struct = pdbio.read_pdb(path)
    chains: Dict[str, Any] = {}
    for ch, rid, _ic, resname, recs in pdbio.iter_residues(struct):
        d = chains.setdefault(ch, {"res": {}, "sg": {}})
        # Heavy-atom criterion is explicit and unified (was previously absent here,
        # so hydrogens would have been counted). On H-free files this is a no-op.
        use = [a for a in recs if pdbio.is_heavy(a)] or list(recs)
        bvals = [a.bfactor for a in use]
        xyz = [a.xyz for a in use]
        r = d["res"].setdefault(rid, {"name": resname, "b": [], "xyz": []})
        r["name"] = resname
        r["b"].extend(bvals)
        r["xyz"].extend(xyz)
        for a in recs:
            if a.resname == "CYS" and a.atomname == "SG":
                d["sg"][rid] = a.xyz

    out = {}
    for ch, d in chains.items():
        ids = sorted(d["res"])
        seq = "".join(AA3.get(d["res"][i]["name"], "X") for i in ids)
        plddt = np.array([float(np.mean(d["res"][i]["b"])) for i in ids])
        out[ch] = {
            "res_ids": ids,
            "res_names": [d["res"][i]["name"] for i in ids],
            "seq": seq,
            "plddt": plddt,
            "sg": d["sg"],
            "ca_xyz": np.array([np.mean(d["res"][i]["xyz"], axis=0) for i in ids]),
        }
    return out


def classify_chain(chains: Dict[str, Any], ch: str) -> str:
    """Classify heavy vs light chain by length. 449 residues is heavy-chain scale, 214 is light-chain scale."""
    n = len(chains[ch]["res_ids"])
    return "H" if n > 300 else "L"


# ---------------------------------------------------------------- Disulfide bonds

def find_disulfides(chains: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find disulfides by SG-SG distance, then split into interchain/intrachain by
    whether they cross chains.

    Two key points:
    1. Use distance only, not CONECT records — prediction models often omit
       SSBOND/CONECT.
    2. One cysteine participates in only one bond. Use a greedy, first-come
       exclusive match ordered by ascending distance. Without exclusivity,
       "close but unbound" pairs get counted too: in this example A-Cys223 is
       close to both B-Cys223 (2.79) and C-Cys214 (2.84), and without
       exclusivity two phantom interchain bonds would appear.
    """
    cand = []
    chs = sorted(chains)
    for i in range(len(chs)):
        for j in range(i, len(chs)):
            ca, cb = chs[i], chs[j]
            for ra, pa in chains[ca]["sg"].items():
                for rb, pb in chains[cb]["sg"].items():
                    if ca == cb and ra >= rb:
                        continue
                    dist = float(np.linalg.norm(pa - pb))
                    limit = SS_BOND_INTER if ca != cb else SS_BOND_INTRA
                    if dist <= limit:
                        cand.append({
                            "a": f"{ca}-Cys{ra}", "b": f"{cb}-Cys{rb}",
                            "dist": round(dist, 2),
                            "kind": "inter" if ca != cb else "intra",
                        })
    cand.sort(key=lambda p: p["dist"])

    used, pairs = set(), []
    for p in cand:
        if p["a"] in used or p["b"] in used:
            continue
        used.add(p["a"])
        used.add(p["b"])
        pairs.append(p)
    return pairs


def cys_audit(chains: Dict[str, Any], ss: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Audit the cysteines: total count, the domain each belongs to, and whether any are unpaired."""
    bonded = set()
    for p in ss:
        bonded.add(p["a"])
        bonded.add(p["b"])

    items = []
    for ch in sorted(chains):
        c = chains[ch]
        kind = classify_chain(chains, ch)
        domains = HC_DOMAINS if kind == "H" else LC_DOMAINS
        for i in c["res_ids"]:
            if c["res_names"][c["res_ids"].index(i)] != "CYS":
                continue
            dom = next((nm for nm, a, b in domains if a <= i <= b), "?")
            tag = f"{ch}-Cys{i}"
            items.append({
                "site": tag, "chain": ch, "chain_kind": kind,
                "res_id": i, "domain": dom,
                "paired": tag in bonded,
                "local_plddt": round(float(c["plddt"][c["res_ids"].index(i)]), 1),
            })

    n_inter = sum(1 for p in ss if p["kind"] == "inter")
    n_intra = sum(1 for p in ss if p["kind"] == "intra")
    conj = sorted({p[k] for p in ss if p["kind"] == "inter" for k in ("a", "b")})

    return {
        "n_cys_total": len(items),
        "n_inter_disulfide": n_inter,
        "n_intra_disulfide": n_intra,
        "n_free_cys": sum(1 for it in items if not it["paired"]),
        "n_conjugatable": 2 * n_inter,
        "conjugatable_sites": conj,
        "cys_list": items,
    }


# ---------------------------------------------------------------- Sequence features

def detect_isotype(hc_seq: str) -> Dict[str, Any]:
    """Call the IgG isotype from hinge motifs. Returns the matched isotype and evidence fragments."""
    hits = {}
    for iso, motifs in HINGE_MOTIFS.items():
        found = [m for m in motifs if m in hc_seq]
        if found:
            hits[iso] = found
    if not hits:
        return {"call": "unknown", "evidence": [], "conf": "C"}
    best = max(hits.items(), key=lambda kv: len(kv[1]))
    conf = "A" if len(best[1]) >= 2 else "B"
    return {"call": best[0], "evidence": best[1], "all_hits": hits, "conf": conf}


def detect_light_chain(lc_seq: str) -> Dict[str, Any]:
    k = [m for m in LC_KAPPA_MOTIFS if m in lc_seq]
    l = [m for m in LC_LAMBDA_MOTIFS if m in lc_seq]
    if k and not l:
        return {"call": "kappa", "evidence": k, "conf": "A"}
    if l and not k:
        return {"call": "lambda", "evidence": l, "conf": "A"}
    if k and l:
        return {"call": "ambiguous", "evidence": {"kappa": k, "lambda": l}, "conf": "C"}
    return {"call": "unknown", "evidence": [], "conf": "C"}


def find_glyco_sites(seq: str, offset: int = 0) -> List[Dict[str, Any]]:
    """N-glycosylation sequon N-X-S/T, where X cannot be Pro.

    Source: the classic rule established by Marshall 1972 / Bause 1979
    (sequon N-X-S/T, X≠P).
    Note: having the sequon does not guarantee glycosylation; it also depends on
    whether the site faces the ER lumen and is accessible. So here we only report
    "the sequon is present"; whether sugar is actually attached is left to
    experimental verification (tier C).
    """
    out = []
    for m in re.finditer(r"N([^P])([ST])", seq):
        pos = m.start() + 1 + offset
        out.append({
            "res_id": pos,
            "motif": seq[m.start():m.end()],
            "note": "sequon only; experimental confirmation needed for actual glycosylation",
        })
    return out


def find_liability_motifs(seq: str, offset: int = 0) -> List[Dict[str, Any]]:
    """Common degradation liabilities.

    Sources:
    - Asn deamidation to Asp: NG / NS are fastest (Robinson & Robinson 2001,
      PNAS 98:4367); NH also occurs, but slowly.
    - Asp isomerization/cleavage: DG / DP are most sensitive (Wakankar &
      Borchardt 2006, J Pharm Sci 95:2321).
    - Met oxidation: surface-exposed Met is high risk (Chumsae et al. 2007,
      J Chromatogr B 850:285).
    - N-terminal Gln cyclization to pyroglutamate (Liu et al. 2011, mAbs 3:453).
    Localization only; no rate prediction.
    """
    out = []

    def add(kind, pat, note, rate):
        for m in re.finditer(pat, seq):
            out.append({"kind": kind, "res_id": m.start() + 1 + offset,
                        "motif": m.group(0), "relative_rate": rate, "note": note})

    add("deamidation", r"N[GS]", "Asn deamidation, NG/NS fastest", "high")
    add("deamidation", r"NH", "Asn deamidation, slower", "medium")
    add("Asp cleavage/isomerization", r"D[GP]", "Asp followed by Gly/Pro; isomerization or peptide-bond cleavage under acidic conditions", "high")
    add("Met oxidation", r"M", "methionine oxidation; high risk when surface-exposed", "medium")
    return out


def count_aa(seq: str, aa: str) -> int:
    return seq.count(aa)


# ---------------------------------------------------------------- Variable-region map

def variable_region_map(chains: Dict[str, Any]) -> Dict[str, Any]:
    """Using the pLDDT profile + domain knowledge, delineate the variable and
    locked regions.

    Logic: low pLDDT = the model thinks this segment's structure is uncertain =
    in the real protein this segment is natively mobile. Add two priors: the hinge
    is natively flexible; the core inside a disulfide-bearing domain is rigid.
    """
    out = {}
    for ch in sorted(chains):
        c = chains[ch]
        ids = np.asarray(c["res_ids"])
        p = c["plddt"]
        kind = classify_chain(chains, ch)
        domains = HC_DOMAINS if kind == "H" else LC_DOMAINS

        segs = []
        i = 0
        while i < len(p):
            if p[i] < PLDDT_MED:
                j = i
                while j + 1 < len(p) and p[j + 1] < PLDDT_MED:
                    j += 1
                if j - i + 1 >= 3:
                    segs.append({
                        "start": int(ids[i]), "end": int(ids[j]), "length": j - i + 1,
                        "mean_plddt": round(float(p[i:j + 1].mean()), 1),
                        "min_plddt": round(float(p[i:j + 1].min()), 1),
                        "domain": next((nm for nm, a, b in domains if a <= ids[i] <= b), "?"),
                        "reason": "pLDDT below 70; model considers conformation uncertain",
                    })
                i = j + 1
            else:
                i += 1

        hinge = next((d for d in domains if d[0] == "HINGE"), None)
        if hinge:
            m = (ids >= hinge[1]) & (ids <= hinge[2])
            if m.any():
                segs.append({
                    "start": hinge[1], "end": hinge[2], "length": int(m.sum()),
                    "mean_plddt": round(float(p[m].mean()), 1) if m.any() else None,
                    "min_plddt": round(float(p[m].min()), 1) if m.any() else None,
                    "domain": "HINGE",
                    "reason": "hinge region, natively flexible, independent of pLDDT",
                })

        out[ch] = {
            "chain_kind": kind,
            "n_res": len(ids),
            "plddt_mean": round(float(p.mean()), 1),
            "plddt_min": round(float(p.min()), 1),
            "frac_below_70": round(float((p < PLDDT_MED).mean()), 3),
            "variable_segments": segs,
            "domain_boundaries": [{"domain": nm, "start": a, "end": b} for nm, a, b in domains],
        }
    return out


# ---------------------------------------------------------------- Assembly

def assess(pdb_path: str, md_sites: List[str] | None = None) -> Dict[str, Any]:
    """Main entry point: runs every check in the sequence QC."""
    chains = read_pdb(pdb_path)
    for ch in chains:
        chains[ch]["kind"] = classify_chain(chains, ch)

    ss = find_disulfides(chains)
    cys = cys_audit(chains, ss)

    hc = [ch for ch in chains if chains[ch]["kind"] == "H"]
    lc = [ch for ch in chains if chains[ch]["kind"] == "L"]
    iso = detect_isotype(chains[hc[0]]["seq"]) if hc else {"call": "unknown"}
    lct = detect_light_chain(chains[lc[0]]["seq"]) if lc else {"call": "unknown"}

    glyco, liab = {}, {}

    # SG coordinates of conjugatable sites, used to measure "how far this pit is
    # from the conjugation site".
    # Only liabilities right next to a site affect conjugation; those dozens of
    # residues away in sequence do not count.
    site_xyz = []
    for tag in cys["conjugatable_sites"]:
        ch_, num = tag.split("-Cys")
        site_xyz.append(chains[ch_]["sg"][int(num)])
    site_xyz = np.array(site_xyz) if site_xyz else np.zeros((0, 3))

    for ch in sorted(chains):
        c = chains[ch]
        domains = HC_DOMAINS if c["kind"] == "H" else LC_DOMAINS
        glyco[ch] = find_glyco_sites(c["seq"])
        liab[ch] = find_liability_motifs(c["seq"])
        for g in glyco[ch]:
            g["domain"] = next((nm for nm, a, b in domains if a <= g["res_id"] <= b), "?")
        for l in liab[ch]:
            l["domain"] = next((nm for nm, a, b in domains if a <= l["res_id"] <= b), "?")
            idx = c["res_ids"].index(l["res_id"]) if l["res_id"] in c["res_ids"] else None
            if idx is None or len(site_xyz) == 0:
                l["dist_to_nearest_site"] = None
                l["near_site"] = False
                continue
            dmin = float(np.min(np.linalg.norm(site_xyz - c["ca_xyz"][idx], axis=1)))
            l["dist_to_nearest_site"] = round(dmin, 1)
            l["near_site"] = dmin <= HOTSPOT_PROXIMITY

    vmap = variable_region_map(chains)

    # Consistency check against the existing MD frame-library site list: this is
    # the most important job for "sequence QC" — confirm that the 8 sites every
    # downstream simulation depends on are exactly the same batch the sequence
    # predicts.
    consistency = None
    if md_sites:
        seq_sites = set(cys["conjugatable_sites"])
        md_set = set(md_sites)
        consistency = {
            "md_sites": sorted(md_set),
            "seq_sites": sorted(seq_sites),
            "match": seq_sites == md_set,
            "only_in_md": sorted(md_set - seq_sites),
            "only_in_seq": sorted(seq_sites - md_set),
        }

    return {
        "source_pdb": os.path.abspath(pdb_path),
        "chains": {ch: {"kind": chains[ch]["kind"], "n_res": len(chains[ch]["res_ids"]),
                        "seq": chains[ch]["seq"]} for ch in sorted(chains)},
        "disulfides": ss,
        "cys_audit": cys,
        "conjugation_sites": [it for it in cys["cys_list"]
                              if it["site"] in cys["conjugatable_sites"]],
        "isotype": iso,
        "light_chain": lct,
        "glyco": glyco,
        "liability": liab,
        "variable_map": vmap,
        "md_consistency": consistency,
        "aa_counts": {ch: {"K": count_aa(chains[ch]["seq"], "K"),
                           "H": count_aa(chains[ch]["seq"], "H"),
                           "M": count_aa(chains[ch]["seq"], "M"),
                           "C": count_aa(chains[ch]["seq"], "C")}
                      for ch in sorted(chains)},
    }


def to_alarms(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert to alarms; field names go through alarms.make_alarm (same shape as the other modules)."""
    a: List[Dict[str, Any]] = []

    def add(step, name, value, unit, level, meaning, action, conf="C", reference=""):
        a.append(make_alarm(step, name, value, unit, level, reference,
                            meaning, action, conf))

    c = res["cys_audit"]
    exp = 8
    if c["n_conjugatable"] != exp:
        add("sequence QC", "conjugatable site count differs from expectation", c["n_conjugatable"], "sites", "crit",
            f"Model assumes {exp} conjugatable thiols (4 interchain disulfides); {c['n_conjugatable']} were inferred",
            "Check this antibody's isotype and disulfide topology; reset the site count for combinatorial statistics", "A")
    else:
        add("sequence QC", "conjugatable site count", c["n_conjugatable"], "sites", "ok",
            "4 interchain disulfides -> 8 conjugatable thiols, consistent with the model's premise", "No action needed", "A")

    if c["n_free_cys"] > 0:
        add("sequence QC", "unpaired cysteine present", c["n_free_cys"], "sites", "crit",
            "Unpaired Cys will be labeled directly by maleimide, forming an unintended conjugation site and risking aggregation",
            "Confirm by mass spec whether it is a free thiol; if so, it must be included in the conjugatable-site set", "A")

    mc = res.get("md_consistency")
    if mc is not None:
        add("sequence QC", "site list matches MD frame library", mc["match"], "", "ok" if mc["match"] else "crit",
            "Conjugatable sites inferred from sequence vs. sites in the existing MD frame library",
            "Inconsistency means downstream simulations ran on the wrong sites", "A")

    iso = res["isotype"]
    if iso.get("call") == "IgG1":
        add("sequence QC", "heavy-chain isotype", iso["call"], "", "ok",
            "Hinge motif matches IgG1, consistent with the model's IgG1-only scope", "No action needed", "B")
    else:
        add("sequence QC", "heavy-chain isotype", iso.get("call"), "", "crit",
            "Model is built only for the IgG1 hinge (2 interchain disulfides); non-IgG1 has a different interchain bond count",
            "Use the disulfide topology for the corresponding isotype, or state the interchain bond count explicitly", "A")

    for ch, gs in res["glyco"].items():
        fab = [g for g in gs if g["domain"] in ("VH", "VL", "CH1", "CL")]
        fc = [g for g in gs if g["domain"] in ("CH2", "CH3")]
        if fab:
            add("sequence QC", f"Fab N-glycosylation sequon (chain {ch})", len(fab), "sites", "warn",
                f"Located at {[g['res_id'] for g in fab]}; glycans may block nearby candidate conjugation sites",
                "Confirm actual glycosylation; if present, subtract the occluded sites in site accessibility", "C")
        if fc:
            add("sequence QC", f"Fc N-glycosylation sequon (chain {ch})", len(fc), "sites", "ok",
                f"Located at {[g['res_id'] for g in fc]}; conservative glycosylation, the model's default case", "No action needed", "B")

    hot = [l for ch, ls in res["liability"].items() for l in ls
           if l["relative_rate"] == "high"]
    near = [l for l in hot if l.get("near_site")]
    if near:
        add("sequence QC", "high-risk degradation liability near conjugation site", len(near), "sites", "warn",
            f"High-risk liability within {HOTSPOT_PROXIMITY} A of a conjugation site: "
            + ", ".join(f"{l['motif']}@{l['res_id']}({l['dist_to_nearest_site']}A)"
                        for l in near[:5]),
            "These sites may drift during processing and alter the reactivity of nearby conjugation sites", "C")
    if len(hot) - len(near):
        add("sequence QC", "degradation liability far from conjugation site", len(hot) - len(near), "sites", "ok",
            "Present but not adjacent to a conjugation site; limited impact on conjugation itself", "No action needed", "C")

    sites = res.get("conjugation_sites") or []
    if sites:
        pl = [s["local_plddt"] for s in sites]
        m = float(np.mean(pl))
        lvl = "crit" if m < PLDDT_LOW else ("warn" if m < PLDDT_MED else "ok")
        add("sequence QC", "conformation confidence at conjugation sites", round(m, 1), "pLDDT", lvl,
            f"Mean local pLDDT across the 8 conjugatable sites = {m:.1f} ("
            + ", ".join(f"{s['site']}:{s['local_plddt']}" for s in sites) + ")",
            "Lower pLDDT means the structure is more uncertain; conformational sampling must perturb the low-confidence regions more strongly",
            "B")

    for ch, v in res["variable_map"].items():
        if v["frac_below_70"] > 0.10:
            add("sequence QC", f"chain {ch} low-confidence residue fraction", v["frac_below_70"], "", "warn",
                f"Residues below pLDDT 70 make up {v['frac_below_70']*100:.1f}%; these segments are conformationally uncertain",
                "Increase perturbation amplitude for these segments during conformational sampling", "B")
    return a


def main(argv=None):
    ap = argparse.ArgumentParser(description="Antibody sequence QC (step 0)")
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--md-sites", default="",
                    help="comma-separated existing MD frame-library site names, for consistency checking")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    sites = [s.strip() for s in a.md_sites.split(",") if s.strip()] or None
    res = assess(a.pdb, sites)
    res["alarms"] = to_alarms(res)

    txt = json.dumps(res, ensure_ascii=False, indent=2)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, "w").write(txt)
        print(f"[written] {a.out}")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
