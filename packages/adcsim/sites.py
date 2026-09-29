from __future__ import annotations

from itertools import combinations

import numpy as np
from Bio.PDB import NeighborSearch, ShrakeRupley  # ShrakeRupley deprecated: use adcsim.sasa.compute_sasa

from .io import parse_site
from .sasa import compute_sasa  # unified SASA core (Biopython adapter)


def list_cysteines(model) -> list[dict]:
    out = []
    for chain in model:
        for residue in chain:
            if residue.get_resname() != "CYS":
                continue
            if residue.id[0] != " ":
                continue
            if "SG" not in residue:
                continue
            out.append(
                {
                    "chain": chain.id,
                    "residue_number": residue.id[1],
                    "icode": residue.id[2],
                    "site": f"{chain.id}:CYS{residue.id[1]}",
                }
            )
    return out


def detect_disulfides(
    model,
    chain_types: dict[str, str],
    intra_cutoff: float = 2.5,
    inter_cutoff: float = 3.0,
) -> list[dict]:
    """Global 1-to-1 matching of Cys SG–SG pairs.

    Intra-chain pairs allowed if d <= intra_cutoff.
    Inter-chain pairs allowed if d <= inter_cutoff AND {H,H} or {H,L}.
    """
    cys = list_cysteines(model)
    candidates = []
    for i, j in combinations(range(len(cys)), 2):
        r1, r2 = cys[i], cys[j]
        res1 = model[r1["chain"]][r1["residue_number"]]
        res2 = model[r2["chain"]][r2["residue_number"]]
        d = float(res1["SG"] - res2["SG"])
        t1 = chain_types[r1["chain"]]
        t2 = chain_types[r2["chain"]]
        if r1["chain"] == r2["chain"] and d <= intra_cutoff:
            bond_type = "intrachain"
        elif r1["chain"] != r2["chain"] and d <= inter_cutoff:
            types = {t1, t2}
            if types in ({"Heavy"}, {"Heavy", "Light"}):
                bond_type = "interchain"
            else:
                continue
        else:
            continue
        # Was hardcoded 2.5, which silently ignored the intra_cutoff parameter.
        strength = "strong" if d <= intra_cutoff else "borderline"
        candidates.append(
            {
                "i": i,
                "j": j,
                "a": r1["site"],
                "b": r2["site"],
                "distance": d,
                "type": bond_type,
                "strength": strength,
            }
        )
    candidates.sort(key=lambda x: x["distance"])
    used = set()
    pairs = []
    for c in candidates:
        if c["i"] in used or c["j"] in used:
            continue
        used.add(c["i"])
        used.add(c["j"])
        pairs.append(c)
    return pairs


def interchain_sites(pairs: list[dict]) -> list[str]:
    sites = []
    for p in pairs:
        if p["type"] != "interchain":
            continue
        sites.append(p["a"])
        sites.append(p["b"])
    return sites


def sasa_table(model, sites: list[str]) -> list[dict]:
    compute_sasa(model)  # unified core: writes atom.sasa on every Atom
    ns = NeighborSearch(list(model.get_atoms()))
    rows = []
    for site in sites:
        chain, resnum = parse_site(site)
        residue = model[chain][(" ", resnum, " ")]
        sg = residue["SG"]
        cys_sasa = float(sum(atom.sasa for atom in residue if hasattr(atom, "sasa")))
        neighbors = [
            a
            for a in ns.search(sg.coord, 4.0, level="A")
            if a.element != "H" and a is not sg
        ]
        rows.append(
            {
                "site": site,
                "cys_sasa": cys_sasa,
                "sg_sasa": float(getattr(sg, "sasa", 0.0) or 0.0),
                "neighbors_4A": len(neighbors),
            }
        )
    return rows


def surface_lysines(model, sasa_min: float = 10.0) -> list[dict]:
    """Find surface-exposed lysines — the conjugation sites for amine-reactive LP.

    In bench terms: when the small molecule grabs amines (NHS ester, isocyanate),
    it can latch onto any lysine whose side-chain -NH2 pokes out of the protein
    surface. Buried lysines are inaccessible, so they are not usable sites. This
    returns the exposed ones, ranked by how much of the side-chain amine is open.

    compute_sasa() (unified core) is called here; it writes atom.sasa.
    """
    compute_sasa(model)
    out = []
    for chain in model:
        for residue in chain:
            if residue.get_resname() != "LYS":
                continue
            if residue.id[0] != " ":
                continue
            nz = residue["NZ"] if "NZ" in residue else None
            if nz is None:
                continue
            nz_sasa = float(getattr(nz, "sasa", 0.0) or 0.0)
            out.append(
                {
                    "chain": chain.id,
                    "residue_number": residue.id[1],
                    "site": f"{chain.id}:LYS{residue.id[1]}",
                    "nz_sasa": round(nz_sasa, 2),
                    "accessible": nz_sasa >= sasa_min,
                }
            )
    out.sort(key=lambda d: d["nz_sasa"], reverse=True)
    return out


def reduction_difficulty_profile(pairs: list[dict], model) -> list[dict]:
    """Per-bond reduction difficulty for cysteine (disulfide) conjugation.

    In bench terms: before the payload can attach, the inter-chain disulfides must
    be reduced open. A bond that is buried (low SG solvent exposure) is harder for
    the reductant to reach than one sitting exposed on the surface. This scores
    each bond by the summed SG SASA of its two cysteines (oxidised structure).

    Thresholds (total SG SASA, Å²): >30 easy, 12–30 medium, <12 hard. These are a
    coarse heuristic (Tier C) — they rank, they do not predict a reduction rate.
    """
    compute_sasa(model)
    rows = []
    for p in pairs:
        # Only the inter-chain disulfides are reduced open to make conjugation
        # sites; structural intra-chain disulfides stay intact and are not part
        # of the "how hard is reduction" picture.
        if p.get("type") != "interchain":
            continue
        cha, ra = parse_site(p["a"])
        chb, rb = parse_site(p["b"])
        try:
            sga = model[cha][ra]["SG"]
            sgb = model[chb][rb]["SG"]
        except Exception:
            continue
        sa = float(getattr(sga, "sasa", 0.0) or 0.0)
        sb = float(getattr(sgb, "sasa", 0.0) or 0.0)
        total = sa + sb
        difficulty = "easy" if total > 30 else ("medium" if total > 12 else "hard")
        rows.append(
            {
                "bond": f"{p['a']}-{p['b']}",
                "type": p["type"],
                "distance_A": round(float(p["distance"]), 2),
                "sg_sasa_A2": round(total, 1),
                "reduction_difficulty": difficulty,
            }
        )
    return rows


def resolve_conjugation_sites(
    model, conj_type: str, chain_types: dict, disulf_cfg: dict, sasa_min: float = 10.0
) -> dict:
    """Route from conjugation type to the sites that should be considered.

    The model's first decision: given the LP warhead (classified elsewhere) and the
    antibody structure, return the reactive sites plus, for cysteine conjugation,
    the disulfide reduction-difficulty profile. For lysine conjugation there is no
    reduction step, so reduction_profile is None and the note says so.
    """
    if conj_type == "lysine":
        lys = surface_lysines(model, sasa_min=sasa_min)
        return dict(
            type="lysine",
            sites=[d["site"] for d in lys if d["accessible"]],
            all_lysines=lys,
            reduction_profile=None,
            note=(
                "lysine conjugation: sites = surface-exposed Lys (no disulfide "
                "reduction step). DAR funnel for lysine is not yet implemented."
            ),
        )
    pairs = detect_disulfides(
        model,
        chain_types,
        intra_cutoff=disulf_cfg.get("intra_cutoff", 2.5),
        inter_cutoff=disulf_cfg.get("inter_cutoff", 3.0),
    )
    sites = interchain_sites(pairs)
    return dict(
        type="cysteine",
        sites=sites,
        pairs=pairs,
        reduction_profile=reduction_difficulty_profile(pairs, model),
        note="cysteine conjugation: sites = inter-chain disulfides; reduction profile below",
    )
