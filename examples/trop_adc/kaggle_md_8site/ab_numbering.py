#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
antibody_numbering.py — antibody conjugation-site numbering scheme (structure-driven, zero external dependencies)

## Why not ANARCI / IMGT
IMGT numbering solves cross-antibody sequence alignment, needs HMMER + germline libraries, heavy.
Conjugation-site prediction only needs 'which structural position on the antibody this site is' — this can be derived purely from structure,
and the derived information (which domain) is more explanatory for ease of reduction than IMGT numbering.

## Address format
    <half>.<domain>.<in-domain index>.<residue>
    H1.CH1.106.Cys      H2.HINGE.2.Cys      L1.CL.107.Cys

All four segments are derived from structure, rules below (no hard-coded residue numbers, no offsets):

1. Chain type H/L
   primary criterion = the chain's intrachain disulfide pair count: heavy chain 4 pairs (VH/CH1/CH2/CH3), light chain 2 pairs (VL/CL).
   secondary criterion = residue count (H ~440-460, L ~210-220); any chain in a heavy-heavy disulfide must be H.

2. Half (symmetry partner)
   use interchain disulfides for union-find connectivity: A-C connected -> half 1; B-D connected -> half 2.
   pure structural derivation, independent of chain-ID letter order.
   -> A223 and B223 get identical addresses (differ only in H1/H2), 'symmetry partner' is encoded in the name.

3. Domain boundaries (key: use intrachain disulfides as natural domain markers)
   The hallmark of an Ig domain is a conserved intrachain disulfide spanning 55-75 aa.
   a) midpoint of each intrachain disulfide pair -> domain center
   b) midpoint between adjacent domain centers -> domain boundary
   c) if a domain contains a heavy-heavy interchain disulfide -> split at the smallest HH residue,
      the latter part goes to the hinge (hinge = segment with no intrachain disulfide, containing HH bonds)
   no offset: the split point is exactly min(HH residue) itself.

4. Domain naming
   heavy chain N->C order: VH, CH1, HINGE, CH2, CH3 (HINGE appears only when HH bonds detected)
   light chain N->C order: VL, CL

## Self-check (the 'control' of scientific judgment)
Symmetry partners must yield fully identical addresses except for the half number. If all 4 mirror-site pairs match,
the rules are self-consistent; a mismatch means a bug in the rules or parsing. This is the script's negative control.
"""

import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PDB = os.path.join(HERE, "B220_235_disulfide_repaired.pdb")
OUT = os.path.join(HERE, "..", "results_hinge", "ab_numbering.json")

# domain names (N->C order)
DOMAIN_NAMES_H = ["VH", "CH1", "CH2", "CH3"]   # HINGE inserted by position
DOMAIN_NAMES_L = ["VL", "CL"]


# ---------------------------------------------------------------- parse
def read_pdb(path):
    """Return (chain -> [(resseq, resname), ...]) and the SSBOND list"""
    chains = defaultdict(dict)      # chain -> {resseq: resname}
    order = []                      # preserve first-seen chain order
    for line in open(path):
        if line.startswith(("ATOM", "HETATM")):
            ch = line[21]
            try:
                ri = int(line[22:26])
            except ValueError:
                continue
            rn = line[17:20].strip()
            if ch not in chains:
                order.append(ch)
            chains[ch][ri] = rn

    ssbond = []
    for line in open(path):
        if line.startswith("SSBOND"):
            # fixed columns: CYS A  223    CYS C  214
            c1, r1 = line[15], int(line[17:21])
            c2, r2 = line[29], int(line[31:35])
            ssbond.append((c1, r1, c2, r2))

    reslist = {ch: sorted(chains[ch].items()) for ch in order}
    return reslist, ssbond, order


# ---------------------------------------------------------------- chain type
def classify_chains(reslist, ssbond, order):
    """Classify chain type. primary = intrachain disulfide pair count; secondary = residue count / participates in HH bond"""
    intra_n = defaultdict(int)
    hh_chains = set()
    for c1, r1, c2, r2 in ssbond:
        if c1 == c2:
            intra_n[c1] += 1
        else:
            # both chains heavy -> HH hinge bond
            pass
    # second pass for HH: needs chain type, coarse-split by intra_n first then confirm
    types = {}
    for ch in order:
        n_res = len(reslist[ch])
        n_intra = intra_n[ch]
        if n_intra >= 3:
            t = "H"
        elif n_intra <= 2:
            t = "L"
        else:
            t = "H" if n_res > 300 else "L"
        # corroboration: residue count
        if t == "H" and n_res < 300:
            t = "L"
        if t == "L" and n_res > 300:
            t = "H"
        types[ch] = t

    # use chain type to confirm HH bonds (both ends H)
    for c1, r1, c2, r2 in ssbond:
        if c1 != c2 and types.get(c1) == "H" and types.get(c2) == "H":
            hh_chains.add(c1)
            hh_chains.add(c2)
    return types, intra_n, hh_chains


# ---------------------------------------------------------------- halves
def assign_halves(ssbond, order, types):
    """Use heavy-light disulfide for union-find connectivity -> half numbering (1-based, stable order)

    Note: only H-L bonds. If all interchain bonds are used for connectivity, the hinge H-H bonds
    (A229-B229 / A232-B232) would join the two heavy chains into one, collapsing four halves into one.
    """
    parent = {c: c for c in order}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for c1, r1, c2, r2 in ssbond:
        if c1 == c2:
            continue                                  # intrachain bonds excluded
        t1, t2 = types.get(c1), types.get(c2)
        if {t1, t2} == {"H", "L"}:                    # only H-L bonds define halves
            union(c1, c2)

    groups = defaultdict(list)
    for ch in order:
        groups[find(ch)].append(ch)
    halves = {}
    for i, (_, members) in enumerate(
            sorted(groups.items(), key=lambda kv: min(order.index(m) for m in kv[1])), 1):
        for ch in members:
            halves[ch] = i
    return halves


# ---------------------------------------------------------------- domain
def build_domains(ch, resseqs, ssbond, ctype, halves):
    """Return [(domain name, start residue, end residue), ...]"""
    intra_pairs = [(r1, r2) for c1, r1, c2, r2 in ssbond
                   if c1 == c2 == ch and r1 < r2]
    intra_pairs.sort()
    if not intra_pairs:
        return [("UNKNOWN", min(resseqs), max(resseqs))]

    centers = [0.5 * (a + b) for a, b in intra_pairs]
    lo, hi = min(resseqs), max(resseqs)

    # domain boundary = midpoint between adjacent domain centers
    bounds = []
    for i in range(len(centers) - 1):
        bounds.append(int(round(0.5 * (centers[i] + centers[i + 1]))))
    segs = []
    prev = lo
    for b in bounds:
        segs.append((prev, b))
        prev = b + 1
    segs.append((prev, hi))

    # naming
    names = list(DOMAIN_NAMES_H if ctype == "H" else DOMAIN_NAMES_L)
    if len(names) < len(segs):                      # more domains than standard -> fallback numbering
        names += [f"D{i+1}" for i in range(len(names), len(segs))]
    names = names[:len(segs)]

    # hinge criterion: heavy-heavy interchain disulfide with identical residue numbers on both ends (hinge symmetric bond).
    # no offset — split point is exactly min(HH residue) itself.
    # note: must check both ends: in SSBOND records chain B may appear only in the second column (A229-B229)
    hh_res = sorted({r1 for c1, r1, c2, r2 in ssbond
                     if c1 == ch and c2 != ch and r1 == r2}
                    | {r2 for c1, r1, c2, r2 in ssbond
                       if c2 == ch and c1 != ch and r1 == r2})

    out = []
    for (s, e), nm in zip(segs, names):
        hh_in = [r for r in hh_res if s <= r <= e]
        if hh_in and ctype == "H":
            cut = hh_in[0]
            if cut > s:                       # split: first half stays in the original domain
                out.append((nm, s, cut - 1))
            out.append(("HINGE", cut, e))     # latter part + remaining gap of this segment = hinge
        else:
            out.append((nm, s, e))
    return out


# ---------------------------------------------------------------- main flow
def main(pdb=PDB, out=OUT):
    reslist, ssbond, order = read_pdb(pdb)
    types, intra_n, hh_chains = classify_chains(reslist, ssbond, order)
    halves = assign_halves(ssbond, order, types)

    print("=" * 74)
    print("antibody site numbering (custom scheme, structure-driven)")
    print("=" * 74)
    print(f"\n[1] chain-type classification (primary = intrachain disulfide pair count)")
    print(f"  {'chain':>3}{'n_res':>8}{'intra_SS':>8}{'half':>8}{'type':>6}")
    for ch in order:
        print(f"  {ch:>3}{len(reslist[ch]):>8}{intra_n[ch]:>8}{halves[ch]:>8}{types[ch]:>6}")

    # domain
    domains = {}
    print(f"\n[2] domain segmentation (intrachain disulfide midpoint -> domain center; hinge split at HH bond)")
    for ch in order:
        resseqs = [r for r, _ in reslist[ch]]
        ds = build_domains(ch, resseqs, ssbond, types[ch], halves)
        domains[ch] = ds
        print(f"  {ch}({types[ch]}{halves[ch]}): "
              + " | ".join(f"{nm}[{s}-{e}]" for nm, s, e in ds))

    def address(ch, resseq):
        for nm, s, e in domains[ch]:
            if s <= resseq <= e:
                idx = resseq - s + 1
                rn = reslist[ch][dict(reslist[ch])[resseq]] \
                    if False else _resname(reslist[ch], resseq)
                return f"{types[ch]}{halves[ch]}.{nm}.{idx}.{rn}"
        return f"{types[ch]}{halves[ch]}.NA.{resseq}."

    # all cysteines
    cys = {}
    for ch in order:
        for r, rn in reslist[ch]:
            if rn == "CYS" or rn == "CYX":
                cys[(ch, r)] = address(ch, r)

    print(f"\n[3] cysteine address mapping ({len(cys)} total)")
    # print by SSBOND
    for c1, r1, c2, r2 in ssbond:
        kind = "intrachain" if c1 == c2 else "interchain"
        print(f"  {kind}  {c1}{r1:>4} - {c2}{r2:>4}   "
              f"{cys.get((c1,r1),'?'):<20} {cys.get((c2,r2),'?'):<20}")

    # ---- self-check: symmetry-partner addresses must match (except half number) ----
    print(f"\n[4] self-check: C2 symmetry-partner address consistency (negative control)")
    # group interchain bonds by (chain_type1, chain_type2, r1, r2); different halves within a group = mirror partners
    grp = defaultdict(list)
    for c1, r1, c2, r2 in ssbond:
        if c1 != c2 and not (types[c1] == "H" and types[c2] == "H"):
            grp[(types[c1], types[c2], r1, r2)].append((c1, r1, c2, r2))

    strip = lambda s: s.split(".", 1)[1] if "." in s else s
    ok, npair, nwarn = True, 0, 0
    for key, members in grp.items():
        if len(members) < 2:
            print(f"  WARN  {key} only {len(members)} entries, no mirror partner")
            nwarn += 1
            continue
        ref = members[0]
        for m in members[1:]:
            if halves[ref[0]] == halves[m[0]]:
                continue
            a1, a2 = cys.get((ref[0], ref[1])), cys.get((ref[2], ref[3]))
            b1, b2 = cys.get((m[0], m[1])), cys.get((m[2], m[3]))
            same = strip(a1) == strip(b1) and strip(a2) == strip(b2)
            ok = ok and same
            npair += 1
            print(f"  {'OK ' if same else 'FAIL'}  "
                  f"{a1} - {a2}   <->   {b1} - {b2}")
    print(f"  => {npair} mirror-site pairs, "
          f"{'all consistent (rules self-consistent)' if ok and npair else 'inconsistency found, rules have a bug'}"
          f"{f', {nwarn} entries without partner' if nwarn else ''}")

    # HH hinge bonds span two halves; their two ends are themselves symmetry partners, validated separately
    print(f"  -- HH hinge bonds (ends are mutual symmetry partners) --")
    for c1, r1, c2, r2 in ssbond:
        if c1 == c2 or not (types[c1] == "H" and types[c2] == "H"):
            continue
        a, b = cys.get((c1, r1)), cys.get((c2, r2))
        if a is None or b is None:
            continue
        same = strip(a) == strip(b)
        ok = ok and same
        npair += 1
        print(f"  {'OK ' if same else 'FAIL'}  {a}  <->  {b}")

    # output
    data = dict(
        pdb=os.path.basename(pdb),
        rule=dict(chain_type="intrachain disulfide pair count (H>=3, L<=2) + residue-count corroboration",
                  half="interchain disulfide union-find connectivity",
                  domain="intrachain disulfide midpoint -> domain center; adjacent centers' midpoint -> boundary; HINGE split at HH bond",
                  address="<half>.<domain>.<in-domain index>.<residue>"),
        chains={ch: dict(type=types[ch], half=halves[ch],
                         n_res=len(reslist[ch]), n_intra_ss=intra_n[ch],
                         domains=[[nm, s, e] for nm, s, e in domains[ch]])
                for ch in order},
        cys_address={f"{c}{r}": a for (c, r), a in sorted(cys.items())},
        # legacy naming (e.g. "A-Cys214") -> new address, for direct replacement of existing artifacts
        legacy_map={f"{c}-Cys{r}": a
                    for (c, r), a in sorted(cys.items())},
        ssbond=[dict(c1=c1, r1=r1, c2=c2, r2=r2,
                     kind="inter" if c1 != c2 else "intra",
                     addr1=cys.get((c1, r1)), addr2=cys.get((c2, r2)))
                for c1, r1, c2, r2 in ssbond],
        symmetry_check_passed=bool(ok),
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(data, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\nwritten: {os.path.relpath(out)}")
    return data


def _resname(reslist_ch, resseq):
    for r, rn in reslist_ch:
        if r == resseq:
            return rn.capitalize()
    return "Xxx"


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else PDB)
