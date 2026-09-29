"""Rebuild the payload conformer library as a solution-realistic ensemble.

The original `payload_ensemble_79.sdf` was generated with ETKDG distance
geometry (useRandomCoords=False), which traps the long flexible vc-MMAE in
extended local minima — every conformer is a straight rod (span ~33-43 A).

In water, vc-MMAE is hydrophobic (clogP ~3-4) and collapses toward a compact
buried state. ETKDGv3 with random coordinates samples this broader space and
recovers collapsed conformers (span ~16 A, Rg ~5.8 A, ~31% lower non-polar
SASA). This script rebuilds a mixed library: collapsed-dominant, a minority
extended — matching the real solution ensemble (mostly folded, some extended).

Hard constraints preserved:
  * 199 atoms with explicit H (94 heavy + 105 H), same atom ordering
  * RDKit atom index 91 == "C91" (reactive maleimide olefin carbon)
  * RDKit atom index 85 == "C85" (linker exit vector)
  * 90-91 is the maleimide C=C (DOUBLE bond)
These are consumed by `config.yaml` chemistry.reactive_atom / exit_atom and
`align_and_place()`. Re-indexing would silently attach the wrong atom.

Output: `payload_ensemble_XX.sdf` written next to the template.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdFreeSASA
from scipy.spatial.distance import pdist

TPL = Path(__file__).resolve().parents[1] / "data" / "MC-Val-Cit-PAB-MMAE_3D.sdf"

# Maleimide ring heavy atoms (5-ring N87 C88 C90 C91 C92 + 2 carbonyl O89 O93).
# The maleimide end must stay solvent-exposed for Cys-SG addition; a collapsed
# pose that buries C91 is physically NOT conjugatable and would be a false
# negative in placement. We therefore require collapsed poses to keep the
# maleimide mostly exposed.
MI_RING = [87, 88, 89, 90, 91, 92, 93]

_OONS_RADII = {6: 1.70, 7: 1.55, 8: 1.52, 1: 1.20, 16: 1.80, 9: 1.47, 17: 1.75, 35: 1.85, 53: 1.98}


def maleimide_sasa(mol, cid: int = 0) -> float:
    """SASA (A^2) of the maleimide ring heavy atoms + their bonded Hs.

    FreeSASA uses the molecule's *current* conformer, so we work on a copy
    that carries exactly conformer `cid` (otherwise every call measures
    conformer 0 regardless of cid, silently defeating per-conformer filters).
    """
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    m.AddConformer(mol.GetConformer(int(cid)), assignId=True)
    radii = {a.GetIdx(): _OONS_RADII.get(a.GetAtomicNum(), 1.70) for a in m.GetAtoms()}
    for a in m.GetAtoms():
        el = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S", 9: "F", 17: "Cl"}.get(a.GetAtomicNum(), "C")
        a.SetProp("SASAClassName", el)
    rdFreeSASA.CalcSASA(m, radii)
    inc = set(MI_RING)
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 1 and any(n.GetIdx() in MI_RING for n in a.GetNeighbors()):
            inc.add(a.GetIdx())
    return sum(a.GetDoubleProp("SASA") for a in m.GetAtoms() if a.GetIdx() in inc)


def mi_exposure_ratio(mol, reference: Chem.Mol, cid: int = 0) -> float:
    """Maleimide SASA / fully-exposed reference SASA (template ~169 A^2)."""
    ref = maleimide_sasa(reference, 0)
    if ref <= 0:
        return 1.0
    return maleimide_sasa(mol, cid) / ref


def span_rg(mol, cid: int = 0) -> tuple[float, float]:
    conf = mol.GetConformer(int(cid))
    heavy = [i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
    x = np.array([conf.GetAtomPosition(i) for i in heavy], dtype=float)
    com = x.mean(0)
    rg = float(np.sqrt(((x - com) ** 2).sum(1).mean()))
    return float(pdist(x).max()), rg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="raw conformers to embed")
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--collapsed-frac", type=float, default=0.6,
                    help="fraction of output that should be collapsed (span<=18 A)")
    ap.add_argument("--extended-frac", type=float, default=0.15)
    ap.add_argument("--max-keep", type=int, default=120)
    ap.add_argument("--raw-sdf", type=str, default=None,
                    help="reuse a cached raw-embedding SDF (skip the 8-min embed)")
    ap.add_argument("--mi-min-exposure", type=float, default=0.7,
                    help="collapsed poses must keep this fraction of maleimide "
                         "SASA vs the fully-exposed template (>0.7 = ~120 A^2)")
    args = ap.parse_args()

    tpl = [x for x in Chem.SDMolSupplier(str(TPL), removeHs=False) if x is not None][0]
    assert tpl.GetNumAtoms() == 199, f"expected 199 atoms, got {tpl.GetNumAtoms()}"
    assert tpl.GetNumHeavyAtoms() == 94

    # sanity: reactive/exit atoms are the right carbons and the maleimide C=C is double
    assert tpl.GetAtomWithIdx(91).GetSymbol() == "C"
    assert tpl.GetAtomWithIdx(85).GetSymbol() == "C"
    assert tpl.GetBondBetweenAtoms(90, 91).GetBondType() == Chem.BondType.DOUBLE

    mol = Chem.Mol(tpl)

    if args.raw_sdf and Path(args.raw_sdf).exists():
        raw = [x for x in Chem.SDMolSupplier(args.raw_sdf, removeHs=False) if x is not None]
        mol.RemoveAllConformers()
        for x in raw:
            mol.AddConformer(x.GetConformer(), assignId=True)
        ids = list(range(len(raw)))
        print(f"loaded {len(ids)} cached conformers from {args.raw_sdf}", flush=True)
    else:
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        params.pruneRmsThresh = 0.5
        params.randomSeed = args.seed
        t0 = time.time()
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=args.n, params=params)
        dt = time.time() - t0
        print(f"embedded {len(ids)}/{args.n} conformers in {dt:.1f}s", flush=True)
        raw_sdf = TPL.parent / "payload_raw_embedding.sdf"
        w = Chem.SDWriter(str(raw_sdf))
        for cid in ids:
            m = Chem.Mol(mol)
            m.RemoveAllConformers()
            m.AddConformer(mol.GetConformer(int(cid)), assignId=True)
            w.write(m)
        w.close()
        print(f"cached raw embedding -> {raw_sdf}", flush=True)

    spans = {cid: span_rg(mol, cid) for cid in ids}
    span_arr = np.array([spans[c][0] for c in ids])

    # --- stratified sampling: collapsed-dominant, some extended.
    # "Collapsed" = span <= 20 A. A 94-heavy-atom vc-MMAE with rigid
    # PABC ring / Val backbone bottoms out at span ~16 A (ETKDGv3 limit),
    # so 16-20 A IS the collapsed globule; 20-25 A is partially folded;
    # >25 A is the extended rod (the old library was all 33-43 A).
    # A collapsed pose qualifies only if the maleimide end stays exposed
    # (>= --mi-min-exposure) — otherwise the pose is physically unconjugatable. ---
    mi_min_exposure = args.mi_min_exposure

    def qualify(cid: int, max_span: float) -> bool:
        if spans[cid][0] > max_span:
            return False
        # A buried maleimide is unconjugatable in ANY pose — filter all.
        if mi_exposure_ratio(mol, tpl, cid) < mi_min_exposure:
            return False
        return True

    collapsed = sorted((c for c in ids if qualify(c, 20.0)), key=lambda c: spans[c][0])
    mid = sorted((c for c in ids if 20.0 < spans[c][0] <= 25.0 and qualify(c, 25.0)),
                 key=lambda c: spans[c][0])
    extended = sorted((c for c in ids if spans[c][0] > 25.0 and qualify(c, 1e9)),
                      key=lambda c: spans[c][0])

    n_out = min(args.max_keep, len(ids))
    n_col = min(int(n_out * args.collapsed_frac), len(collapsed))
    n_ext = min(int(n_out * args.extended_frac), len(extended))
    # fill the rest from mid, then from whatever remains
    n_mid = n_out - n_col - n_ext
    pick_col = collapsed[:n_col]
    pick_ext = extended[-n_ext:] if n_ext else []   # longest-spanned (most extended)
    pick_mid = mid[:n_mid]
    # if mid is exhausted (unlikely), top up from remaining collapsed/extended
    if len(pick_mid) < n_mid:
        for extra in (collapsed[n_col:] + extended[n_ext:]):
            if len(pick_mid) >= n_mid:
                break
            pick_mid.append(extra)
    picked = pick_col + pick_mid + pick_ext

    print(f"picked {len(picked)}: collapsed={len(pick_col)} mid={len(pick_mid)} extended={len(pick_ext)}")

    out_path = Path(args.out) if args.out else TPL.parent / f"payload_ensemble_{len(picked)}.sdf"
    w = Chem.SDWriter(str(out_path))
    for cid in picked:
        m = Chem.Mol(mol)
        m.RemoveAllConformers()
        m.AddConformer(mol.GetConformer(int(cid)), assignId=True)
        w.write(m)
    w.close()

    # report span distribution of the final library
    final_spans = sorted(spans[c][0] for c in picked)
    arr = np.array(final_spans)
    print(f"\n=== final library {len(picked)} conformers ===")
    print(f"span: min={arr.min():.1f} p10={np.percentile(arr,10):.1f} med={np.median(arr):.1f} "
          f"p90={np.percentile(arr,90):.1f} max={arr.max():.1f} A")
    print(f"<=18A: {(arr<=18).sum()}, 18-25A: {((arr>18)&(arr<=25)).sum()}, >25A: {(arr>25).sum()}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()