"""Cheap, reproducible score card for a polished ADC PDB.

Only quantities that can be honestly computed from a single PDB:

- rg / span / reach   MMAE compactness (smaller = more folded onto Fc)
- surface_contact     LPP heavy atoms within 2.4-4.5 A of the protein
- cdr_min_dist        min distance LPP heavy -> nearest CDR CA (biological: does
                      conjugation crowd the antigen-binding site?)
- mmff_stress         vacuum MMFF single-point on each LPP (too-tight fold penalized)

NOT computed here (and not faked):
- sidechain perturbation after conjugation: the laptop polish keeps the protein
  rigid, so there is nothing to measure. This only becomes meaningful after
  RunPod `constrained_min.py` releases the 8 A side-chain shell; then compute
  the sidechain heavy-atom RMSD vs the unperturbed pose.
- Delta-G / solubility / experimental DAR: wet-lab or expensive methods.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import cKDTree

from .polish import _ligand_mol, _xyz, ligand_xyz, load_adc_pdb
# SASA now goes through the unified core's RDKit adapter (adcsim.sasa.sasa_of_rdkit)
# instead of solvation.sasa, so LP SASA uses the same algorithm + OONS radii as the
# antibody side. solvation.sasa remains only as the legacy engine used by polish.py.
from .sasa import sasa_of_rdkit as sasa

# Trastuzumab CDR, Kabat numbering (verified against the actual residue names
# and chain layout of the hinge PDB: A/B=heavy 449 residues, C/D=light 214).
HEAVY_CDR = {
    "H1": range(26, 33),   # GFNIKDT
    "H2": range(50, 66),   # RIYPTNGYTRYADSVKG
    "H3": range(99, 110),  # WGGDGFYAMDY
}
LIGHT_CDR = {
    "L1": range(24, 35),   # RASQDVNTAVA
    "L2": range(50, 57),   # SASFLYS
    "L3": range(89, 98),   # QQHYTTPPT
}


def _cdr_ca_coords(lines: list[str]) -> np.ndarray:
    pts = []
    for ln in lines:
        if not ln.startswith("ATOM"):
            continue
        if ln[12:16].strip() != "CA":
            continue
        chain = ln[21]
        try:
            resi = int(ln[22:26])
        except ValueError:
            continue
        table = HEAVY_CDR if chain in ("A", "B") else LIGHT_CDR
        if any(resi in r for r in table.values()):
            pts.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    return np.asarray(pts, dtype=float) if pts else np.zeros((0, 3))


def _lpp_heavy_coords(adc: dict) -> tuple[np.ndarray, list[np.ndarray]]:
    ligs = sorted(adc["ligands"])
    all_xyz = []
    per = []
    for rid in ligs:
        xyz = ligand_xyz(adc["ligands"][rid])
        per.append(xyz)
        all_xyz.append(xyz)
    if not all_xyz:
        return np.zeros((0, 3)), []
    return np.vstack(all_xyz), per


def _mmff_stress_kcal(adc: dict) -> float | None:
    """Sum of vacuum MMFF94s single-point energies over all LPP residues."""
    total = 0.0
    n = 0
    for lig in adc["ligands"].values():
        if lig.get("c91") is None:
            continue
        try:
            mol, index = _ligand_mol(lig, adc["conect"])
            mol_h = Chem.AddHs(mol, addCoords=True)
            # embed hydrogens first; raw added-H coords give exploded energies
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=50)
            props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
            if props is None:
                continue
            if not AllChem.MMFFHasAllMoleculeParams(mol_h):
                continue  # skip instead of reporting a bogus number
            ff = AllChem.MMFFGetMoleculeForceField(mol_h, props, ignoreInterfragInteractions=True)
            total += ff.CalcEnergy()
            n += 1
        except Exception:
            continue
    return round(total, 1) if n else None


def _hydrophobic_exposure(adc: dict) -> float | None:
    """Total non-polar SASA (Å²) summed over all LPP residues — proxy for
    aggregation risk.  Higher DAR → more exposed hydrophobic payload surface
    → higher risk of self-association and aggregation.

    Uses the same OONS-based FreeSASA as solvation.py.  Returns None if
    RDKit cannot reconstruct the ligand mol (rare edge case).
    """
    # All SASA calls go through the unified core's RDKit adapter (adcsim.sasa):
    # OONS radii, probe radius and the polar/non-polar convention are defined in
    # exactly one place, identical to the antibody-side SASA, so this module no
    # longer keeps a second copy of the recipe.
    total_npolar = 0.0
    ok = 0
    for lig in adc["ligands"].values():
        if lig.get("c91") is None:
            continue
        try:
            mol, index = _ligand_mol(lig, adc["conect"])
            mol_h = Chem.AddHs(mol, addCoords=True)
            conf = mol_h.GetConformer()
            for serial, idx in index.items():
                xyz = _xyz(lig["lines"][serial])
                conf.SetAtomPosition(idx, (float(xyz[0]), float(xyz[1]), float(xyz[2])))
            _, _, npolar = sasa(mol_h)
            total_npolar += npolar
            ok += 1
        except Exception:
            continue
    return round(float(total_npolar), 1) if ok else None


def score_card(pdb_path: str) -> dict:
    lines = Path(pdb_path).read_text().splitlines()
    adc = load_adc_pdb(pdb_path)
    lig_all, lig_per = _lpp_heavy_coords(adc)

    # compactness (vs each LPP anchor = its Cys-SG)
    rgs, reaches, spans = [], [], []
    for rid in sorted(adc["ligands"]):
        lig = adc["ligands"][rid]
        xyz = ligand_xyz(lig)
        com = xyz.mean(axis=0)
        rgs.append(float(np.sqrt(((xyz - com) ** 2).sum(axis=1).mean())))
        spans.append(float(np.max(np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2))))
        sg = _xyz(lig["sg_line"])
        reaches.append(float(np.linalg.norm(xyz - sg, axis=1).max()))

    # surface contact: LPP heavy atoms 2.4-4.5 A from protein
    prot = np.asarray([_xyz(adc["atoms"][s]) for s in adc["protein_serials"]], dtype=float)
    tree = cKDTree(prot)
    d, _ = tree.query(lig_all, k=1)
    contact = int(np.sum((d >= 2.4) & (d <= 4.5)))

    # distance to CDR
    cdr = _cdr_ca_coords(lines)
    cdr_min = None
    if len(cdr) and len(lig_all):
        cdr_min = round(float(cKDTree(cdr).query(lig_all, k=1)[0].min()), 2)

    return {
        "file": Path(pdb_path).name,
        "n_lpp": len(adc["ligands"]),
        "rg_mean": round(float(np.mean(rgs)), 2),
        "span_mean": round(float(np.mean(spans)), 2),
        "reach_mean": round(float(np.mean(reaches)), 2),
        "surface_contact": contact,
        "cdr_min_dist_A": cdr_min,
        "mmff_stress_kcal": _mmff_stress_kcal(adc),
        # The name must not lie: _hydrophobic_exposure returns the SUM of
        # non-polar SASA over all LPP residues (Å²), NOT a per-atom value.
        # Calling it per_atom invited comparisons against single-molecule Rg.
        "hydrophobic_exposure_total_A2": _hydrophobic_exposure(adc),
    }


DEFAULT_REPS = [
    "ADC_DAR1_min.pdb",
    "ADC_DAR2_min.pdb",
    "ADC_DAR3_min.pdb",
    "ADC_DAR4_min.pdb",
    "ADC_DAR5_min.pdb",
    "ADC_DAR6_min.pdb",
    "ADC_DAR7_omit_B229_min.pdb",
    "ADC_DAR7_omit_B232_min.pdb",
]


def score_dir(in_dir: str, out_json: str | None = None) -> list[dict]:
    in_dir = Path(in_dir)
    cards = []
    for name in DEFAULT_REPS:
        p = in_dir / name
        if not p.exists():
            continue
        cards.append(score_card(str(p)))
    out_json = out_json or str(in_dir / "score_cards.json")
    Path(out_json).write_text(
        json.dumps(
            {
                "score_cards": cards,
                "note": (
                    "scorecard != binding free energy; computational DAR != "
                    "experimental DAR; cdr_min_dist uses Kabat CDR (Trastuzumab)."
                ),
            },
            indent=2,
        )
    )
    return cards


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(prog="adcsim-scorecard")
    p.add_argument("--dir", required=True, help="dir with *_min.pdb")
    args = p.parse_args()
    cards = score_dir(args.dir)
    print(json.dumps(cards, indent=2))