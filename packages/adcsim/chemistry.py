"""Optional chemistry sanity checks.

The middle notebook (ADC_docking.ipynb) built a post-addition template
where the C90=C91 double bond is reduced and C91 receives the Cys-S.
The complete 201-atom template kept S at origin; only topology matters.
This module reproduces that topology check for new LP SDF files.

It also hosts classify_conjugation(): the model's first decision when given a
small molecule + an antibody — read the LP warhead and decide WHICH chemistry is
happening, because that decides which sites to go looking for on the antibody.
"""

from pathlib import Path

from rdkit import Chem


def classify_conjugation(sdf_path: str) -> dict:
    """Read the LP warhead and decide which conjugation chemistry it implies.

    In bench terms: the small molecule carries a reactive group that decides WHAT
    on the antibody it will grab. A maleimide grabs a free thiol (a cysteine that
    has been unmasked by reducing a disulfide); an NHS / active ester or an
    isocyanate grabs an amine (a lysine side chain, or the N-terminus). Knowing
    this decides which sites to look for — and whether a reduction step applies.

    Returns {conjugation_type, warhead, evidence, confidence}.
      conjugation_type: "cysteine" | "lysine" | "unknown"
      confidence: always "C" here — the warhead->site-type rule is standard chemistry,
                 but this specific classifier is not yet benchmarked against labelled data.
    """
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        return dict(conjugation_type="unknown", warhead=None,
                    evidence="could not read SDF", confidence="C")

    maleimide = Chem.MolFromSmarts("O=C1C=CC(=O)N1")       # thiol-Michael -> cysteine
    isocyanate = Chem.MolFromSmarts("[NX2]=[CX2]=[OX1]")    # amine-reactive -> lysine
    active_ester = Chem.MolFromSmarts("[#6](=O)O[N+](=O)[O-]")  # NHS-type -> lysine

    if maleimide is not None and mol.HasSubstructMatch(maleimide):
        return dict(conjugation_type="cysteine", warhead="maleimide (thiol-Michael)",
                    evidence="maleimide ring detected -> reacts with a free thiol",
                    confidence="C")
    if isocyanate is not None and mol.HasSubstructMatch(isocyanate):
        return dict(conjugation_type="lysine", warhead="isocyanate",
                    evidence="isocyanate -> reacts with an amine (Lys / N-terminus)",
                    confidence="C")
    if active_ester is not None and mol.HasSubstructMatch(active_ester):
        return dict(conjugation_type="lysine", warhead="active ester (e.g. NHS)",
                    evidence="active ester -> reacts with an amine (Lys / N-terminus)",
                    confidence="C")
    return dict(conjugation_type="unknown", warhead=None,
                evidence="no recognised warhead (maleimide / active ester / isocyanate)",
                confidence="C")


def check_c91_adduct(sdf_path: str) -> dict:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        raise ValueError(f"cannot read {sdf_path}")
    bond = mol.GetBondBetweenAtoms(90, 91)
    if bond is None:
        raise ValueError("90–91 bond missing")
    out = {
        "atom90": mol.GetAtomWithIdx(90).GetSymbol(),
        "atom91": mol.GetAtomWithIdx(91).GetSymbol(),
        "bond_90_91": str(bond.GetBondType()),
        "n_atoms": mol.GetNumAtoms(),
    }
    out["ok"] = (
        out["atom90"] == "C"
        and out["atom91"] == "C"
        and bond.GetBondType() == Chem.BondType.DOUBLE
    )
    return out
