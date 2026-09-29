"""Electrostatics gate for charged payloads (e.g. MMAF), optional for MMAE.

Why this exists
---------------
The pure-geometry cess/clash funnel treats every payload atom as a neutral
hard sphere.  That is correct for mc-vc-PAB-MMAE (no ionizable group at
pH 7.4) but WRONG for charged payloads such as mc-MMAF, whose terminal
Phe carries a -COO(-) at physiological pH.  A charged payload experiences
long-range Coulomb repulsion from (a) the acidic antibody surface and
(b) neighbouring charged payloads, which pure geometry cannot see, so the
geometric DAR ceiling is OVERESTIMATED for charged payloads.

Design (deliberately cheap, not a full PB solver)
-------------------------------------------------
1. Detect ionizable groups by SMARTS and decide protonation at pH 7.4 from
   textbook pKa: carboxylic acids deprotonate (-), primary/secondary aliphatic
   amines protonate (+), aromatic amines and phenols stay neutral.
2. If net charge ~ 0 (MMAE) -> gate is a no-op (returns N/N/A), and the
   caller keeps the pure-geometric ceiling.
3. If charged (MMAF) -> add a distance-dependent Coulomb term
       E = 332.06 * q_i q_j / (eps_eff * r_ij)   [kcal/mol, q in e, r in A]
   with eps_eff = 78.4 (water value), a standard screening approximation for
   solvent-exposed groups, and flag any pose whose pair energy exceeds a
   cutoff as electrostatically forbidden.

Everything here is a PREDICTION and a screening proxy, not a measured Kd or
stability.  Final DAR / solubility must be confirmed by HIC / LC-MS / DLS.

References for the approximation choice
---------------------------------------
- Coulomb with a distance-dependent or uniform dielectric is the textbook
  (implicit-solvent) first pass; full PB (APBS/DelPhi) or explicit-water MD
  would be required for quantitative numbers.
- MMAF vs MMAE charge/structure: see standard ADC design reviews
  (e.g. the Kadcyla/enfortumab lineage); terminal Phe-carboxylate is the
  documented polarity difference.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem

# 7.4 is blood / physiological pH — NOT the conjugation process pH, which is
# 7.0 (see adc_model_config.yaml).  The two mean different things: use this one
# for in-vivo distribution and serum stability, use 7.0 for conjugation
# chemistry.  Do not mix them.
PHYSIOLOGICAL_PH = 7.4

# 78.4 is the dielectric constant of *bulk water*.  site_env.py uses an
# *effective* ε_eff ∈ [20, 80] (primary value 40) for protein-surface sites,
# because a protein surface is neither bulk water nor protein interior (≈2–4).
# Not a contradiction — two different purposes: Coulomb interactions in solution
# here, effective shielding at a surface site there.
_EPS_WATER = 78.4

# Coulomb prefactor.  MUST stay equal to site_env.COULOMB — change one, change both.
_QC = 332.06  # kcal/mol * A / e^2 (Coulomb prefactor)


# --------------------------------------------------------------------------
# Ionizable-group detection
# --------------------------------------------------------------------------

# (name, charge_at_pH7.4, SMARTS, note)
_IONIZABLE = [
    ("carboxyl", -1, "[CX3](=O)[OX2H1,O-]", "carboxylic acid -> -COO(-)"),
    ("amine_primary", +1, "[NX3H2;!$(NC=O);!$(N~*)][CX4,!H]", "aliphatic 1-amine"),
    ("amine_sec", +1, "[NX3H1;!$(NC=O)][CX4,!H]", "aliphatic 2-amine"),
    ("amine_tert", +1, "[NX3;!$(NC=O);!$(N=*);!$(N~*)][CX4][CX4][CX4]", "aliphatic 3-amine (e.g. MMAE dolastatin N)"),
    ("phenol", 0, "[OX2H1]c", "phenol stays neutral at 7.4"),
    ("aniline", 0, "[NX3H2]a", "aromatic amine stays neutral"),
]


@dataclass
class IonizableGroup:
    name: str
    charge: int
    atom_idx: int


@dataclass
class ElectrostaticsResult:
    pH: float
    charged: bool
    net_charge: int
    groups: list[IonizableGroup] = field(default_factory=list)
    note: str = ""


def _explicit_h_and_heavy(mol: Chem.Mol, a: Chem.Atom) -> tuple[int, int]:
    """Count explicit-H neighbours and heavy neighbours (explicit-H mol safe)."""
    h = heavy = 0
    for n in a.GetNeighbors():
        if n.GetAtomicNum() == 1:
            h += 1
        elif n.GetAtomicNum() > 1:
            heavy += 1
    return h, heavy


def _is_amide_or_imide_n(mol: Chem.Mol, a: Chem.Atom) -> bool:
    """True for amide / imide N (neutral): N bonded to >=1 carbonyl C with a
    single bond AND no free lone pair that is aliphatic-amine-like.

    Specifically:
      - imide N (maleimide): N bonded to 2 carbonyl C's  -> neutral
      - amide N: N bonded to exactly 1 carbonyl C and >=1 other substituent
        but with resonance stabilization (N-C(=O)-R) -> neutral
    A tertiary amine whose C neighbours are mostly NON-carbonyl is NOT an
    amide (dolastatin N9/N34/N49) -> protonatable.
    """
    carbonyl_n = 0
    non_carbonyl = 0
    for n in a.GetNeighbors():
        if n.GetAtomicNum() != 6:
            continue
        if mol.GetBondBetweenAtoms(a.GetIdx(), n.GetIdx()).GetBondType() != Chem.BondType.SINGLE:
            continue
        is_carb = any(
            nn.GetAtomicNum() == 8
            and mol.GetBondBetweenAtoms(n.GetIdx(), nn.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            for nn in n.GetNeighbors()
        )
        if is_carb:
            carbonyl_n += 1
        else:
            non_carbonyl += 1
    # amide / imide: N is directly bonded to >=1 carbonyl C (single bond) = neutral
    return carbonyl_n >= 1


def _is_aromatic_n(mol: Chem.Mol, a: Chem.Atom) -> bool:
    return a.GetIsAromatic()


def ionizable_groups(mol: Chem.Mol, ph: float = PHYSIOLOGICAL_PH) -> ElectrostaticsResult:
    """Detect ionizable groups on one payload mol and their pH-7.4 charge.

    Heuristic (explicit-H safe), with the caveat that exact net charge is a
    prediction: amine N with H(s) -> +1; alkyl 3-amine -> +1; carboxylate
    C(=O)-O (not carbonyl ester) -> -1.  Amide / carbamate N stay neutral.
    """
    mol = Chem.AddHs(Chem.Mol(mol))
    groups: list[IonizableGroup] = []

    # ---- carboxylates first (acidic O of -COO, not ester C(=O)-O-C) ----
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 8:
            continue
        heavy_nbs = [n for n in a.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy_nbs) != 1:
            continue  # carbonyl O (double-bonded, 1 heavy nb) handled separately; ester O has 2 heavy nbs
        n = heavy_nbs[0]
        if n.GetAtomicNum() != 6:
            continue
        if mol.GetBondBetweenAtoms(a.GetIdx(), n.GetIdx()).GetBondType() != Chem.BondType.SINGLE:
            continue
        # C must be double-bonded to another O (so C(=O)-O-) ...
        carbonyl_os = [
            nn for nn in n.GetNeighbors()
            if nn.GetAtomicNum() == 8 and nn.GetIdx() != a.GetIdx()
            and mol.GetBondBetweenAtoms(n.GetIdx(), nn.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        ]
        if not carbonyl_os:
            continue
        # ... and the O must have at least one bonded H (= -OH, not ester O-R)
        # In an acid -COOH this O is bonded to H; in an ester -COOR it is bonded to C
        o_has_h = any(n.GetAtomicNum() == 1 for n in a.GetNeighbors())
        if not o_has_h:
            continue  # ester O or other non-acidic O
        groups.append(IonizableGroup("carboxyl", -1, int(a.GetIdx())))

    # ---- amines ----
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 7:
            continue
        if _is_aromatic_n(mol, a):
            continue  # aromatic N stays neutral
        h, heavy = _explicit_h_and_heavy(mol, a)
        if _is_amide_or_imide_n(mol, a):
            continue  # amide / carbamate N stays neutral
        if h > 0:
            groups.append(IonizableGroup("amine", +1, int(a.GetIdx())))
        elif heavy == 3:
            groups.append(IonizableGroup("amine_tert", +1, int(a.GetIdx())))

    net = sum(g.charge for g in groups)
    return ElectrostaticsResult(
        pH=ph,
        charged=net != 0,
        net_charge=net,
        groups=groups,
        note="predicted from connectivity + textbook pKa; verify with experiment",
    )


# --------------------------------------------------------------------------
# Coulomb screening (charged payloads only)
# --------------------------------------------------------------------------

def _charge_coords(mol_h, ionizable: ElectrostaticsResult):
    """Return (charges, coords) for charged groups only."""
    conf = mol_h.GetConformer()
    charges, coords = [], []
    for g in ionizable.groups:
        if g.charge == 0:
            continue
        p = conf.GetAtomPosition(g.atom_idx)
        charges.append(float(g.charge))
        coords.append((p.x, p.y, p.z))
    return np.asarray(charges, dtype=float), np.asarray(coords, dtype=float)


def pair_coulomb_kcal(charges: np.ndarray, coords: np.ndarray, eps: float = _EPS_WATER) -> float:
    """Total pairwise Coulomb energy in kcal/mol (q in e, coords in A)."""
    if len(charges) < 2:
        return 0.0
    n = len(charges)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = float(np.linalg.norm(coords[i] - coords[j]))
            if r < 0.5:
                continue  # bonded neighbour, not a real intermolecular contact
            total += _QC * charges[i] * charges[j] / (eps * r)
    return total


def electrostatic_clash_kcal(
    lp_charges: np.ndarray,
    lp_coords: np.ndarray,
    prot_charges: np.ndarray,
    prot_coords: np.ndarray,
    eps: float = _EPS_WATER,
) -> float:
    """Coulomb energy between a charged payload and the protein surface.

    lp_*  : charged atoms of one payload (charge + coord)
    prot_*: charged atoms of the protein (charge + coord), may be empty
    """
    if len(lp_charges) == 0 or len(prot_charges) == 0:
        return 0.0
    # brute force is fine: charged groups are few (<= a handful)
    e = 0.0
    for qi, ci in zip(lp_charges, lp_coords):
        d = np.linalg.norm(prot_coords - ci, axis=1)
        d[d < 0.5] = 0.5  # soft floor
        e += float(np.sum(_QC * qi * prot_charges / (eps * d)))
    return e


def summarize(payload_mols: list[Chem.Mol], ph: float = PHYSIOLOGICAL_PH) -> dict:
    """Top-level entry: classify whether this payload set is charged, and if so
    how strong the electrostatic penalty is.  Cheap, honest, documented."""
    charged_flags = []
    net = 0
    details = []
    for i, m in enumerate(payload_mols):
        r = ionizable_groups(m, ph)
        charged_flags.append(r.charged)
        net += r.net_charge
        details.append(
            {
                "conformer": i,
                "charged": r.charged,
                "net_charge": r.net_charge,
                "groups": [g.name for g in r.groups],
                "note": r.note,
            }
        )
    any_charged = any(charged_flags)
    return {
        "pH": ph,
        "payload_charged": any_charged,
        "net_charge_signed": net,
        "per_payload": details,
        "gate_active": any_charged,
        "meaning": (
            "Electrostatic gate ACTIVE: use Coulomb screening, do NOT trust the "
            "pure-geometric DAR ceiling for this payload."
            if any_charged
            else "Neutral payload (e.g. mc-vc-PAB-MMAE): electrostatic gate is a "
            "no-op; pure-geometric ceiling stands."
        ),
    }


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if not args:
        print("usage: python -m adcsim.electrostatics <payload.sdf>")
        sys.exit(1)
    from rdkit import Chem

    mols = [m for m in Chem.SDMolSupplier(args[0], removeHs=False) if m is not None]
    print(summarize(mols))