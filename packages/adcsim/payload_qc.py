"""linker-payload QC (0-b): morphology and homogeneity stability of the LP (small
molecule) in conjugation solution.

Answers a concrete experimental failure: **the small molecule is too hydrophobic →
it self-collapses into a clump in the aqueous phase → its maleimide reactive end gets
buried by its own hydrophobic core → it cannot conjugate.** This is a pitfall the user
actually hit, not a hypothetical failure mode.

Input: the LP conformational ensemble SDF + the `solutions:` config section (which
decides whether the solvent is water or a DMSO-containing mixed phase).
Output: reactive-end availability f_active —— multiplies directly into the effective
payload concentration of covalent conjugation.

---- Why this step cannot be skipped ----
Covalent conjugation asks "does it react on contact?", assuming the payload is unfolded
with its reactive end facing out. If the payload curls up in water on its own, that
premise no longer holds and any downstream calculation is moot. So this is an upstream
health-check for covalent conjugation, at the LP input stage rather than in the model stage.

---- How it is computed (every step is sourced, no hand-waving) ----
1. Conformational ensemble: by default data/payload_ensemble_79.sdf (79 ETKDG/MMFF conformers).
   Generates on the fly with RDKit ETKDG when the ensemble is missing.
2. Per conformer:
   a) MMFF intrinsic energy E_mmff (computed by RDKit; same force field, so conformers are comparable)
   b) Per-atom SASA (Shrake-Rupley, 1.4 Å probe, pure numpy implementation, verifiable)
   c) Solvation free energy: atom solvation parameter (ASP) weighted sum
      ΔG_solv = Σ σ_atom · SASA_atom
      σ values per `ASP_EISENBERG`, source Eisenberg & McLachlan, Nature 1986;319:199-203,
      DOI 10.1038/319199a0 —— positive means exposing that atom to water costs free energy (hydrophobic),
      negative means favorable (polar).
   d) Total energy = E_mmff + ΔG_solv → Boltzmann weight (temperature from solutions.reaction_medium.temperature_C)
3. **Population-weighted** observables over the ensemble:
   · f_active     = fraction of the population whose reactive end (the more exposed of the
                    maleimide's two alkene carbons) SASA reaches the threshold
   · hydrophobic self-burial = 1 − that conformer's exposed nonpolar area / the most-extended
                    conformer's nonpolar area in the ensemble
   · Rg collapse  = how much the tightest conformer collapses relative to the loosest

---- Honest boundaries ----
· This is an **implicit solvent (SASA-type)** model, not explicit-solvent MD. It gives the
  relative ranking of "which morphology is more favored", not true solvation free-energy values.
  Overall Tier C.
· The conformational ensemble itself is generated with MMFF (gas phase); using it for Boltzmann
  re-weighting is a second-order reuse: which conformers the ensemble covers caps the answer.
  **This is the largest source of uncertainty in this module.**
  Removing it requires re-running the ensemble with explicit water / mixed-solvent MD, at much higher cost.
· Co-solvent is now **used** in the solubility assessment: the hydrophobic linker-payload is
  dissolved in an organic co-solvent (DMSO / DMA / DMF) stock and added to an aqueous reaction
  medium, so the operative criterion is the mixed-solvent solubility at the configured co-solvent
  fraction (Yalkowsky–Rubino log-linear cosolvency model, see the solubility section). The pure-water
  GSE value is retained only as the f = 0 lower-bound reference. The co-solvent solubilizing power σ
  is estimated from logP via a class-average correlation and is flagged TO BE CALIBRATED against a
  measured stock-dilution / DLS solubility — we do not assert precipitation without that measurement.
· Where this system's (vc-MMAE) f_active lands and how much it sways the final conclusion is
  printed directly when you run `payload_qc.py`.

Usage: python payload_qc.py
"""

from __future__ import annotations

import os
import sys
import math
import numpy as np

# Shared physical constants (same source as site_env.R_KCAL / solvation, to avoid each module copying its own)
from .solvation import RGAS_KCAL_MOL_K as R_KBT, UM2_TO_CM2 as UM2_PER_CM2

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Crippen, rdMolDescriptors, Descriptors, rdDistGeom
    _HAS_RDKIT = True
except ImportError:  # pragma: no cover
    _HAS_RDKIT = False

# Unified SASA core. payload_qc.sasa now delegates to it, keeping the historical
# 240-point density so conformer-search numbers are bit-for-bit identical to the
# old self-contained numpy implementation (verified: max per-atom diff == 0.0).
from .sasa import shrake_rupley
from .clash import hydrogen_false_collision_count
from .io import load_antibody, heavy_atoms

HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(os.path.dirname(HERE))
_DEFAULT_ENSEMBLE = os.path.join(
    _PROJ, "examples", "trop_adc", "data", "payload_ensemble_79.sdf")
_DEFAULT_SINGLE = os.path.join(
    _PROJ, "examples", "trop_adc", "data", "MC-Val-Cit-PAB-MMAE_3D.sdf")
_CFG_PATH = os.path.join(
    _PROJ, "examples", "trop_adc", "kaggle_md_8site", "adc_model_config.yaml")

# Fallback values used only when the config file is missing; the normal path always reads yaml.
_FALLBACK = dict(
    temperature_C=4.0,          # = solutions.reaction_medium.temperature_C
    co_solvent_vv=0.10,         # = solutions.reaction_medium.co_solvent_vv
    cosolvent="DMSO",           # = solutions.payload_solvent.solvent
    sigma_override=None,         # = payload_qc.cosolvent_sigma (measured; else class-average)
    f_active_warn=0.70,
    f_active_crit=0.40,
    ensemble=None,              # path relative to project root
    # Critical aggregation concentration (CAC) model, see clustering_report().
    # TIER C: fitted on one payload pair in one medium. Ranking only.
    cac_logP_c0=-1.60,          # log10 CAC (M) of a logP = 0 solute in water
    cac_logP_c1=0.60,           # decades of CAC lost per unit logP
    cac_cosolvent_sigma=1.60,   # decades of CAC recovered per unit v/v co-solvent
    known_payloads=None,        # measured CAC anchors, keyed by payload name
)


def load_cfg(path=_CFG_PATH):
    """Read yaml config; fall back to built-in defaults and warn on stderr if missing or unreadable."""
    cfg = dict(_FALLBACK)
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        print(f"[warn] {path} not found / unparsable, using built-in fallback values", file=sys.stderr)
        return cfg
    med = (raw.get("solutions") or {}).get("reaction_medium") or {}
    psv = (raw.get("solutions") or {}).get("payload_solvent") or {}
    qc = (raw.get("payload_qc") or {})
    cfg["temperature_C"] = float(med.get("temperature_C", cfg["temperature_C"]))
    cfg["co_solvent_vv"] = float(med.get("co_solvent_vv", cfg["co_solvent_vv"]))
    cfg["cosolvent"] = str(psv.get("solvent", "DMSO"))
    cfg["f_active_warn"] = float(qc.get("f_active_warn", cfg["f_active_warn"]))
    cfg["f_active_crit"] = float(qc.get("f_active_crit", cfg["f_active_crit"]))
    cfg["ensemble"] = qc.get("ensemble", cfg["ensemble"])
    # Co-solvent solubilizing power σ: a measured override takes priority; otherwise the
    # class-average estimate (C3) is used and flagged TO BE CALIBRATED.
    cfg["sigma_override"] = qc.get("cosolvent_sigma")
    cfg["feed_ratio"] = float((raw.get("process") or {}).get(
        "payload_feed_ratio", 6.0))
    cfg["mab_uM"] = float((raw.get("process") or {}).get("mab_uM", 67.0))
    cfg["cac_logP_c0"] = float(qc.get("cac_logP_c0", cfg["cac_logP_c0"]))
    cfg["cac_logP_c1"] = float(qc.get("cac_logP_c1", cfg["cac_logP_c1"]))
    cfg["cac_cosolvent_sigma"] = float(qc.get("cosolvent_sigma",
                                              cfg["cac_cosolvent_sigma"]))
    cfg["known_payloads"] = qc.get("known_payloads") or {}
    return cfg

# ---- van der Waals radii (Å) and probe for Shrake-Rupley --------------------------------
VDW = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "F": 1.47}
DEFAULT_R = 1.80
PROBE = 1.4

# ---- Atom solvation parameters σ (kcal mol⁻¹ Å⁻²) -------------------------------------
# Eisenberg & McLachlan, Nature 1986;319:199-203. Original table unit cal/mol/Å², converted to kcal here.
# Meaning: free-energy change per 1 Å² of this atom exposed to water. Positive = costly (hydrophobic), negative = favorable (polar).
ASP_EISENBERG = {"C": 0.016, "S": 0.021, "N": -0.006, "O": -0.006}
ASP_DEFAULT = -0.006
ASP_SOURCE = "Eisenberg & McLachlan, Nature 1986;319:199-203, DOI 10.1038/319199a0"

# Maleimide ring: O=C1C=CC(=O)N1 —— the 3rd and 4th atoms (index 2,3 in 1-based) are the alkene-carbon pair
MALEIMIDE_SMARTS = "O=C1C=CC(=O)N1"
_ALKENE_POS = (2, 3)

# Threshold for whether the reactive end is "open enough"; same basis as sasa_lo in site accessibility
REACTIVE_SASA_THRESHOLD = 5.0  # Å²

R_KBT = 1.98720425e-3  # kcal mol⁻¹ K⁻¹


# ============================================================================
# basics
# ============================================================================
def golden_sphere(n: int = 240) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(theta) * np.sin(phi),
                 np.sin(theta) * np.sin(phi),
                 np.cos(phi)]


_SPHERE = golden_sphere(240)


def atom_arrays(mol):
    coords, symbols = [], []
    conf = mol.GetConformer()
    for a in mol.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        coords.append((p.x, p.y, p.z))
        symbols.append(a.GetSymbol())
    coords = np.asarray(coords, dtype=float)
    radii = np.array([VDW.get(s, DEFAULT_R) for s in symbols], dtype=float)
    return coords, np.array(symbols), radii


def sasa(coords: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Shrake-Rupley: occlusion among the same molecule's own atoms. Returns per-atom SASA (Å²).

    Delegates to the unified core :func:`adcsim.sasa.shrake_rupley` at the
    historical 240-point density (``golden_sphere(240)`` + OONS radii), so this
    function is now a thin adapter and the conformer-search SASA is bit-identical
    to the previous self-contained implementation.
    """
    return shrake_rupley(np.asarray(coords, dtype=float), np.asarray(radii, dtype=float),
                         probe=PROBE, n_points=240)


# ============================================================================
# load conformational ensemble
# ============================================================================
def load_ensemble(sdf_path: str = None, n_conf_fallback: int = 79, seed: int = 42,
                  minimize: bool = True):
    if not _HAS_RDKIT:
        raise RuntimeError("rdkit is required")
    path = sdf_path or _DEFAULT_ENSEMBLE
    if os.path.exists(path):
        sup = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
        mols = [m for m in sup if m is not None]
        if mols:
            # Conformers in the ensemble are not individually minimized, so their energies are not
            # comparable → each must be MMFF-minimized.
            # Boltzmann-weighting without minimization conflates the random generation strain with
            # energy, yielding a false conclusion.
            if minimize:
                for m in mols:
                    try:
                        AllChem.MMFFOptimizeMolecule(m, maxIters=800)
                    except Exception:
                        pass
            return mols, path
    # Ensemble absent → generate single-conformer ensemble on the fly
    base_path = _DEFAULT_SINGLE if os.path.exists(_DEFAULT_SINGLE) else None
    if base_path is None:
        raise FileNotFoundError(f"neither ensemble {path} nor single conformer {_DEFAULT_SINGLE} found")
    m0 = Chem.SDMolSupplier(base_path, removeHs=False, sanitize=False)[0]
    Chem.SanitizeMol(m0)
    m0 = Chem.AddHs(m0, addCoords=True)
    # rdkit removed AllChem.ETKDGv3Params(); rdDistGeom.ETKDGv3() is the current factory.
    # Keep the old call as a fallback so the module still runs on older rdkit builds.
    try:
        ps = rdDistGeom.ETKDGv3()
    except AttributeError:
        ps = AllChem.ETKDGv3Params()
    ps.randomSeed = seed
    cids = AllChem.EmbedMultipleConfs(m0, numConfs=n_conf_fallback, params=ps)
    if AllChem.MMFFHasAllMoleculeParams(m0):
        AllChem.MMFFOptimizeMoleculeConfs(m0, maxIters=600)
    mols = []
    for cid in cids:
        frag = Chem.Mol(m0)
        frag.RemoveAllConformers()
        frag.AddConformer(Chem.Conformer(m0.GetConformer(cid)), assignId=True)
        mols.append(frag)
    return mols, f"(ETKDG generated {len(mols)} on the fly, source: {os.path.basename(base_path)})"


def find_reactive_carbons(mol):
    """Return the atom indices of the maleimide alkene-carbon pair; return None if not found."""
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    patt = Chem.MolFromSmarts(MALEIMIDE_SMARTS)
    hits = mol.GetSubstructMatches(patt) if patt is not None else ()
    if not hits:
        return None
    ring = hits[0]
    return (ring[_ALKENE_POS[0]], ring[_ALKENE_POS[1]])


def molprops(mol):
    m = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass
    return dict(
        mol_logp=round(Crippen.MolLogP(m), 2),
        tpsa=round(rdMolDescriptors.CalcTPSA(m), 1),
        mw=round(Descriptors.MolWt(m), 1),
        n_atoms=mol.GetNumAtoms(),
    )


# ============================================================================
# main analysis
# ============================================================================
def analyse(mols, temperature_C: float = 4.0, reactive_sasa_threshold: float = None,
            asp_scale: float = 1.0):
    """asp_scale: scales the C/S hydrophobic solvation parameters overall (>1 = simulate a more
    hydrophobic payload).

    It is a **mechanistic self-check knob, not a physical parameter**: it answers "how hydrophobic
    must it be to collapse", so we can judge whether this module has any resolving power for the
    real failure case. Default 1.0 = use literature values as-is.
    """
    thr = reactive_sasa_threshold or REACTIVE_SASA_THRESHOLD
    T = temperature_C + 273.15
    kBT = R_KBT * T

    reactive = find_reactive_carbons(mols[0])
    props = molprops(mols[0])

    rows, bycmol = [], []
    for i, m in enumerate(mols):
        coords, sym, radii = atom_arrays(m)
        s = sasa(coords, radii)

        # Nonpolar (hydrophobic) = C and S not directly bonded to N/O; simplified criterion: element C/S counts as nonpolar
        nonpolar = np.isin(sym, ["C", "S"])

        # MMFF intrinsic energy
        e_mmff = None
        try:
            mol_h = Chem.AddHs(Chem.Mol(m), addCoords=True)
            ff = AllChem.MMFFGetMoleculeForceField(
                mol_h, AllChem.MMFFGetMoleculeProperties(mol_h))
            if ff is not None:
                e_mmff = float(ff.CalcEnergy())
        except Exception:
            e_mmff = None
        bycmol.append(e_mmff)

        sigma = np.array([(ASP_EISENBERG.get(s_, ASP_DEFAULT) * asp_scale
                           if s_ in ("C", "S") else ASP_EISENBERG.get(s_, ASP_DEFAULT))
                          for s_ in sym])
        e_solv = float((sigma * s).sum())
        e_tot = (e_mmff if e_mmff is not None else 0.0) + e_solv

        c = coords.mean(0)
        rg = float(np.sqrt(((coords - c) ** 2).sum(1).mean()))

        r_sasa = float(max(s[list(reactive)])) if reactive is not None else float("nan")
        rows.append(dict(i=i, rg=rg, sasa_tot=float(s.sum()),
                         sasa_nonpolar=float(s[nonpolar].sum()),
                         reactive_sasa=r_sasa,
                         e_mmff=e_mmff, e_solv=e_solv, e_tot=e_tot))

    # Boltzmann weights (relative total energy)
    et = np.array([r["e_tot"] for r in rows])
    w = np.exp(-(et - et.min()) / kBT)
    w /= w.sum()

    a_np = np.array([r["sasa_nonpolar"] for r in rows])
    rgs = np.array([r["rg"] for r in rows])
    a_ref = a_np.max()                       # most-extended conformer in the ensemble
    rg_ref = rgs.max()

    for r, wi in zip(rows, w):
        r["weight"] = float(wi)
        r["hydrophobic_self_burial"] = float(1 - r["sasa_nonpolar"] / a_ref) if a_ref > 0 else None
        r["collapsed"] = bool(r["reactive_sasa"] < thr)

    # Population-weighted observables
    f_active = float(sum(r["weight"] for r in rows if not r["collapsed"]))
    # a_ref == 0 → the line above writes None; multiplying directly would raise TypeError, so skip explicitly and leave 0.0
    burial_w = float(sum(r["weight"] * r["hydrophobic_self_burial"] for r in rows
                         if r["hydrophobic_self_burial"] is not None))
    rg_w = float(sum(r["weight"] * r["rg"] for r in rows))
    reactive_sasa_w = float(sum(r["weight"] * r["reactive_sasa"] for r in rows))

    return dict(
        ensemble_size=len(rows),
        temperature_C=temperature_C,
        reactive_atoms=reactive,
        reactive_sasa_threshold=thr,
        props=props,
        rows=rows,
        summary=dict(
            f_active=f_active,
            mean_hydrophobic_self_burial=burial_w,
            mean_rg_A=rg_w,
            rg_spread_A=(float(rgs.min()), float(rg_ref)),
            mean_reactive_sasa_A2=reactive_sasa_w,
            mean_nonpolar_exposed_A2=float(sum(r["weight"] * r["sasa_nonpolar"] for r in rows)),
            top_conformer_population=float(w.max()),
            effective_population_entropy=float(-(w * np.log(w + 1e-300)).sum()),
        ),
    )


# ============================================================================
# Co-solvent (mixed-solvent) solubility — cosolvency model
# ============================================================================
# The pure-water General Solubility Equation (GSE) below gives the solubility at
# ZERO co-solvent (f = 0). Real ADC conjugation is never in pure water: the
# hydrophobic linker-payload is dissolved in an organic co-solvent (DMSO / DMA /
# DMF) stock and added to an aqueous reaction medium at a few-to-tens percent
# v/v. In that mixed solvent the solubility is orders of magnitude higher than in
# pure water, so judging "does it dissolve?" against the pure-water value is an
# overstatement and produced a false CRIT in the previous version of this module.
#
# We therefore switch the criterion to the co-solvent-aware solubility at the
# CONFIGURED co-solvent fraction. The pure-water value is retained only as the
# f = 0 lower-bound reference.
#
# Model: Yalkowsky–Rubino log-linear cosolvency equation [C1]:
#     log10 S_mix(M) = log10 S_w(M) + σ · f
# where f is the co-solvent volume fraction and σ is the co-solvent solubilizing
# power (slope of log(S_mix/S_w) vs f) for this solute/co-solvent pair.
#
# σ is estimated from the solute partition coefficient via Li & Yalkowsky [C3]:
#     σ = A_c + B_c · logP
# with per-co-solvent (A_c, B_c) tabulated in COSOLVENT_SIGMA_COEFF. These are
# CLASS AVERAGES from a 607-dataset regression over 15 co-solvents; for an
# individual molecule (here vc-MMAE) the true σ can differ substantially (the
# regressions have R² ≈ 0.7–0.95 and wide scatter). Every σ used here is therefore
# flagged TO BE CALIBRATED against a measured stock-dilution / DLS solubility.
#
# Sources (unified record also in docs/cosolvent_solubility_references.md):
#  [C1] Yalkowsky SH, Rubino JT. Solubilization by cosolvents I: organic solutes
#       in propylene glycol–water mixtures. J Pharm Sci 1985;74(4):416-421.
#       DOI 10.1002/jps.2600690814   (log-linear cosolvency model, framework)
#  [C2] Rubino JT, Blanchard J, Yalkowsky SH. Solubilization by Cosolvents IV:
#       Benzocaine, Diazepam and Phenytoin in Aprotic Cosolvent–Water Mixtures.
#       PDA J Pharm Sci Technol 1987;41(5):172-176.  (aprotic co-solvents DMSO/DMA/DMF)
#  [C3] Li A, Yalkowsky SH. Predicting Cosolvency. 1. Solubility Ratio and Solute
#       log Kow. Ind Eng Chem Res 1998;37(11):4470-4475.
#       DOI 10.1021/ie980232v   (σ = A + B·logP for 15 co-solvents, 607 datasets)
#  [C4] Jouyban-Gharamaleki A, Acree WE. Method for predicting the solubility of
#       drugs in binary solvent mixtures at various temperatures. Int J Pharm
#       2001;218(1-2):29-37. DOI 10.1016/S0378-5173(00)00658-6
#       (Jouyban–Acree model — more accurate, handles multi-component & T, but
#        requires measured training points; documented as the alternative, not the
#        default, because no measured points exist yet.)
GSE_SOURCE = ("Yalkowsky SH, Valvani SC. J Pharm Sci 1980;69(8):912-922, "
              "DOI 10.1002/jps.2600690814  (pure-water solubility, f=0 reference)")
COSOLVENCY_MODEL_SOURCE = ("Yalkowsky SH, Rubino JT. J Pharm Sci 1985;74(4):416-421, "
                           "DOI 10.1002/jps.2600690814  (log-linear cosolvency model)")
COSOLVENT_SIGMA_SOURCE = ("Li A, Yalkowsky SH. Ind Eng Chem Res 1998;37(11):4470-4475, "
                          "DOI 10.1021/ie980232v  (σ = A + B·logP); aprotic co-solvents "
                          "verified in Rubino, Blanchard & Yalkowsky, PDA J Pharm Sci Technol "
                          "1987;41(5):172-176. CLASS AVERAGES — TO BE CALIBRATED for vc-MMAE.")


# ---------------------------------------------------------------------------
# Measured aqueous-solubility anchor — takes precedence over the GSE estimate.
#
# Why this exists: the GSE equation badly underestimates the aqueous solubility
# of complex peptidic payloads. For MMAE it predicts 26.3 µM, whereas the
# vendor-measured value in PBS (pH 7.2) is ~0.5 mg/mL ≈ 696 µM — about 26×
# higher. Using the GSE value as the f = 0 base made the model claim the working
# concentration exceeded solubility by 15×, which contradicts wet-lab reality
# (no precipitation observed at 8–10 equiv with 10% v/v DMA).
#
# Measured value, BARE payload (MMAE, MW 718.0 g/mol):
#   0.5 mg/mL in PBS pH 7.2 -> (0.5/718.0) mol/L = 6.96e-4 M = 696 µM
#   Cayman Chemical, Monomethyl Auristatin E, Item No. 16267 (product insert).
#
# CAVEAT — must be stated wherever this number is used: this is the BARE
# payload. The species actually conjugated is vc-MMAE (payload + cleavable
# linker), which is larger and more hydrophobic, so its true solubility is
# LOWER. 696 µM is therefore an OPTIMISTIC UPPER BOUND for the f = 0 base and
# the real margin is tighter. Replace with a measured vc-MMAE value once
# available; set to None to fall back to the GSE estimate.
# ---------------------------------------------------------------------------
MEASURED_SW_UM = 696.0
MEASURED_SW_SOURCE = (
    "Cayman Chemical, Monomethyl Auristatin E, Item No. 16267 (product insert): "
    "solubility in PBS (pH 7.2) ≈ 0.5 mg/mL; MW 718.0 g/mol -> ≈696 µM. "
    "BARE payload — vc-MMAE is larger and less soluble; treat as an upper bound."
)


def gse_solubility_M(logp: float, mp_C: float = 25.0) -> float:
    """Pure-water molar solubility (mol/L) — the f = 0 reference / lower bound.
    mp_C=25 means the liquid/amorphous assumption (most optimistic upper bound)."""
    return 10 ** (0.5 - logp - 0.01 * (mp_C - 25.0))


# Per-co-solvent σ-slope coefficients (σ = A + B·logP) — class averages [C3].
# Values are the regression coefficients reported for the aprotic co-solvents in
# C2/C3 (DMSO/DMA/DMF) and acetonitrile. R² ≈ 0.9 for the 7-point regressions.
# Every value is a class average and is TO BE CALIBRATED for vc-MMAE.
COSOLVENT_SIGMA_COEFF = {
    "DMSO": (0.89, 0.87),   # N=7, R²=0.95
    "DMA":  (0.89, 0.86),   # N=7, R²=0.95
    "DMF":  (0.87, 0.87),   # N=7, R²=0.94
    "ACN":  (0.35, 1.03),   # acetonitrile; N=8, R²=0.90 (less used in ADC) — CALIBRATE
}
# Co-solvents not listed here cannot be estimated and must supply a measured σ.
DEFAULT_F_SCAN = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20)


def cosolvent_sigma(cosolvent: str, logp: float, sigma_override=None):
    """Return (σ, calibrated_flag) for the log-linear cosolvency slope.

    σ = A + B·logP from C3 (class average) unless an explicit measured
    `sigma_override` is supplied. Unknown co-solvent → (nan, False)."""
    if sigma_override is not None:
        return float(sigma_override), True
    key = (cosolvent or "").upper()
    coeff = COSOLVENT_SIGMA_COEFF.get(key)
    if coeff is None:
        return float("nan"), False
    A, B = coeff
    return A + B * logp, False


def mix_solubility_M(logp: float, sw_M: float, cosolvent: str, f_vv: float,
                     sigma_override=None):
    """Yalkowsky–Rubino log-linear cosolvency model [C1].

    log10 S_mix(M) = log10 S_w(M) + σ·f .
    Returns (S_mix_M, sigma, sigma_calibrated)."""
    sigma, calibrated = cosolvent_sigma(cosolvent, logp, sigma_override)
    if not math.isfinite(sigma) or sw_M <= 0 or f_vv < 0:
        return float("nan"), sigma, calibrated
    s_mix = 10 ** (math.log10(sw_M) + sigma * f_vv)
    return s_mix, sigma, calibrated


def cosolvent_solubility_report(logp: float, sw_M: float, cosolvent: str,
                                working_uM: float, f_list=DEFAULT_F_SCAN,
                                sigma_override=None):
    """Sensitivity scan over co-solvent volume fraction f.

    Returns a dict with the pure-water reference and, for each f, the co-solvent
    (mixed-solvent) solubility estimate, the fold of working concentration over it,
    and the σ used. σ is a class average unless overridden → flagged calibrated."""
    sigma, calibrated = cosolvent_sigma(cosolvent, logp, sigma_override)
    rows = []
    for f in f_list:
        s_mix, _, _ = mix_solubility_M(logp, sw_M, cosolvent, f, sigma_override)
        s_uM = s_mix * 1e6
        rows.append(dict(f_vv=f, smix_uM=s_uM,
                         fold=working_uM / s_uM if math.isfinite(s_uM) and s_uM > 0 else float("inf")))
    return dict(sw_uM=sw_M * 1e6, cosolvent=cosolvent, sigma=sigma,
                sigma_calibrated=calibrated, working_uM=working_uM, rows=rows)


def attach_solubility(res, working_uM: float, cosolvent: str = "DMSO",
                      f_configured: float = 0.10, mp_list=(25.0, 100.0, 200.0),
                      sigma_override=None):
    """Attach the co-solvent-aware solubility assessment.

    The pure-water GSE value is kept as the f = 0 reference only. The operative
    criterion is the mixed-solvent solubility at the configured co-solvent
    fraction."""
    logp = res["props"]["mol_logp"]
    sw_rows = []
    for mp in mp_list:
        s_M = gse_solubility_M(logp, mp)
        sw_rows.append(dict(mp_C=mp, sw_uM=s_M * 1e6,
                            fold=working_uM / (s_M * 1e6) if s_M > 0 else float("inf")))
    # f = 0 base: prefer the MEASURED value; fall back to the GSE estimate only
    # when no measurement exists. The GSE number is always retained as a
    # cross-check, but it is never the operative criterion.
    sw_gse_uM = sw_rows[0]["sw_uM"]   # MP=25 (liquid/amorphous, most optimistic)
    sw_opt_uM = float(MEASURED_SW_UM) if MEASURED_SW_UM is not None else sw_gse_uM
    rep = cosolvent_solubility_report(logp, sw_opt_uM / 1e6, cosolvent, working_uM,
                                     DEFAULT_F_SCAN, sigma_override)
    smix_cfg, sigma_cfg, cal_cfg = mix_solubility_M(logp, sw_opt_uM / 1e6, cosolvent,
                                                    f_configured, sigma_override)
    # σ required to dissolve the working concentration at the configured f
    sigma_req = ((math.log10(working_uM / 1e6) - math.log10(sw_opt_uM / 1e6)) / f_configured
                 if f_configured > 0 else float("inf"))
    s = res["summary"]
    s["sw_uM_ref"] = sw_opt_uM
    s["sw_is_measured"] = MEASURED_SW_UM is not None
    s["sw_source_used"] = (MEASURED_SW_SOURCE if MEASURED_SW_UM is not None
                           else GSE_SOURCE)
    s["sw_gse_estimate_uM"] = sw_gse_uM   # kept as cross-check, not the criterion
    s["sw_scan"] = sw_rows
    s["gse_source"] = GSE_SOURCE
    s["cosolvent"] = cosolvent
    s["co_solvent_vv_configured"] = f_configured
    s["sigma_est"] = rep["sigma"]
    s["sigma_calibrated"] = rep["sigma_calibrated"]
    s["sigma_required"] = sigma_req
    s["smix_uM_configured"] = smix_cfg * 1e6
    s["working_conc_uM"] = float(working_uM)
    s["mix_fold_configured"] = (working_uM / (smix_cfg * 1e6)
                                if math.isfinite(smix_cfg) and smix_cfg > 0 else float("inf"))
    s["cosolvent_scan"] = rep["rows"]
    s["cosolvency_model_source"] = COSOLVENCY_MODEL_SOURCE
    s["cosolvent_sigma_source"] = COSOLVENT_SIGMA_SOURCE
    return res


def to_alarms(res, limits=None):
    """Emit checks isomorphic to those in packages/adcsim/alarms.py."""
    limits = limits or {}
    s = res["summary"]
    f = s["f_active"]
    warn = limits.get("f_active_warn", 0.70)
    crit = limits.get("f_active_crit", 0.40)
    if f < crit:
        lv, ref = "crit", f"≥ {crit}"
    elif f < warn:
        lv, ref = "warn", f"≥ {warn}"
    else:
        lv, ref = "ok", f"≥ {warn}"
    items = [dict(
        step="linker-payload QC",
        item="Reactive-end availability",
        value=round(f, 3),
        unit="(population fraction)",
        reference=ref,
        level=lv,
        confidence="C",
        meaning="fraction of conformers where the payload buries its maleimide end inside its own "
                "hydrophobic core in aqueous phase; the lower it is, the more molecules must first "
                "unfold before they can react",
        action="raise the organic co-solvent fraction (cap 10–20% v/v) / switch to a more hydrophilic "
               "linker / use DLS or solubility measurements to confirm whether clustering truly occurs",
    )]
    # Driving force for intermolecular aggregation: the "greasy area" a single molecule presents
    # outward when spread in water.
    # Note the distinction: f_active measures **intramolecular** self-burial (curling up and burying
    # the reactive end); this item measures the available contact surface for **intermolecular**
    # aggregation (several molecules sticking together).
    # The "small molecule clumps up" failure the user hit could be either mechanism; this item only
    # covers the latter, and its threshold must be calibrated.
    a_np = s["mean_nonpolar_exposed_A2"]
    warn_np = limits.get("nonpolar_exposed_warn_A2", 200.0)
    items.append(dict(
        step="linker-payload QC",
        item="Exposed nonpolar area (intermolecular aggregation driving force)",
        value=round(a_np, 1),
        unit="Å²",
        reference=f"≤ {warn_np:.0f} (placeholder threshold, must be calibrated with your own DLS/solubility data)",
        level="warn" if a_np > warn_np else "ok",
        confidence="C",
        meaning="the hydrophobic area a single molecule presents outward when spread in solution; the "
                "larger it is, the more readily several molecules stick together. **This is the "
                "intermolecular-aggregation driving force, not intramolecular self-burial** — the two "
                "mechanisms differ, do not conflate them.",
        action="raise the organic co-solvent fraction (cap 10–20% v/v), lower the payload concentration, "
               "add a surfactant (e.g. PS20), or use DLS to look at the particle-size distribution directly",
    ))
    # ---- Co-solvent-aware solubility criterion (replaces the pure-water criterion) ----
    # Mechanism tested: at the configured co-solvent fraction, is the nominal
    # linker-payload concentration below the mixed-solvent solubility? Excess
    # would remain as aggregates/precipitate and not participate in conjugation.
    # This is now judged against the CO-SOLVENT solubility, never the f=0 pure-water value.
    smix_cfg = s.get("smix_uM_configured")
    if smix_cfg is not None and math.isfinite(smix_cfg):
        work = s["working_conc_uM"]
        fv = s.get("co_solvent_vv_configured", 0.0)
        cosolv = s.get("cosolvent", "DMSO")
        sigma = s.get("sigma_est", float("nan"))
        cal = s.get("sigma_calibrated", False)
        fold = s.get("mix_fold_configured", work / smix_cfg)
        sigma_req = s.get("sigma_required", float("nan"))
        temp_C = s.get("temperature_C", 4.0)
        if cal:
            # measured σ → real criterion
            lv = "crit" if fold > 10 else ("warn" if fold > 2 else "ok")
            meaning = (f"At the configured {fv*100:.0f}% {cosolv} co-solvent fraction the measured "
                       f"mixed-solvent solubility is {smix_cfg:.0f} µM (log-linear cosolvency model [C1], "
                       f"measured σ={sigma:.2f}). Nominal linker-payload concentration {work:.0f} µM is "
                       f"{fold:.1f}× that solubility; the excess would remain as aggregates/precipitate and "
                       f"not participate in covalent conjugation.")
            action = ("lower the payload feed concentration / raise the co-solvent fraction (cap 10–20% v/v) "
                      "/ switch to a more hydrophilic linker / confirm by DLS or post-centrifugation "
                      "supernatant concentration.")
        else:
            # Class-average σ, uncalibrated → never assert precipitation; flag for measurement.
            lv = "info"
            meaning = (f"Criterion now uses the co-solvent-aware solubility, not pure water. At the configured "
                       f"{fv*100:.0f}% {cosolv} fraction the ESTIMATED mixed-solvent solubility is {smix_cfg:.0f} µM "
                       f"(log-linear cosolvency model [C1], σ={sigma:.2f} from a class-average logP correlation "
                       f"[C3] — TO BE CALIBRATED for vc-MMAE; the true σ can differ by ±50%). Nominal "
                       f"concentration {work:.0f} µM. The previously reported '>15× over pure-water solubility "
                       f"→ large amount undissolved, CRIT' conclusion is RETIRED: it compared the working "
                       f"concentration against the f=0 pure-water lower bound and overstated the deficit "
                       f"(the co-solvent-aware estimate at {fv*100:.0f}% is ≈{fold:.1f}× and is itself uncalibrated). "
                       f"Confirm with a measured stock-dilution / DLS solubility before any process call.")
            if fold <= 1.0:
                # Already inside the solubility envelope — no extra solubilisation
                # needed, so a "required σ" is meaningless (it would be <= 0).
                action = (f"Nominal concentration is already INSIDE the estimated solubility envelope "
                          f"({work:.0f} µM vs {smix_cfg:.0f} µM at {fv*100:.0f}% {cosolv}); no extra "
                          f"solubilisation is indicated. Note the margin rests on the BARE-payload "
                          f"measured S_w (696 µM, {cosolv}/PBS) — vc-MMAE is larger and less soluble, so "
                          f"confirm with one turbidity / DLS check on the actual vc-MMAE stock before "
                          f"treating this margin as specification.")
            else:
                action = (f"Measure vc-MMAE solubility in {cosolv}/PBS at {temp_C:.0f} °C "
                          f"(or confirm the working stock dilutes into the reaction medium without turbidity). "
                          f"To dissolve {work:.0f} µM at {fv*100:.0f}% co-solvent the required σ ≈ {sigma_req:.1f}; "
                          f"the class average is {sigma:.2f}. A higher true σ, a higher true pure-water S_w than "
                          f"the GSE estimate, or kinetic tolerance of fast maleimide–thiol conjugation would all "
                          f"close this gap — all resolve only with a measurement.")
        items.append(dict(
            step="linker-payload QC",
            item="Linker-payload solubility in the co-solvent reaction medium",
            value=round(fold, 1),
            unit="x (working conc / mixed-solvent solubility; >1 = above estimate)",
            reference="mixed-solvent solubility at configured co-solvent fraction [C1]; pure-water GSE kept only as f=0 reference",
            level=lv,
            confidence="C" if cal else "C (σ uncalibrated)",
            meaning=meaning,
            action=action,
        ))
    items.append(dict(
        step="linker-payload QC",
        item="Computed partition coefficient logP",
        value=res["props"]["mol_logp"],
        unit="(Crippen)",
        reference="≤ 5 as a warning reference (needs calibration with your own data)",
        level="warn" if res["props"]["mol_logp"] > 5 else "ok",
        confidence="C",
        meaning="the higher the logP, the more it tends to self-aggregate in aqueous phase rather than spread out",
        action="when logP > 5, always check the reactive-end availability, not just the feed ratio",
    ))
    return items


# ============================================================================
# Clustering risk and the hydrophobic burden of the conjugated product
# ============================================================================
# Bench picture. Dissolve the linker-payload in the reaction medium. Below a
# certain concentration each molecule sits on its own and its maleimide end is
# reachable. Above it the molecules clump together, the maleimide ends up inside
# the clump, and feeding more stops raising the DAR. That concentration is the
# critical aggregation concentration (CAC).
#
# Model form (config payload_qc.cac_*; same constants as dar_v6_complete.py):
#     log10 CAC_water  = cac_logP_c0 - cac_logP_c1 * logP
#     log10 CAC_medium = log10 CAC_water + cosolvent_sigma * f_vv
#     f_available      = min(1, CAC_medium / working_conc)
# TIER C: calibrated on one payload pair in one medium. It ranks candidates; it
# does not give a calibrated absolute concentration. A measured CAC in
# known_payloads always wins over the logP correlation.
#
# "How much will cluster" is reported as a LOWER BOUND, not a probability. If
# the working concentration is C and only CAC of it can stay as single
# molecules, then at least (C - CAC) / C of the material sits in clumps. It is a
# lower bound because some clustering happens below the CAC as well.
CAC_LOGP_C0_DEFAULT = -1.60
CAC_LOGP_C1_DEFAULT = 0.60
CAC_SIGMA_DEFAULT = 1.60


def _lookup_anchor(known_payloads, name):
    """Find a measured CAC anchor by payload name, tolerating aliases and case.

    A screening run names the molecule the way the chemist names it ("eribulin",
    "艾日布林"), while the config keys it the way the patent does ("DL012"). A
    failed lookup silently degrades a measured payload to a structure-only
    estimate, which is exactly how the eribulin false negative happened, so the
    lookup matches the key, any "aliases" list, and case-insensitive substrings.
    """
    if not known_payloads or not name:
        return {}
    if name in known_payloads:
        return known_payloads[name] or {}
    low = str(name).lower()
    for key, val in known_payloads.items():
        if str(key).lower() == low:
            return val or {}
        aliases = [str(a).lower() for a in (val or {}).get("aliases", []) or []]
        if low in aliases or any(a in low for a in aliases if a):
            return val or {}
        if str(key).lower() in low or low in str(key).lower():
            return val or {}
    return {}


def clustering_report(res, working_uM, cosolvent="DMSO", f_vv=0.10, name=None,
                      cac_measured_M=None, c0=None, c1=None, sigma=None,
                      known_payloads=None, smix_uM=None):
    """Structure-only clustering risk for one linker-payload.

    Returns the usable fraction, the lower bound on how much of the material
    sits in clumps, a low/medium/high band, and plain-language lines.

    smix_uM is the co-solvent solubility estimate from attach_solubility(); when
    given, it is used as the optimistic end of a range. The two ends come from
    two different models (logP -> CAC, and log-linear cosolvency solubility)
    that are known to disagree for payloads carrying a polar solubilising tail,
    so both are reported rather than one being silently picked.
    """
    c0 = CAC_LOGP_C0_DEFAULT if c0 is None else float(c0)
    c1 = CAC_LOGP_C1_DEFAULT if c1 is None else float(c1)
    sigma = CAC_SIGMA_DEFAULT if sigma is None else float(sigma)
    logp = float(res["props"]["mol_logp"])
    work = float(working_uM)

    measured = False
    if cac_measured_M is None and name and known_payloads:
        anc = _lookup_anchor(known_payloads, name)
        if anc.get("cac_water_M") is not None:
            measured = True
            log10_water = math.log10(float(anc["cac_water_M"]))
            sig = float(anc.get("cosolvent_sigma", sigma))
            cac_uM = 10.0 ** (log10_water + sig * f_vv) * 1e6
        else:
            cac_uM = 10.0 ** (c0 - c1 * logp + sigma * f_vv) * 1e6
    elif cac_measured_M is not None:
        measured = True
        cac_uM = float(cac_measured_M) * 1e6
    else:
        cac_uM = 10.0 ** (c0 - c1 * logp + sigma * f_vv) * 1e6

    f_avail = min(1.0, cac_uM / work) if work > 0 else 1.0
    clumped = max(0.0, 1.0 - f_avail)
    if clumped < 0.20:
        band_struct = "low"
    elif clumped < 0.50:
        band_struct = "medium"
    else:
        band_struct = "high"
    # A structure-only estimate is not evidence of safety. Eribulin (logP 2.91) is scored
    # "low" by the logP correlation yet measurably clumps at ~296-364 uM in 8 % DMSO, so the
    # pure-structure branch has a demonstrated false negative. Without a measured anchor the
    # band must stay "unknown" so no screening caller can read it as a pass.
    band = band_struct if measured else "unknown"

    lines = [
        f"logP = {logp:.2f}; estimated CAC in {f_vv*100:.0f}% {cosolvent} = "
        f"{cac_uM:.0f} uM ({'measured anchor' if measured else 'logP correlation, TIER C'}).",
        f"Working concentration = {work:.0f} uM, i.e. {work/cac_uM:.2f}x the CAC."
        if cac_uM > 0 else "CAC could not be estimated.",
        f"Usable fraction = {f_avail:.2f}; at least {clumped*100:.0f}% of the material "
        f"is expected to sit in clumps rather than as single molecules.",
        f"Band: {band}.",
    ]
    if not measured:
        lines.append(
            "NO MEASURED ANCHOR for this payload. The percentage above is a WORST-CASE upper "
            "bound from a single logP correlation, and it is known to over-flag: vc-MMAE "
            "(logP 5.08) is scored 'high' by it yet reaches DAR 8 in practice, because it "
            "carries a polar Val-Cit-PAB tail. Use this number to reject candidates, never to "
            "predict a DAR. One turbidity or DLS reading turns it into a real number.")
    if band == "high":
        lines.append("Feeding more past this point is not expected to raise the DAR: the extra "
                     "linker-payload goes into clumps whose maleimide is buried.")
    elif band == "medium":
        lines.append("Part of the feed is expected to be unusable; treat the DAR ceiling as "
                     "soft and confirm with one turbidity or DLS check.")
    elif band == "low":
        lines.append("Most of the feed is expected to stay single; the DAR ceiling is set by "
                     "the sites and the feed, not by clustering.")
    else:
        lines.append("UNKNOWN: this payload has no measured anchor, and a clean structure-only "
                     "score has already produced one false negative (eribulin). Do not clear "
                     "this payload on this number. One turbidity or DLS reading at the working "
                     "concentration turns it into a real, B-tier verdict.")

    # Optimistic end: the co-solvent solubility model (log-linear cosolvency).
    clumped_opt = None
    if smix_uM is not None and math.isfinite(smix_uM) and smix_uM > 0 and work > 0:
        clumped_opt = max(0.0, 1.0 - min(1.0, float(smix_uM) / work))
    if clumped_opt is not None:
        lo, hi = sorted((clumped_opt, float(clumped)))
        lines.append(
            f"Range across the two models: {lo*100:.0f}-{hi*100:.0f}% clustered "
            f"(solubility model {clumped_opt*100:.0f}%, CAC model {clumped*100:.0f}%). "
            f"For payloads carrying a polar tail (large tPSA) the CAC model over-flags; "
            f"for payloads without one the CAC model is the conservative end to trust."
            if abs(clumped - clumped_opt) > 0.05 else
            f"Both models agree: about {clumped*100:.0f}% clustered.")

    return dict(
        name=name, logp=logp, cosolvent=cosolvent, cosolvent_vv=f_vv,
        cac_uM=float(cac_uM), working_uM=work,
        f_available=float(f_avail),
        clustered_fraction_lower_bound=float(clumped),
        clustered_fraction_optimistic=clumped_opt,
        band=band, band_structural=band_struct, measured=measured,
        screen_action=("proceed" if (measured and band == "low") else
                       "confirm with turbidity/DLS, do not raise feed" if measured else
                       "unknown: one turbidity or DLS reading required before clearing"),
        tier="B" if measured else "C",
        model="log10 CAC = c0 - c1*logP + sigma*f  (single-payload-pair calibration, TIER C)",
        lines=lines,
    )


def adc_hydrophobic_burden(dar, payload_nonpolar_A2, dar_ref=4.0):
    """Hydrophobic area the conjugated product carries, per antibody.

    Conjugation glues `dar` copies of the payload onto the antibody surface.
    Each copy contributes the nonpolar area it exposes, so the added hydrophobic
    area is dar * payload_nonpolar_A2. That number drives both HIC retention and
    the tendency of the finished ADC to stick to itself, so it is reported as
    the deliverable; the banding thresholds are TIER C placeholders because no
    calibrated limit exists yet.
    """
    per = float(payload_nonpolar_A2)
    added = float(dar) * per
    added_ref = float(dar_ref) * per
    ratio = added / added_ref if added_ref > 0 else float("nan")
    if math.isfinite(ratio) and ratio <= 1.0:
        band = "low"
    elif math.isfinite(ratio) and ratio <= 1.6:
        band = "medium"
    else:
        band = "high"
    return dict(
        dar=float(dar), per_payload_nonpolar_A2=round(per, 1),
        added_nonpolar_A2=round(added, 1),
        reference_dar=float(dar_ref),
        added_nonpolar_A2_at_reference=round(added_ref, 1),
        ratio_vs_reference=(round(ratio, 2) if math.isfinite(ratio) else None),
        band=band, tier="C",
        note=("Added hydrophobic area is a structure-only quantity. It drives HIC retention "
              "and self-association of the finished ADC, but the low/medium/high cut-offs are "
              "placeholders referenced to a DAR-4 process and need one HIC/SEC run to anchor."),
    )


# --------------------------------------------------------------------------- #
# Screening chain: steric fit + reaction kinetics + one-line verdict
# --------------------------------------------------------------------------- #
# A screening run asks four things, and before this block only two of them could
# be answered from here:
#   1  what is this molecule           -> analyse()            (above)
#   2  will it stay dissolved          -> clustering_report()   (above)
#   3  does it physically fit the site -> steric_check()        (here)
#   4  does it finish reacting in time -> kinetics_check()      (here)
#   5  overall verdict                 -> screen_payload()      (here)
# Steps 3 and 4 existed only inside stand-alone scripts, so every screening run
# had to be assembled by hand and the four answers were never combined.
# --------------------------------------------------------------------------- #

_REPLACE_COLLAPSED = None


def _load_replace_collapsed(path=None):
    """Import the placement script by path.

    examples/trop_adc/runpod/replace_collapsed.py is a command-line script, not
    an importable module, so it is loaded the way the test suite loads the main
    model. The pose search is deliberately NOT re-implemented here: a second
    copy would drift from the one that produced the delivered DAR1-DAR8
    structures, and a screening answer that contradicts those structures is
    worse than no answer at all.
    """
    global _REPLACE_COLLAPSED
    if _REPLACE_COLLAPSED is not None:
        return _REPLACE_COLLAPSED
    path = path or os.path.join(_PROJ, "examples", "trop_adc", "runpod",
                                "replace_collapsed.py")
    if not os.path.exists(path):
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("_replace_collapsed", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    _REPLACE_COLLAPSED = mod
    return mod


def steric_check(sdf_path=None, ref_pdb=None, npz_path=None, cutoff=2.0,
                 n_frames=12, n_roll=24, n_dirs=32):
    """Screening step 3: does this molecule physically fit on the cysteine sites?

    The question, in bench terms: the antibody has eight thiols and several of
    them sit in the hinge, a narrow gap between the two heavy chains. A payload
    that is bulky, or that refuses to curl up, simply may not have room there no
    matter how much of it is added. That is a geometry question, so it is
    answered by geometry: place the molecule on every site of a real MD snapshot
    and measure how close its heavy atoms come to the protein.

    Heavy atoms only. Hydrogens are excluded on purpose - they rotate, and
    counting them once produced a false "collision" on a pose that was in fact
    fine. The threshold is a heavy-atom contact distance in Angstrom.

    Returns per-site clearance plus the multi-site answer (how many of the eight
    can be occupied at once), because the two questions differ: a molecule can
    fit alone at every site and still not fit eight times over.
    """
    out = dict(available=False, step="steric fit", cutoff=cutoff)
    rc = _load_replace_collapsed()
    if rc is None:
        out["reason"] = "placement script replace_collapsed.py not found"
        return out
    sdf_path = sdf_path or _DEFAULT_ENSEMBLE
    if not os.path.exists(sdf_path):
        out["reason"] = f"no conformer file at {sdf_path}"
        return out
    ref_pdb = ref_pdb or os.path.join(
        _PROJ, "examples", "trop_adc", "structures",
        "trastuzumab_alphafold_model2.pdb")
    npz_path = npz_path or os.path.join(
        _PROJ, "examples", "trop_adc", "data", "md_af_nolock", "allcut_r2.npz")
    if not (os.path.exists(ref_pdb) and os.path.exists(npz_path)):
        out["reason"] = "MD frames or reference structure not found"
        return out
    try:
        pools = rc.build_tier_pools(sdf_path)
    except Exception as exc:                                   # pragma: no cover
        out["reason"] = f"conformer pool could not be built ({exc})"
        return out
    if not pools:
        out["reason"] = ("no usable conformers - the file must carry the maleimide "
                         "anchor atoms named C85/C91, like the delivered ensembles")
        return out
    frames, meta, _rows, sg_idx = rc.load_md_context(ref_pdb, npz_path)
    sites = sorted(sg_idx.keys())
    stride = max(1, frames.shape[0] // max(1, n_frames))
    # full load: occupy all eight sites on the same snapshot, so the answer
    # already includes payload-payload crowding
    full = rc.scan_snapshots(frames, meta, sg_idx, sites, pools, cutoff,
                             n_roll, n_dirs, stride=stride)
    best = full[0]
    # single load: same snapshot, one site at a time, to see WHICH site is tight
    per_site = {}
    for s in sites:
        one = rc.scan_snapshots(frames, meta, sg_idx, [s], pools, cutoff,
                                n_roll, n_dirs, frame_list=[best["t"]])
        b = one[0]
        per_site["%s%s" % s] = dict(
            placed=int(b["n"]), dmin=round(float(b["dmin"]), 2),
            ok=bool(b["n"] > 0 and b["n_ok"] > 0),
            tier=(b["tiers"][0] if b["tiers"] else None))
    n_ok_sites = sum(1 for v in per_site.values() if v["ok"])
    worst = min((v["dmin"] for v in per_site.values()), default=float("nan"))
    out.update(
        available=True,
        n_sites=len(sites),
        n_frames_scanned=len(range(0, frames.shape[0], stride)),
        best_frame=int(best["t"]),
        max_sites_occupied=int(best["n"]),
        all_eight_fit=bool(best["n"] == len(sites) and best["n_ok"] == len(sites)),
        sites_clear_single=n_ok_sites,
        worst_dmin_A=round(float(worst), 2),
        per_site=per_site,
        tier="B",
        note=("Geometry from real MD snapshots, so the shape is physical; the "
              "contact-distance cut-off is an empirical choice and the conformers "
              "are MMFF gas-phase, so this ranks and flags rather than predicts "
              "a yield. A site flagged tight is a real crowding warning."),
    )
    # Separate report: how many LP hydrogens would falsely count as clashes if H
    # were included. This does NOT change the heavy-atom verdict above; it only
    # quantifies why hydrogens are excluded (they rotate and produce false positives).
    try:
        mol = pools[0][1][0]["mol"]
        prot_model = load_antibody(ref_pdb)
        _, protein_xyz = heavy_atoms(prot_model)
        h_false = 0
        for (_site, _sgi, r) in best["placed"]:
            placed_heavy = np.asarray(r["new"], dtype=float)
            h_false += hydrogen_false_collision_count(
                protein_xyz, placed_heavy, mol, cutoff)
        out["hydrogen_false_collisions"] = int(h_false)
        out["hydrogen_false_collisions_note"] = (
            "LP hydrogens within the contact cutoff of the protein if H were counted "
            "as a clash. These are false (H rotates); this is why the verdict uses "
            "heavy atoms only. A non-zero number is expected, not a failure.")
    except Exception:  # pragma: no cover
        pass
    return out


# --------------------------------------------------------------------------- #
# Continuous steric discount.
#
# The whole point of the screening chain was to stop answering "does it fit? yes/no"
# and start answering "how much does crowding lower the chance it actually couples?"
# vc-MMAE is the reference: it really reached DAR 8, so its per-site frame-wise
# clearance distribution IS "a molecule that couples fine on this antibody". A new
# molecule is compared to that distribution: the fraction of frames in which it
# clears the cut-off, divided by the same fraction for vc-MMAE, is how much of the
# conjugation probability it keeps. Every term in the ratio is measured geometry
# (heavy-atom clearance on real MD snapshots) - there is no adjustable parameter
# and nothing fitted to a measured DAR.
# --------------------------------------------------------------------------- #

_STERIC_REF_PATH = os.path.join(
    _PROJ, "examples", "trop_adc", "data", "steric_ref_vcmmae.json")


def _clear_fraction(dmins, cutoff):
    """Fraction of frames in which the payload clears `cutoff` Angstrom."""
    if not dmins:
        return 0.0
    return float(sum(1.0 for d in dmins if d >= cutoff) / len(dmins))


def _load_steric_ref(path=None):
    """Load the vc-MMAE reference clearance distribution."""
    import json
    path = path or _STERIC_REF_PATH
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return None


def steric_discount(sdf_path=None, ref_pdb=None, npz_path=None, cutoff=2.0,
                    n_frames=12, n_roll=24, n_dirs=32, ref_path=None):
    """Per-site conjugation probability discount from crowding.

    In bench terms: the antibody has eight thiols and the hinge is tight. A
    payload that leaves less room than vc-MMAE did on the same snapshots will
    couple to fewer thiols in the same reaction time; how much fewer is the
    ratio of "frames where it clears" between the new molecule and vc-MMAE.

    Returns
    -------
    dict with per-site `p_steric` (0..1, 1 = no discount) plus the reference
    fractions used, so the number is always traceable back to geometry.
    """
    ref = _load_steric_ref(ref_path)
    if ref is None:
        return dict(available=False, reason="reference file not found")
    if not _HAS_RDKIT:
        return dict(available=False, reason="rdkit missing")
    rc = _load_replace_collapsed()
    if rc is None:
        return dict(available=False, reason="placement script not found")
    sdf_path = sdf_path or _DEFAULT_ENSEMBLE
    if not os.path.exists(sdf_path):
        return dict(available=False, reason=f"no conformer file at {sdf_path}")
    ref_pdb = ref_pdb or os.path.join(
        _PROJ, "examples", "trop_adc", "structures",
        "trastuzumab_alphafold_model2.pdb")
    npz_path = npz_path or os.path.join(
        _PROJ, "examples", "trop_adc", "data", "md_af_nolock", "allcut_r2.npz")
    if not (os.path.exists(ref_pdb) and os.path.exists(npz_path)):
        return dict(available=False, reason="MD frames or reference structure not found")
    try:
        pools = rc.build_tier_pools(sdf_path)
    except Exception as exc:                                   # pragma: no cover
        return dict(available=False, reason=f"conformer pool could not be built ({exc})")
    if not pools:
        return dict(available=False, reason="no usable conformers (need C85/C91 anchors)")
    frames, meta, _rows, sg_idx = rc.load_md_context(ref_pdb, npz_path)
    sites = sorted(sg_idx.keys())
    stride = max(1, frames.shape[0] // max(1, n_frames))
    frame_idx = list(range(0, frames.shape[0], stride))
    ref_by_site = ref.get("per_site_dmin_by_frame", {})

    per_site, discounts = {}, {}
    for (ch, resi) in sites:
        key = "%s%s" % (ch, resi)
        scan = rc.scan_snapshots(frames, meta, sg_idx, [(ch, resi)], pools,
                                 cutoff, n_roll, n_dirs, frame_list=frame_idx)
        dmins = [float(r["dmin"]) for r in scan]
        clear = _clear_fraction(dmins, cutoff)
        ref_clear = _clear_fraction(ref_by_site.get(key, []), cutoff)
        if ref_clear <= 0.0:                    # no reference frames cleared
            p = 1.0
        else:
            p = min(1.0, clear / ref_clear)
        per_site[key] = dict(
            dmin_med=round(float(sorted(dmins)[len(dmins)//2]), 3)
            if dmins else None,
            dmin_min=round(float(min(dmins)), 3) if dmins else None,
            clear_fraction=round(clear, 4),
            ref_clear_fraction=round(ref_clear, 4),
            p_steric=round(float(p), 4),
        )
        discounts[key] = p
    mean_p = float(sum(discounts.values()) / len(discounts)) if discounts else 1.0
    min_p = float(min(discounts.values())) if discounts else 1.0
    return dict(
        available=True,
        molecule="unknown (compared against vc-MMAE reference)",
        ref_molecule=ref.get("molecule", "vc-MMAE"),
        cutoff=cutoff,
        n_frames=len(frame_idx),
        mean_p_steric=round(mean_p, 4),
        min_p_steric=round(min_p, 4),
        sites_like_vcmmae=float(sum(1 for v in discounts.values() if v >= 0.99)),
        n_sites=len(discounts),
        per_site=per_site,
        note=("p_steric is the ratio of measured clear fractions (new molecule / "
              "vc-MMAE) on the same MD snapshots, capped at 1. No fitted parameter; "
              "reference traceable to steric_ref_vcmmae.json."),
    )


def f_thiolate(pka, ph):
    """Fraction of thiols carrying a negative charge at this pH - the reacting form."""
    return 1.0 / (1.0 + 10.0 ** (pka - ph))


def load_site_teff(path=None, t_total_s=7200.0, labels=None):
    """Effective exposure time per site, from the stored MD SASA series.

    In bench terms: a thiol that spends most of its time buried inside the
    antibody is only open for reaction part of the time. The MD trajectory says
    what fraction of the time each of the eight sites is open; multiplying that
    fraction by the reaction window gives the time it is actually available.
    """
    path = path or os.path.join(
        _PROJ, "examples", "trop_adc", "results_hinge",
        "md_af_allcut_4rep_series.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    sasa = d["sasa_A2"]
    lo = float(d["sasa_gate_lo"])
    span = float(d["sasa_gate_span"])
    g = np.clip((sasa - lo) / span, 0.0, 1.0)
    frac = g.reshape(-1, g.shape[-1]).mean(axis=0)
    names = [str(x) for x in (labels if labels is not None else d["labels"])]
    return {n: float(t_total_s * f) for n, f in zip(names, frac)}


def kinetics_check(t_eff_by_site, feed_uM, f_available=1.0, ph=7.0, pka=8.3,
                   k2_ref=300.0, ph_ref=7.0, k_hyd=0.0, t_total_s=7200.0,
                   yield_max=0.995, band_floor=0.90):
    """Screening step 4: does the reaction finish inside the process window?

    The question, in bench terms: the thiol is open, the payload is dissolved -
    but is two hours at this pH actually enough for the bond to form? The answer
    is the fraction of sites that end up conjugated, per site, where the "clock"
    is the time the site was open and the "concentration" is the payload that is
    genuinely available (after clustering is deducted).

    This mirrors pc_from_teff() in the main model on purpose: same pH term, same
    rate constant, same saturation form, so the screening answer and the model
    answer cannot disagree. It is re-stated here rather than imported because the
    main model is a script, not a module; tests/test_chemical.py pins the two
    together.
    """
    ft = f_thiolate(pka, ph)
    ft_ref = f_thiolate(pka, ph_ref)
    k2 = k2_ref * (ft / ft_ref) if ft_ref > 0 else k2_ref
    p0 = float(feed_uM) * float(f_available) * 1e-6          # mol/L available
    per_site, vals = {}, []
    for name, t_eff in (t_eff_by_site or {}).items():
        if k_hyd <= 1e-12:
            auc = p0 * float(t_eff)
        else:
            auc = (p0 * (float(t_eff) / t_total_s)
                   * (1.0 - math.exp(-k_hyd * t_total_s)) / k_hyd)
        pc = yield_max * (1.0 - math.exp(-k2 * auc))
        per_site[name] = round(float(pc), 4)
        vals.append(float(pc))
    lo = min(vals) if vals else float("nan")
    mean = sum(vals) / len(vals) if vals else float("nan")
    if math.isfinite(lo) and lo >= band_floor:
        band = "ok"
    elif math.isfinite(lo) and lo >= 0.5:
        band = "slow"
    else:
        band = "incomplete"
    return dict(
        step="reaction kinetics",
        available=bool(vals),
        thiolate_fraction=round(float(ft), 4),
        k2_apparent=round(float(k2), 2),
        available_conc_uM=round(float(feed_uM) * float(f_available), 1),
        per_site_pc=per_site,
        pc_min=round(float(lo), 4),
        pc_mean=round(float(mean), 4),
        band=band,
        tier="B",
        note=("Same pH term and rate constant as the main model. The warhead, not "
              "the payload, sets the rate: two payloads sharing a maleimide react "
              "at the same speed, so a slow verdict here points at pH, time, or "
              "available concentration - not at the payload's identity."),
    )


def screen_payload(res=None, working_uM=None, name=None, cosolvent="DMSO",
                   f_vv=0.10, steric=None, kinetics=None, limits=None,
                   known_payloads=None, c0=None, c1=None, sigma=None):
    """Screening verdict: combine the four answers into one line.

    Everything above returns a number that has to be interpreted. This function
    does the interpreting and, more importantly, says what to do next:

        proceed          - nothing blocked, take it to the bench
        measure_first    - not blocked, but a needed number is missing
        not_recommended  - something is physically blocked

    `measure_first` is deliberately its own verdict. The failure it replaces was
    a molecule that passed screening on structure alone and then clumped in the
    real reaction; a clean-looking "low risk" was the bug, not the molecule.
    """
    limits = limits or {}
    blockers, measures, lines = [], [], []
    cfg = load_cfg()
    known_payloads = cfg["known_payloads"] if known_payloads is None else known_payloads
    # ---- step 2: solubility / clustering ----
    cl = None
    if res is not None and working_uM:
        cl = clustering_report(res, working_uM, cosolvent=cosolvent, f_vv=f_vv,
                               name=name, known_payloads=known_payloads,
                               c0=c0, c1=c1, sigma=sigma,
                               smix_uM=res["summary"].get("smix_uM_configured"))
        lines.extend(cl["lines"])
        if cl["band"] == "high":
            blockers.append("clustering")
        elif cl["band"] == "unknown":
            measures.append(cl["screen_action"])
        f_avail = float(cl.get("f_available", 1.0))
    else:
        f_avail = 1.0
        measures.append("solubility: no working concentration given, step skipped")
    # ---- step 1: reactive end buried? ----
    if res is not None:
        f = res["summary"]["f_active"]
        crit = limits.get("f_active_crit", cfg["f_active_crit"])
        if f < crit:
            blockers.append("reactive end buried")
            lines.append(f"reactive-end availability {f:.2f} is below {crit}: "
                         "the molecule curls up and hides its maleimide.")
    # ---- step 3: steric fit ----
    if steric is not None:
        if not steric.get("available"):
            measures.append(f"steric fit not run ({steric.get('reason', 'no input')})")
        else:
            tight = [k for k, v in steric["per_site"].items() if not v["ok"]]
            lines.append(
                f"steric fit: {steric['sites_clear_single']}/{steric['n_sites']} sites "
                f"clear alone, worst heavy-atom clearance {steric['worst_dmin_A']} A, "
                f"{'all eight' if steric['all_eight_fit'] else steric['max_sites_occupied']} "
                f"fit at once (cut-off {steric['cutoff']} A).")
            # Only a site that cannot hold the payload AT ALL is a hard block.
            # Eight-at-once crowding that forces a less compact pose is not: the
            # molecule can extend, and an extended pose still couples. Treating
            # "needs to extend" as a block would wrongly reject vc-MMAE, which
            # reaches DAR 8 on the bench.
            if (steric["max_sites_occupied"] < steric["n_sites"]
                    or steric["sites_clear_single"] < steric["n_sites"]):
                blockers.append("steric fit")
            elif not steric["all_eight_fit"]:
                lines.append(
                    "eight at once forces some payloads into a less compact pose - "
                    "real crowding in the hinge, but not blocking: an extended pose "
                    "still couples, only less snugly.")
            if tight:
                lines.append("tight sites: " + ", ".join(tight))
    else:
        measures.append("steric fit not run: give a conformer file with C85/C91 anchors")
    # ---- step 4: kinetics ----
    if kinetics is not None:
        if not kinetics.get("available"):
            measures.append("kinetics not run (no site exposure times)")
        else:
            lines.append(
                f"kinetics: worst site {kinetics['pc_min']:.3f}, mean "
                f"{kinetics['pc_mean']:.3f} at pH with "
                f"{kinetics['thiolate_fraction']*100:.1f}% thiolate "
                f"(band {kinetics['band']}).")
            if kinetics["band"] == "incomplete":
                blockers.append("reaction does not finish")
            elif kinetics["band"] == "slow":
                blockers.append("reaction slow")
    else:
        measures.append("kinetics not run: give per-site exposure times")
    if blockers:
        verdict = "not_recommended"
    elif measures:
        verdict = "measure_first"
    else:
        verdict = "proceed"
    return dict(
        verdict=verdict, blockers=blockers, next_measurements=measures,
        f_available=f_avail, clustering=(cl or {}), lines=lines,
        tier=(cl or {}).get("tier", "C"),
    )


def main():
    if not _HAS_RDKIT:
        print("rdkit missing, cannot run"); return
    cfg = load_cfg()
    ens = os.path.join(_PROJ, cfg["ensemble"]) if cfg["ensemble"] else None
    mols, src = load_ensemble(ens)
    print("=" * 96)
    print("LP QC: morphology and homogeneity stability of the small molecule in conjugation solution")
    print("=" * 96)
    print(f"Conformational ensemble source: {src}")
    print(f"Conjugation system: PBS pH 7.0 + {cfg['co_solvent_vv']*100:.0f}% organic co-solvent, "
          f"{cfg['temperature_C']:.0f} °C (read from adc_model_config.yaml)")
    p = molprops(mols[0])
    print(f"\nMolecular properties: MW={p['mw']}  logP={p['mol_logp']}  TPSA={p['tpsa']}  atoms={p['n_atoms']}")
    print(f"logP {p['mol_logp']} → {'rather hydrophobic, worth vigilance' if p['mol_logp'] > 5 else 'acceptable'}")

    res = analyse(mols, temperature_C=cfg["temperature_C"])
    s = res["summary"]
    rows = res["rows"]

    print(f"\nReactive-end atom indices {res['reactive_atoms']}, exposure threshold {res['reactive_sasa_threshold']} Å²")
    print(f"\n{'conf':>4}{'weight':>8}{'Rg(Å)':>8}{'reactSASA':>11}{'nonpolar':>11}{'selfBurial':>9}  form")
    top = sorted(range(len(rows)), key=lambda i: -rows[i]["weight"])[:12]
    for i in top:
        r = rows[i]
        print(f"{r['i']:>4}{r['weight']:>8.3f}{r['rg']:>8.2f}{r['reactive_sasa']:>11.1f}"
              f"{r['sasa_nonpolar']:>11.0f}{r['hydrophobic_self_burial']:>9.2f}"
              f"  {'collapsed·reactive end buried' if r['collapsed'] else 'extended'}")

    print(f"\n{'─'*96}\nPopulation-weighted results")
    print(f"  Reactive-end availability f_active = {s['f_active']:.3f}"
          f"   ({s['f_active']*100:.1f}% of molecules can react without first untying their own hydrophobic core)")
    print(f"  Mean reactive-end exposed area     = {s['mean_reactive_sasa_A2']:.1f} Å²")
    print(f"  Mean hydrophobic self-burial      = {s['mean_hydrophobic_self_burial']*100:.1f}%")
    print(f"  Mean radius of gyration            = {s['mean_rg_A']:.2f} Å"
          f" (ensemble range {s['rg_spread_A'][0]:.2f}–{s['rg_spread_A'][1]:.2f})")
    print(f"  Top-conformer population           = {s['top_conformer_population']:.3f}"
          f"   population entropy = {s['effective_population_entropy']:.2f} (larger = more heterogeneous)")

    # ---- Co-solvent-aware solubility vs working concentration ----
    feed = cfg.get("feed_ratio", 6.0)
    mab_uM = cfg.get("mab_uM", 67.0)
    work_uM = feed * mab_uM
    cosolvent = cfg.get("cosolvent", "DMSO")
    f_cfg = cfg.get("co_solvent_vv", 0.10)
    sigma_ovr = cfg.get("sigma_override", None)
    res = attach_solubility(res, work_uM, cosolvent=cosolvent,
                            f_configured=f_cfg, sigma_override=sigma_ovr)
    s = res["summary"]

    print(f"\n{'─'*96}\nCo-solvent solubility vs working concentration "
          f"(log-linear cosolvency model [C1]; σ from [C3])")
    print(f"  Co-solvent: {cosolvent} at configured fraction f = {f_cfg*100:.0f}% v/v "
          f"(read from adc_model_config.yaml).")
    print(f"  Working concentration = {feed:g} equiv × {mab_uM:g} µM antibody = {work_uM:.0f} µM")
    print(f"  Pure-water GSE solubility (f = 0 LOWER BOUND, {GSE_SOURCE}): "
          f"{s['sw_uM_ref']:.1f} µM")
    print(f"  Estimated σ (co-solvent solubilizing power) = {s['sigma_est']:.2f} "
          f"({'MEASURED override' if s['sigma_calibrated'] else 'class average [C3], TO BE CALIBRATED'})")
    # Sensitivity table — the deliverable for wet-lab scoping.
    print(f"\n  {'co-solvent %':>11}{'est. mixed-solubility(µM)':>24}{'working/solubility':>20}{'verdict':>12}")
    for r in s["cosolvent_scan"]:
        fv = r["f_vv"]
        sm = r["smix_uM"]
        fold = r["fold"]
        if not math.isfinite(sm):
            verdict = "n/a (unknown σ)"
        elif not s["sigma_calibrated"]:
            verdict = "estimate*"
        elif fold <= 2:
            verdict = "ok"
        elif fold <= 10:
            verdict = "warn"
        else:
            verdict = "crit"
        print(f"  {fv*100:>10.0f}%{sm:>24.1f}{fold:>19.1f}×{verdict:>12}")
    print("  * estimate: σ is a class average [C3], not measured for vc-MMAE — calibrate before use.")
    print(f"  Pure-water GSE at f=0 → {s['sw_uM_ref']:.1f} µM, fold {work_uM/s['sw_uM_ref']:.1f}× "
          f"(this was the OLD, invalid criterion: it ignored co-solvent).")
    print(f"  σ required to dissolve {work_uM:.0f} µM at the configured {f_cfg*100:.0f}% co-solvent "
          f"≈ {s['sigma_required']:.1f} (class average is {s['sigma_est']:.2f}).")
    if not s["sigma_calibrated"]:
        print("  → The old alarm's '>15× over pure water → large amount undissolved, CRIT' is RETIRED;")
        print("    the co-solvent-aware estimate is preliminary and must be confirmed by measurement.")

    print(f"\n{'─'*96}\nHow to use it in the model")
    eff = s["f_active"]
    print(f"  · Intramolecular self-burial penalty: effective conc. × {eff:.3f}"
          f"  → {'no penalty (reactive end always faces out)' if eff > 0.999 else 'penalty applied'}")
    if math.isfinite(s["smix_uM_configured"]):
        print(f"  · Co-solvent solubility cap (estimate, TO BE CALIBRATED): effective conc. ≤ "
              f"{s['smix_uM_configured']:.0f} µM at {f_cfg*100:.0f}% {cosolvent} "
              f"(nominal working {work_uM:.0f} µM, fold {s['mix_fold_configured']:.1f}×)")
    print("    The two are independent mechanisms: the former is 'curling up on its own', the latter is")
    print("    'several molecules clumping together in the mixed solvent'.")
    print("    The mixed-solvent solubility is estimated with the log-linear cosolvency model and the")
    print("    co-solvent σ; both the aqueous base and σ are uncalibrated for vc-MMAE, so we emit an")
    print("    info/calibration alarm rather than folding an uncertain number into the concentration.")

    print(f"\n{'─'*96}\nalarm items")
    for a in to_alarms(res, limits=dict(f_active_warn=cfg["f_active_warn"],
                                        f_active_crit=cfg["f_active_crit"])):
        print(f"  [{a['level']}] {a['item']} = {a['value']} {a['unit']}  reference {a['reference']}")
        if a["level"] != "ok":
            print(f"        meaning: {a['meaning']}")
            print(f"        action: {a['action']}")

    # ---- Clustering risk and hydrophobic burden of the finished ADC ----
    cl = clustering_report(res, work_uM, cosolvent=cosolvent, f_vv=f_cfg,
                           name=cfg.get("payload_name", "vc-MMAE"),
                           c0=cfg["cac_logP_c0"], c1=cfg["cac_logP_c1"],
                           sigma=cfg["cac_cosolvent_sigma"],
                           known_payloads=cfg["known_payloads"],
                           smix_uM=s.get("smix_uM_configured"))
    print(f"\n{'─'*96}\nClustering risk: will the linker-payload clump before it can react?")
    for ln in cl["lines"]:
        print("  " + ln)
    print(f"  tier {cl['tier']}; model: {cl['model']}")

    a_np = s["mean_nonpolar_exposed_A2"]
    print(f"\n{'─'*96}\nHydrophobic burden of the finished ADC (per antibody)")
    print(f"  {'DAR':>5}{'added nonpolar area (A2)':>28}{'vs DAR-4 process':>18}{'band':>8}")
    for d in (0, 2, 4, 6, 8):
        b = adc_hydrophobic_burden(d, a_np)
        r = b["ratio_vs_reference"]
        print(f"  {d:>5}{b['added_nonpolar_A2']:>28.0f}"
              f"{(f'{r:.2f}' if r is not None else 'n/a'):>18}{b['band']:>8}")
    print(f"  one payload copy contributes {a_np:.0f} A2 of exposed nonpolar surface.")
    print(f"  {adc_hydrophobic_burden(4, a_np)['note']}")

    print(f"\n{'─'*96}\nMechanistic self-check: how hydrophobic must it be to collapse? (asp_scale scales the C/S hydrophobic solvation cost)")
    print("  This is not a prediction for any real molecule; it tests whether this module has resolving power.")
    print(f"  {'hydrophobic-cost ×':>12}{'reactive avail.':>14}{'mean self-burial':>14}{'exposed nonpolar area':>16}")
    for k in (1.0, 2.0, 3.0, 5.0, 8.0):
        r2 = analyse(mols, temperature_C=cfg["temperature_C"], asp_scale=k)["summary"]
        print(f"  {k:>12.1f}{r2['f_active']:>14.3f}"
              f"{r2['mean_hydrophobic_self_burial']*100:>13.1f}%"
              f"{r2['mean_nonpolar_exposed_A2']:>16.1f}")

    print(f"\n{'─'*96}\nboundaries")
    print("  · Implicit solvent (SASA-type) gives ranking, not true values; the ensemble itself is MMFF gas-phase generated.")
    print("  · Co-solvent is now used in the solubility assessment via the log-linear cosolvency model [C1]; "
          "the co-solvent σ is a class-average estimate [C3] and is flagged TO BE CALIBRATED for vc-MMAE.")
    print("  · **A single-molecule ensemble cannot see intermolecular aggregation**: this module's conformational ensemble contains only one molecule,")
    print("    so it can catch 'curling up and burying the reactive end' but not 'several molecules clumping together'.")
    print("    The latter can only be proxied by exposed nonpolar area, or measured directly with DLS / solubility data.")
    print(f"  · Solvation parameter source: {ASP_SOURCE}")


if __name__ == "__main__":
    main()
