"""Cysteine-conjugation ADC batches that failed or underperformed.

Each entry is a case where the process condition looked sufficient (enough
TCEP, enough payload) but the measured DAR fell short.  These are the cases
that can test whether the model can reproduce failure, not just success --
success cases do not separate a good model from a bad one.

Sources are public (patents, journals, white papers).  Fields marked None
were not reported in the source; do not invent them.
"""

# ---- Case 1: trisulfide consumes TCEP without freeing thiols ----
# Source: Cumnock et al., Bioconjugate Chem. 2013, 24, 1154-1160
#         DOI 10.1021/bc4000299
# Three batches of the same IgG1 with trisulfide levels 7.4%, 3.0%, 0.4%.
# Same "preset" TCEP:mAb ratio (target DAR 3.5).  High-trisulfide batch
# came back ~0.6 DAR lower.  Mechanism: TCEP reacts with trisulfide to
# give thiophosphine + disulfide, producing NO free thiol -- 2 eq TCEP
# per trisulfide is wasted before any interchain bond opens.
CUMNOCK_2013 = dict(
    id="Cumnock-2013",
    antibody="IgG1 (internal, target undisclosed)",
    payload="unspecified (cysteine-maleimide)",
    chemistry="interchain disulfide TCEP reduction + maleimide",
    tcep_eq=None,                  # "preset" ratio, value not published
    reduction_temp_C=None,
    reduction_time_h=None,
    ph=None,
    payload_feed_eq=None,
    mab_conc=None,
    conjugation_temp_C=None,
    conjugation_time_h=None,
    measured_dar=None,             # reported as ~0.6 below target for high-trisulfide batch
    dar_method="HIC",
    trisulfide_pct=[7.4, 3.0, 0.4],
    dar_gap_vs_target=0.6,
    failure_mechanism=(
        "trisulfide bonds react with TCEP but do not produce free thiols; "
        "each trisulfide wastes 2 eq TCEP before any interchain disulfide opens"),
    source="Bioconjugate Chem. 2013, 24, 1154-1160 (DOI 10.1021/bc4000299)",
    usability="B",                # has trisulfide%-DAR link + mechanism, misses TCEP/payload eq
)

# ---- Case 2: IgG4 harder to reduce than IgG1 under identical TCEP ----
# Source: Technology Networks white paper "Bioconjugation Chemistries for
#         ADC Preparation" (Table 2).  IgG1 2.2 eq TCEP -> DAR ~4; IgG4
#         same 2.2 eq -> DAR ~2.5.  Full dose-response for IgG4: 1/2/3.4/
#         4/5.4/6.4 eq -> 1.24/2.47/3.20/4.07/4.55/4.96.
# 25 mg/mL antibody, 37 C, 2 h reduction, 6.6 eq mc-LP payload.
IGG4_VS_IGG1 = dict(
    id="IgG4-vs-IgG1",
    antibody="IgG1 vs IgG4 (undisclosed name)",
    payload="mc-LP (MMAE/MMAF class)",
    chemistry="interchain disulfide TCEP reduction + maleimide",
    tcep_eq_igg1=2.2,
    tcep_eq_igg4_dose_response=[(1.0, 1.24), (2.0, 2.47), (3.4, 3.20),
                                (4.0, 4.07), (5.4, 4.55), (6.4, 4.96)],
    reduction_temp_C=37,
    reduction_time_h=2,
    ph=None,
    payload_feed_eq=6.6,
    mab_conc_mg_mL=25,
    conjugation_temp_C=None,
    conjugation_time_h=1,
    measured_dar_igg1=4.0,
    measured_dar_igg4_at_2_2eq=2.5,
    dar_method="HIC",
    failure_mechanism=(
        "IgG4 interchain disulfides are less accessible / harder to reduce "
        "than IgG1; a TCEP level that suffices for IgG1 is insufficient for IgG4"),
    source="Technology Networks white paper, Table 2 (exact URL not retained)",
    usability="A",                # full dose-response curve + measured DAR
)

# ---- Case 3: wild-type anti-PSMA, 38.7% DAR0 at TCEP 1.24 eq ----
# Source: TREA / Byondis patent "Dual conjugation process for preparing
#         antibody-drug conjugates", Table 1 (wild-type control entry 3e).
# Wild-type (no engineered Cys) at TCEP 1.24 eq -> DAR 1.8, DAR0 38.7%.
# Compare: engineered HC41 at TCEP 1.0 eq -> DAR 3.2, DAR0 0.6%.
ANTI_PSMA_WT = dict(
    id="anti-PSMA-WT-TCEP1.24",
    antibody="anti-PSMA (wild-type, non-engineered)",
    payload="vc-seco-DUBA (SYD980 class)",
    chemistry="interchain disulfide reduction + maleimide",
    tcep_eq=1.24,
    reduction_temp_C=None,
    reduction_time_h=None,
    ph=None,
    payload_feed_eq=None,
    mab_conc=None,
    conjugation_temp_C=None,
    conjugation_time_h=None,
    measured_dar=1.8,
    dar0_fraction=0.387,           # 38.7% of antibody completely unconjugated
    hmw_pct=12.2,
    dar_method="HIC",
    failure_mechanism=(
        "wild-type interchain disulfides of this antibody have low reduction "
        "efficiency under these conditions; nearly 4 in 10 antibodies get "
        "zero payload despite a TCEP level that should approach DAR 2"),
    source="TREA/Byondis patent, Table 1, entry 3e (wild-type control)",
    usability="B",                # TCEP-DAR-DAR0% complete; misses T/pH/conc
)

# ---- Case 4: engineered cysteine needs 50x reductant to clear DAR0 ----
# Source: Miret et al., ChemistrySelect 2020, 5(11), 3187-3190
#         DOI 10.1002/slct.201903913
# Trastuzumab_cys114 + vcMMAE, target DAR 2.  Low/medium reductant ->
# high DAR0 fraction.  Needed up to 50x molar excess to approach target.
MIRET_2020 = dict(
    id="Miret-2020-trz-cys114",
    antibody="Trastuzumab_cys114 (engineered cysteine, anti-HER2)",
    payload="vcMMAE",
    chemistry="engineered cysteine + maleimide (site-directed)",
    tcep_eq=None,                  # reported as "molar excess" up to 50x, exact value not tabulated
    reduction_temp_C=None,
    reduction_time_h=None,
    ph=None,
    payload_feed_eq=None,
    mab_conc=None,
    conjugation_temp_C=None,
    conjugation_time_h=None,
    measured_dar=None,            # qualitative: "high DAR0 until 50x excess"
    dar_method="HIC-HPLC / MS",
    failure_mechanism=(
        "engineered cysteine at position 114 is poorly reduced at low/medium "
        "reductant excess; large fraction of antibody remains unconjugated "
        "(high DAR0) until reductant is pushed to ~50x molar excess"),
    source="ChemistrySelect 2020, 5(11), 3187-3190 (DOI 10.1002/slct.201903913)",
    usability="C",                # qualitative only, no tabulated DAR
)

ALL_FAILURES = [CUMNOCK_2013, IGG4_VS_IGG1, ANTI_PSMA_WT, MIRET_2020]

if __name__ == "__main__":
    for c in ALL_FAILURES:
        print(f"\n--- {c['id']} (usability {c['usability']}) ---")
        for k in ("antibody", "payload", "chemistry", "tcep_eq",
                  "measured_dar", "failure_mechanism", "source"):
            v = c.get(k)
            if v is not None:
                print(f"  {k}: {v}")
