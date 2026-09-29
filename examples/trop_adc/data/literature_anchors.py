"""Cysteine-conjugation anchors mined from the public literature.

Every row has BOTH a TCEP equivalent AND a measured DAR (that was the
inclusion rule). Fields the source did not report are None — never guessed.

Why this file exists: the model's only knob is
    K = k_TCEP x [mAb] x t_reduction
so a row is only directly usable if the mAb concentration AND the reduction
time are known. Rows missing those are still kept, because the missing field
can often be recovered from the original paper — but they must not be fed to
the model until it is.

`usable` marks whether every field the model needs is present.
"""

# ---------------------------------------------------------------------------
# Fully specified: mAb concentration AND reduction time AND temperature all known
# ---------------------------------------------------------------------------
USABLE = [
    dict(id="T-DXd-WO2019044947", antibody="trastuzumab (T-DXd / MAAL-9001)",
         isotype="IgG1 humanised", payload="mc-GGGG-DXd", tcep_eq=5.0,
         tcep_note="patent states 0.3-3 eq PER DISULFIDE; full reduction ~5 eq/Ab",
         t_red_C=37, t_red_h=1.5, ph=7.5, mab_mg_mL=10.0, payload_eq=10.0,
         t_conj_C=20, t_conj_h=1.0, dar=8.0, dar_method="HIC 7.9 / LC-MS 7.94",
         source="WO2019044947A1 + FDA BLA761139 chemistry review",
         usable=True, note="full reduction; near the ceiling so weakly informative"),

    dict(id="886-13-US20220267467-Ex6",
         antibody="886-13 (IgG1-Fab / IgG1-Fc, engineered extra hinge disulfide)",
         isotype="IgG1", payload="mc-VC-PAB-MMAE", tcep_eq=4.4,
         t_red_C=10, t_red_h=3.0, ph=5.5, mab_mg_mL=4.0, payload_eq=10.0,
         t_conj_C=4, t_conj_h=1.0, dar=4.0, dar_method="HIC (D4 85.5%)",
         source="US20220267467A1 Example 6",
         usable=True, note="has an ENGINEERED extra hinge disulfide - not a wild-type IgG1"),

    dict(id="SHR-A1811", antibody="trastuzumab variant (SHR-1805)", isotype="IgG1",
         payload="mc-GGGG-SHR169265", tcep_eq=2.5, t_red_C=37, t_red_h=2.0,
         ph=7.5, mab_mg_mL=10.0, payload_eq=7.0, t_conj_C=4, t_conj_h=14.0,
         dar=6.0, dar_method="HIC (5.7 +/- 0.4)",
         source="representative protocol + Zhang et al. PLoS One 2025 "
                "(10.1371/journal.pone.0326691)",
         usable=True, note="tcep_eq 2.5 comes from a representative protocol, "
                           "must be checked against the Hengrui patent before use"),
]

# ---------------------------------------------------------------------------
# Missing a field the model needs (usually mAb concentration). Kept because
# the original paper can supply it; do NOT feed to the model as-is.
# ---------------------------------------------------------------------------
INCOMPLETE = [
    dict(id="TMAb-BioProcessIntl", antibody="TMAb (Genentech process-dev reference)",
         isotype="IgG1", payload="vcMMAE / mcMMAF", tcep_eq=1.9,
         t_red_C=20, t_red_h=3.0, ph=7.5, mab_mg_mL=None, payload_eq=5.0,
         t_conj_C=None, t_conj_h=None, dar=3.5, dar_method="HIC",
         source="BioProcess International, 'ADC Product Development Strategy'",
         usable=False, missing=["mab_mg_mL", "t_conj_C", "t_conj_h"],
         note="target DAR 3.5 reached at 1.9 eq; 1.5 eq fell short - a clean "
              "'not enough reductant' control"),

    dict(id="mAb1-1-native-Cumnock2013", antibody="mAb1-1 (7.4% trisulfide)",
         isotype="IgG1", payload="mc-vc-MMAE", tcep_eq=2.0,
         t_red_C=25, t_red_h=1.5, ph=7.5, mab_mg_mL=None, payload_eq=None,
         t_conj_C=None, t_conj_h=None, dar=3.46, dar_method="HIC",
         source="Cumnock et al., Bioconjug Chem 2013, 24, 1154-1160, Table 1 "
                "+ methods (RT, 90 min, pH 7.5)",
         usable=False,
         missing=["mab_mg_mL", "payload_eq"],
         mab_conc_note="methods say unconjugated mAbs were >5 g/L - a LOWER BOUND only, "
                       "so it cannot be used as the exact concentration K needs",
         note="TRISULFIDE case: native batch gives DAR 3.46 where 2.0 eq should "
              "give ~4.0. The 0.54 gap is TCEP wasted on trisulfides."),

    dict(id="mAb1-1-desulf-Cumnock2013", antibody="mAb1-1 (trisulfide removed)",
         isotype="IgG1", payload="mc-vc-MMAE", tcep_eq=2.0,
         t_red_C=25, t_red_h=1.5, ph=7.5, mab_mg_mL=None, payload_eq=None,
         t_conj_C=None, t_conj_h=None, dar=4.04, dar_method="HIC",
         source="Cumnock et al. 2013, Table 1 + methods", usable=False,
         missing=["mab_mg_mL", "payload_eq"],
         note="same antibody, trisulfide removed -> DAR recovers to 4.04. "
              "PAIR with the native row: the difference IS the trisulfide effect."),

    dict(id="mAb1-2-native-Cumnock2013", antibody="mAb1-2 (3.0% trisulfide)",
         isotype="IgG1", payload="mc-vc-MMAE", tcep_eq=2.0,
         t_red_C=25, t_red_h=1.5, ph=7.5, mab_mg_mL=None, payload_eq=None,
         t_conj_C=None, t_conj_h=None, dar=3.70, dar_method="HIC",
         source="Cumnock et al. 2013, Table 1 + methods", usable=False,
         missing=["mab_mg_mL", "payload_eq"],
         note="intermediate trisulfide level -> intermediate DAR loss"),

    dict(id="mAb1-2-desulf-Cumnock2013", antibody="mAb1-2 (trisulfide removed)",
         isotype="IgG1", payload="mc-vc-MMAE", tcep_eq=2.0,
         t_red_C=25, t_red_h=1.5, ph=7.5, mab_mg_mL=None, payload_eq=None,
         t_conj_C=None, t_conj_h=None, dar=4.00, dar_method="HIC",
         source="Cumnock et al. 2013, Table 1 + methods", usable=False,
         missing=["mab_mg_mL", "payload_eq"]),

    dict(id="CStone-002-MMAE", antibody="CStone 002 (anti-ITGB4)", isotype="IgG1 humanised",
         payload="mc-VC-PABC-MMAE", tcep_eq=2.35, t_red_C=37, t_red_h=1.0,
         ph=None, mab_mg_mL=None, payload_eq=5.5, t_conj_C=None, t_conj_h=None,
         dar=3.8, dar_method="HIC / RP-HPLC",
         source="WO2024/208354 A1, Table 6 (scheme A)", usable=False,
         missing=["ph", "mab_mg_mL", "t_conj_C", "t_conj_h"],
         note="SAME antibody also appears at 8.0 eq -> DAR 8.0 (scheme B), so this "
              "is one half of a same-antibody dose curve"),

    dict(id="CStone-002-DXd", antibody="CStone 002 (anti-ITGB4)", isotype="IgG1 humanised",
         payload="mc-GGGG-DXd", tcep_eq=8.0, t_red_C=25, t_red_h=4.0,
         ph=None, mab_mg_mL=None, payload_eq=12.0, t_conj_C=20, t_conj_h=0.5,
         dar=8.0, dar_method="HIC / RP-HPLC",
         source="WO2024/208354 A1 (scheme B)", usable=False,
         missing=["ph", "mab_mg_mL"],
         note="other half of the CStone 002 dose curve"),

    dict(id="CStone-008-MMAE", antibody="CStone 008 (anti-ITGB4)", isotype="IgG1 humanised",
         payload="mc-VC-PABC-MMAE", tcep_eq=2.35, t_red_C=37, t_red_h=1.0,
         ph=None, mab_mg_mL=None, payload_eq=5.5, t_conj_C=None, t_conj_h=None,
         dar=3.9, dar_method="HIC / RP-HPLC",
         source="WO2024/208354 A1, Table 6 (scheme A)", usable=False,
         missing=["ph", "mab_mg_mL", "t_conj_C", "t_conj_h"]),

    dict(id="CStone-008-DXd", antibody="CStone 008 (anti-ITGB4)", isotype="IgG1 humanised",
         payload="mc-GGGG-DXd", tcep_eq=8.0, t_red_C=25, t_red_h=4.0,
         ph=None, mab_mg_mL=None, payload_eq=12.0, t_conj_C=20, t_conj_h=0.5,
         dar=8.0, dar_method="HIC / RP-HPLC",
         source="WO2024/208354 A1 (scheme B)", usable=False,
         missing=["ph", "mab_mg_mL"]),

    dict(id="CStone-005-MMAE", antibody="CStone 005 (anti-ITGB4)", isotype="IgG1 humanised",
         payload="mc-VC-PABC-MMAE", tcep_eq=2.35, t_red_C=37, t_red_h=1.0,
         ph=None, mab_mg_mL=None, payload_eq=5.5, t_conj_C=None, t_conj_h=None,
         dar=3.2, dar_method="HIC / RP-HPLC",
         source="WO2024/208354 A1, Table 6", usable=False,
         missing=["ph", "mab_mg_mL", "t_conj_C", "t_conj_h"],
         note="LOWEST DAR of the MMAE group at the same TCEP - a failure-ish point"),

    dict(id="CStone-001-MMAE", antibody="CStone 001 (anti-ITGB4)", isotype="IgG1 humanised",
         payload="mc-VC-PABC-MMAE", tcep_eq=2.35, t_red_C=37, t_red_h=1.0,
         ph=None, mab_mg_mL=None, payload_eq=5.5, t_conj_C=None, t_conj_h=None,
         dar=3.5, dar_method="HIC / RP-HPLC",
         source="WO2024/208354 A1, Table 6", usable=False,
         missing=["ph", "mab_mg_mL", "t_conj_C", "t_conj_h"]),

    dict(id="CStone-004-MMAE", antibody="CStone 004 (anti-ITGB4)", isotype="IgG1 humanised",
         payload="mc-VC-PABC-MMAE", tcep_eq=2.35, t_red_C=37, t_red_h=1.0,
         ph=None, mab_mg_mL=None, payload_eq=5.5, t_conj_C=None, t_conj_h=None,
         dar=3.5, dar_method="HIC / RP-HPLC",
         source="WO2024/208354 A1, Table 6", usable=False,
         missing=["ph", "mab_mg_mL", "t_conj_C", "t_conj_h"]),

    dict(id="CStone-006-MMAE", antibody="CStone 006 (anti-ITGB4)", isotype="IgG1 humanised",
         payload="mc-VC-PABC-MMAE", tcep_eq=2.35, t_red_C=37, t_red_h=1.0,
         ph=None, mab_mg_mL=None, payload_eq=5.5, t_conj_C=None, t_conj_h=None,
         dar=3.5, dar_method="HIC / RP-HPLC",
         source="WO2024/208354 A1, Table 6", usable=False,
         missing=["ph", "mab_mg_mL", "t_conj_C", "t_conj_h"]),

    dict(id="CStone-002-SN38", antibody="CStone 002 (anti-ITGB4)", isotype="IgG1 humanised",
         payload="mc-VC-PAB-DMEA-SN38", tcep_eq=8.0, t_red_C=25, t_red_h=4.0,
         ph=None, mab_mg_mL=None, payload_eq=12.0, t_conj_C=20, t_conj_h=0.5,
         dar=7.8, dar_method="HIC / RP-HPLC",
         source="WO2024/208354 A1 (scheme B)", usable=False,
         missing=["ph", "mab_mg_mL"],
         note="same antibody, third payload - useful for the payload-dependence question"),

    dict(id="trastuzumab-US10266606", antibody="trastuzumab", isotype="IgG1 humanised",
         payload="mc-VC-PAB-MMAE", tcep_eq=1.15, t_red_C=25, t_red_h=1.0,
         ph=6.0, mab_mg_mL=None, payload_eq=2.2, t_conj_C=25, t_conj_h=16.0,
         dar=1.75, dar_method="HIC",
         source="US10266606 B2 (purification example)", usable=False,
         missing=["mab_mg_mL"],
         note="low DAR at low TCEP - a mild failure point"),
]

ALL = USABLE + INCOMPLETE


if __name__ == "__main__":
    n_ok = sum(1 for r in ALL if r.get("usable"))
    print(f"{len(ALL)} rows total, {n_ok} fully specified, {len(ALL) - n_ok} missing fields")
    print("\nfully specified:")
    for r in USABLE:
        print(f"  {r['id']:28s} TCEP {r['tcep_eq']:>5} eq  {r['t_red_C']:>4} C "
              f"{r['t_red_h']:>4} h  {r['mab_mg_mL']:>5} mg/mL  DAR {r['dar']}")
    print("\nmissing fields (count by field):")
    from collections import Counter
    c = Counter(f for r in INCOMPLETE for f in r.get("missing", []))
    for k, v in c.most_common():
        print(f"  {k:16s} {v}")
