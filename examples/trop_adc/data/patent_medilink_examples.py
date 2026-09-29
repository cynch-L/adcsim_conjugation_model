#!/usr/bin/env python
"""HER2-ADC-005..015 + single-payload controls + B7H3 examples.

Source: MediLink WO2024235127A1 (PCT/CN2024/092419)
  - Examples 2.5-2.18 : spec pages 141-149 (PDF pages 143-151)
  - Example 3.1/3.2   : spec page 151 (PDF page 153)
  - Table 9 kinetics  : spec page 155 (PDF page 157)
OCR: pypdfium2 render @2x + macOS Vision (zh-Hans, en-US), 2026-09-27.
Digits below are transcribed from the OCR text; ratios flagged possible OCR
confusion are marked with * in note.
"""

# Each row: (id, mAb_mg, vol_mL, pH, tcep_eq, T_red_C, t_red_min,
#            payload_trial_A, eq_A, payload_B, eq_B, t_conj_h, T_conj_C,
#            mode, dar_total, n_A, n_B, spec_page)
EXAMPLES = [
    # ---- Example 2.x : dual payload, simultaneous unless noted ----
    dict(id="HER2-ADC-001", mab_mg=12.55, vol_mL=None, ph=7.50, tcep_eq=5.5,
         T_red_C=25, t_red_min=90, A="DL003", eq_A=6.0, B="DL001", eq_B=3.0,
         t_conj_h=2, T_conj_C=25, mode="simultaneous",
         dar_total=7.6, n_A=6.00, n_B=1.62, spec_page=138,
         ms=dict(LC=[("DAR0", 18.2), ("DL003", 55.3), ("DL001", 26.4)],
                 HC=[("3*DL003", 70.0), ("2*DL003+DL001", 16.2),
                     ("DL003+2*DL001", 2.5), ("3*DL001", 11.2),
                     ("DAR0", 0.0), ("?", 0.0)])),
    dict(id="HER2-ADC-002a", mab_mg=6.28, vol_mL=None, ph=7.45, tcep_eq=5.5,
         T_red_C=30, t_red_min=90, A="DL003", eq_A=5.5, B="DL007", eq_B=3.5,
         t_conj_h=3, T_conj_C=30, mode="simultaneous",
         dar_total=7.7, n_A=5.1, n_B=2.6, spec_page=139,
         ms=dict(LC=[("DAR0", 13.3), ("DL003", 47.6), ("DL007", 39.1)],
                 HC=[("2*DL003+DL007", 22.2), ("DL003+2*DL007", 1.2),
                     ("3*DL003", 54.2), ("3*DL007", 22.5)])),
    dict(id="HER2-ADC-002b", mab_mg=6.28, vol_mL=None, ph=7.45, tcep_eq=5.5,
         T_red_C=30, t_red_min=90, A="DL003", eq_A=5.5, B="DL007", eq_B=3.5,
         t_conj_h=3, T_conj_C=30, mode="sequential", stage1_h=1, stage2_h=2,
         dar_total=7.8, n_A=5.2, n_B=2.6, spec_page=140,
         ms=dict(LC=[("DAR0", 12.0), ("DL003", 50.2), ("DL007", 37.8)],
                 HC=[("2*DL003+DL007", 5.0), ("DL003+2*DL007", 0.0),
                     ("3*DL003", 66.0), ("3*DL007", 29.0)])),
    dict(id="HER2-ADC-003", mab_mg=25.1, vol_mL=None, ph=7.47, tcep_eq=5.5,
         T_red_C=30, t_red_min=90, A="DL003", eq_A=6.2, B="DL007", eq_B=3.0,
         t_conj_h=3, T_conj_C=30, mode="sequential", stage1_h=1, stage2_h=2,
         dar_total=7.8, n_A=5.9, n_B=1.9, spec_page=140,
         ms=dict(LC=[("DAR0", 7.6), ("DL003", 56.9), ("DL007", 35.5)],
                 HC=[("2*DL003+DL007", 14.9), ("DL003+2*DL007", 8.2),
                     ("3*DL003", 67.4), ("3*DL007", 9.5)])),
    dict(id="HER2-ADC-004", mab_mg=14.0, vol_mL=None, ph=7.47, tcep_eq=5.5,
         T_red_C=25, t_red_min=90, A="DL002", eq_A=4.5, B="DL007", eq_B=4.2,
         t_conj_h=2, T_conj_C=25, mode="simultaneous",
         dar_total=7.8, n_A=4.3, n_B=3.5, spec_page=141,
         note="n_A here is DL002, n_B is DL007 (A/B labels follow the feeds)",
         ms=dict(LC=[("DAR0", 8.4), ("DL002", 47.7), ("DL007", 43.9)],
                 HC=[("2*DL002+DL007", 34.2), ("DL002+2*DL007", 28.5),
                     ("3*DL002", 22.0), ("3*DL007", 13.6),
                     ("DAR2-2*DL002", 0.5), ("DAR2-2*DL007", 0.5),
                     ("DAR2-DL002+DL007", 0.6)])),
    dict(id="HER2-ADC-005", mab_mg=12.35, vol_mL=0.5, ph=7.56, tcep_eq=5.5,
         T_red_C=25, t_red_min=90, A="DL003", eq_A=6.0, B="DL012", eq_B=5.0,
         t_conj_h=2, T_conj_C=25, mode="simultaneous",
         dar_total=7.9, n_A=6.38, n_B=1.54, spec_page=141,
         ms=dict(LC=[("DAR0", 3.8), ("DL003", 65.7), ("DL012", 30.6)],
                 HC=[("2*DL003+DL012", 24.5), ("DL003+2*DL012", 4.8),
                     ("3*DL003", 66.5), ("3*DL012", 4.2)])),
    dict(id="HER2-ADC-006", mab_mg=12.35, vol_mL=0.5, ph=7.56, tcep_eq=5.5,
         T_red_C=25, t_red_min=90, A="DL003", eq_A=7.0, B="DL012", eq_B=4.0,
         t_conj_h=2, T_conj_C=25, mode="simultaneous",
         dar_total=7.9, n_A=6.99, n_B=0.96, spec_page=142,
         ms=dict(LC=[("DAR0", 2.7), ("DL003", 77.7), ("DL012", 19.5)],
                 HC=[("2*DL003+DL012", 23.2), ("DL003+2*DL012", 2.6),
                     ("3*DL003", 74.3), ("3*DL012", 0.0)])),
    dict(id="HER2-ADC-007", mab_mg=14.82, vol_mL=0.6, ph=7.44, tcep_eq=5.5,
         T_red_C=25, t_red_min=90, A="DL003", eq_A=5.6, B="DL012", eq_B=2.0,
         t_conj_h=2, T_conj_C=25, mode="simultaneous",
         dar_total=7.8, n_A=5.85, n_B=1.93, spec_page=142,
         ms=dict(LC=[("DAR0", 11.3), ("DL003", 59.3), ("DL012", 29.4)],
                 HC=[("2*DL003+DL012", 16.8), ("DL003+2*DL012", 2.0),
                     ("3*DL003", 65.8), ("3*DL012", 15.4)])),
    dict(id="HER2-ADC-008", mab_mg=17.29, vol_mL=0.7, ph=7.37, tcep_eq=5.5,
         T_red_C=25, t_red_min=90, A="DL003", eq_A=4.7, B="DL012", eq_B=4.5,
         t_conj_h=2, T_conj_C=25, mode="simultaneous",
         dar_total=8.0, n_A=4.75, n_B=3.25, spec_page=143,
         note="patent gives the masses: DL003 0.073 mg, DL012 0.087 mg",
         ms=dict(LC=[("DAR0", 0.0), ("DL003", 36.0), ("DL012", 64.0)],
                 HC=[("2*DL003+DL012", 23.9), ("DL003+2*DL012", 7.4),
                     ("3*DL003", 48.8), ("3*DL012", 19.9)])),
    dict(id="HER2-ADC-009", mab_mg=17.29, vol_mL=0.7, ph=7.37, tcep_eq=5.5,
         T_red_C=25, t_red_min=90, A="DL003", eq_A=5.4, B="DL012", eq_B=4.0,
         t_conj_h=2, T_conj_C=25, mode="simultaneous",
         dar_total=7.9, n_A=5.41, n_B=2.54, spec_page=144,
         note="patent gives the masses: DL003 0.084 mg, DL012 0.077 mg",
         ms=dict(LC=[("DAR0", 2.5), ("DL003", 48.4), ("DL012", 49.1)],
                 HC=[("2*DL003+DL012", 21.5), ("DL003+2*DL012", 7.8),
                     ("3*DL003", 57.1), ("3*DL012", 13.6)])),
    dict(id="HER2-ADC-010", mab_mg=19.76, vol_mL=0.8, ph=7.43, tcep_eq=5.5,
         T_red_C=25, t_red_min=90, A="DL003", eq_A=6.0, B="DL014", eq_B=4.0,
         t_conj_h=2, T_conj_C=25, mode="simultaneous",
         dar_total=8.0, n_A=5.8, n_B=2.2, spec_page=145,
         ms=dict(LC=[("DAR0", 1.0), ("DL003", 53.9), ("DL014", 45.1)],
                 HC=[("2*DL003+DL014", 11.9), ("DL003+2*DL014", 2.6),
                     ("3*DL003", 70.0), ("3*DL014", 15.6)])),
    dict(id="HER2-ADC-011", mab_mg=12.35, vol_mL=0.5, ph=7.50, tcep_eq=5.5,
         T_red_C=25, t_red_min=90, A="DL003", eq_A=3.0, B="DL014", eq_B=1.6,
         t_conj_h=2, T_conj_C=25, mode="simultaneous",
         dar_total=3.9, n_A=3.0, n_B=0.9, spec_page=145, note="dar_total not stated"),
    dict(id="HER2-ADC-012", mab_mg=12.35, vol_mL=0.5, ph=7.50, tcep_eq=5.5,
         T_red_C=25, t_red_min=90, A="DL003", eq_A=4.0, B="DL014", eq_B=1.6,
         t_conj_h=2, T_conj_C=25, mode="simultaneous",
         dar_total=5.4, n_A=4.1, n_B=1.3, spec_page=146, note="dar_total not stated"),
    dict(id="HER2-ADC-013", mab_mg=12.35, vol_mL=0.5, ph=7.50, tcep_eq=5.5,
         T_red_C=25, t_red_min=90, A="DL003", eq_A=6.1, B="DL012", eq_B=2.5,
         t_conj_h=2, T_conj_C=25, mode="simultaneous",
         dar_total=5.6, n_A=4.7, n_B=0.9, spec_page=146, note="dar_total not stated"),
    dict(id="HER2-ADC-014", mab_mg=11.52, vol_mL=0.4, ph=7.45, tcep_eq=5.75,
         T_red_C=25, t_red_min=90, A="DL003", eq_A=6.8, B="DL017", eq_B=3.0,
         t_conj_h=6, T_conj_C=25, mode="sequential", stage1_h=2, stage2_h=4,
         dar_total=8.0, n_A=7.2, n_B=0.8, spec_page=147),
    dict(id="HER2-ADC-015", mab_mg=11.52, vol_mL=0.4, ph=7.45, tcep_eq=5.75,
         T_red_C=25, t_red_min=90, A="DL003", eq_A=6.2, B="DL017", eq_B=3.0,
         t_conj_h=6, T_conj_C=25, mode="sequential", stage1_h=2, stage2_h=4,
         dar_total=8.0, n_A=7.0, n_B=1.0, spec_page=147),
    # ---- B7H3 antibody (different mAb, same platform) ----
    dict(id="B7H3-ADC-001", mab_mg=342.71, vol_mL=None, ph=7.50, tcep_eq=5.5,
         T_red_C=25, t_red_min=90, A="DL003", eq_A=6.0, B="DL014", eq_B=2.05,
         t_conj_h=4, T_conj_C=25, mode="simultaneous",
         dar_total=7.78, n_A=5.96, n_B=1.82, spec_page=147,
         note="mAb stock 17.2 mg/mL; combined payload stock mass 0.680 g"),
    dict(id="B7H3-ADC-002", mab_mg=345.01, vol_mL=None, ph=7.50, tcep_eq=5.5,
         T_red_C=25, t_red_min=90, A="DL003", eq_A=6.25, B="DL014", eq_B=2.14,
         t_conj_h=4, T_conj_C=25, mode="simultaneous",
         dar_total=7.98, n_A=6.21, n_B=1.77, spec_page=148, note="stock mass 0.714 g"),
    dict(id="B7H3-ADC-003", mab_mg=339.44, vol_mL=None, ph=7.50, tcep_eq=5.5,
         T_red_C=25, t_red_min=120, A="DL003", eq_A=6.03, B="DL014", eq_B=3.0,
         t_conj_h=4, T_conj_C=25, mode="sequential", stage1_h=1, stage2_h=3,
         dar_total=8.0, n_A=6.01, n_B=1.99, spec_page=148),
]

# ---- Single-payload controls: the most valuable rows in the whole patent ----
SINGLE = [
    dict(id="HER2-DL012-DAR2", mab_mg=9.88, vol_mL=0.4, ph=7.45, tcep_eq=5.5,
         T_red_C=25, t_red_min=90, payload="DL012", eq=5.0, mass_mg=0.055,
         t_conj_h=2, T_conj_C=25, dar=1.8, spec_page=151),
    dict(id="HER2-DL003-DAR8", mab_mg=16.1, vol_mL=5.0, ph=7.46, tcep_eq=None,
         T_red_C=None, t_red_min=90, payload="DL003", eq=10.0, mass_mg=1.332,
         t_conj_h=2, T_conj_C=None, dar=8.0, spec_page=151,
         note="5.0 mL stock diluted with 0.10 mL 20 mM PB + 100 mM EDTA pH 7.6; "
              "1.0 mL taken, TCEP 0.0278 mL of 20 mM (0.556 umol); RT reduction"),
]

# ---- Table 9: conjugation kinetics of B7H3-ADC-001 ----
KINETICS = [
    dict(t="10min", dar_total=6.59, dar_DL003=6.06, dar_DL014=0.53),
    dict(t="30min", dar_total=7.15, dar_DL003=6.02, dar_DL014=1.5),
    dict(t="1h",    dar_total=7.53, dar_DL003=6.05, dar_DL014=1.48),
    dict(t="2h",    dar_total=7.70, dar_DL003=5.95, dar_DL014=1.76),
    dict(t="3h",    dar_total=7.77, dar_DL003=6.02, dar_DL014=3.75),
    dict(t="4h",    dar_total=7.78, dar_DL003=5.96, dar_DL014=1.82),
]
KINETICS_NOTE = (
    "OCR of the 3h DL014 cell reads 3.75, which breaks monotonicity against both "
    "the 2h (1.76) and 4h (1.82) rows; most likely a mis-OCR of 1.75. Flagged, "
    "do not use that row for fitting without re-checking the scan."
)
KINETICS_TEXT = (
    "DL003 reaches DAR6.0 within ~10 min, utilisation close to 100%; the second "
    "payload levels off after ~2 h. The patent concludes from this that the "
    "drug-linker RATIO can be set by the feed ratio alone."
)

MAB_MW = 150000.0       # g/mol, IgG1 trastuzumab glycosylated average
PAYLOAD_STOCK_MM = 10.0  # mmol/L in DMSO, as stated for every example
TCEP_STOCK_MM = 20.0     # mmol/L in water


def mab_uM(row):
    """mAb concentration in uM from mg + volume."""
    if not row.get("vol_mL"):
        return None
    return row["mab_mg"] * 1e-3 / MAB_MW / (row["vol_mL"] * 1e-3) * 1e6


def dmso_vv(row):
    """DMSO volume fraction actually delivered, assuming a single step.

    Every payload is fed as a 10 mmol/L DMSO stock. Total equivalents of
    payload stock -> DMSO volume as a fraction of the aqueous reaction volume.
    """
    eq = (row.get("eq_A") or 0.0) + (row.get("eq_B") or 0.0)
    if row.get("eq") is not None:          # single-payload rows
        eq = row["eq"]
    # uL stock per mL of reaction volume
    return eq * PAYLOAD_STOCK_MM * -1 if False else eq / (PAYLOAD_STOCK_MM * 1e3) * 1e0
