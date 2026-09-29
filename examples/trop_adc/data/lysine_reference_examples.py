"""Public reference data for LYSINE-conjugated ADCs (2026-09-27).

Why this file exists
--------------------
The model so far only covers cysteine conjugation (maleimide + thiol). Extending
it to lysine conjugation needs two things the cysteine route never required:

  1. a process condition set that is fully specified (pH, equivalents, solvent,
     time, antibody concentration), so the model can be run zero-prior;
  2. a measured list of WHICH lysines actually react, because lysine sites
     cannot be derived from the sequence the way interchain disulfides can.

Both are public for trastuzumab emtansine — and that is the same antibody the
model already uses, so it is a same-antibody control, not an analogy.

Honesty rules used here (same as patent_medilink_examples.py)
------------------------------------------------------------
* every number carries its source;
* nothing is invented. Where the source reports a count but not the full list
  (e.g. "44 sites identified"), the count is recorded and the missing detail is
  marked as needing the original figure — it is NOT filled in with guesses;
* tier B = measured and traceable, tier C = reported but not independently
  verifiable here.

Sources
-------
[PMC6698837] "In-Depth Comparison of Lysine-Based Antibody-Drug Conjugates
  Prepared on Solid Support Versus in Solution" — protocol, DAR distribution,
  peptide-map conjugation sites.
[SCIEX]      SCIEX technical note, intact analysis of T-DM1 (BioPharmaView).
[Sterling]   Sterling Pharma AN-005, intact mass of Kadcyla on X500B.
[FDA-MYLOTARG] accessdata.fda.gov Mylotarg label (2000) — 50% unconjugated,
  DAR 0-6, mean 2-3.
"""

from __future__ import annotations

# =============================================================================
# 1. T-DM1 / Kadcyla — the anchor case (trastuzumab, IgG1, lysine)
# =============================================================================
TDM1 = {
    "name": "trastuzumab emtansine (T-DM1, Kadcyla)",
    "antibody": "trastuzumab",
    "isotype": "IgG1 kappa",
    "conjugation_chemistry": {
        "reactive_group": "NHS ester",
        "target_residue": "lysine side-chain amine (and N-terminus)",
        "linker_payload": "SMCC-DM1",
        "mass_shift_per_payload_Da": 956.0,     # PMC6698837, intact MS
        "note": ("Two-step in the marketed process (antibody + SMCC, then DM1); "
                 "the on-bead/off-bead study used a one-step SMCC-DM1 addition, "
                 "which is why it shows no free-linker species."),
    },
    # --- Fully specified protocol (PMC6698837, off-bead / in-solution arm) ----
    # This is the part that lets the model be run with zero fitted parameters.
    "protocol": {
        "antibody_conc_mg_mL": 1.0,
        "buffer": "10 mM sodium bicarbonate",
        "ph": 8.5,
        "payload": "SMCC-DM1, 20 mM stock in DMA",
        "feed_eq": 23.0,                        # mol SMCC-DM1 per mol antibody
        "time_h": 1.0,
        "temperature_C": None,                  # reported as "room temperature"
        "temperature_note": "reported as room temperature; no exact value given",
        "source": "PMC6698837",
        "tier": "B",
    },
    # --- Measured outcome ----------------------------------------------------
    "dar": {
        "mean_measured": 3.5,
        "mean_replicates": {
            "SCIEX_intact": 3.49,
            "Sterling_AN005": 3.43,
            "literature_review": 3.5,
        },
        "distribution_range": [0, 8],           # intact MS, SCIEX
        "distribution_source": "SCIEX intact analysis; Sterling AN-005 (Endo-S treated)",
        "unconjugated_fraction": 0.04,
        "unconjugated_note": ("~4% unconjugated antibody, measured on the ONE-STEP "
                              "on-bead/off-bead samples, not on the marketed two-step "
                              "Kadcyla process. Treat as process-specific, not as a "
                              "property of lysine conjugation."),
        "tier": "B",
    },
    # --- Which lysines react -------------------------------------------------
    # The number the model cannot compute from sequence, and the reason a
    # peptide map is mandatory for the lysine route.
    "sites": {
        "n_sites_identified": 44,
        "n_sites_note": ("44 lysine conjugation sites identified by tryptic peptide "
                         "mapping LC-MS/MS across on-bead and off-bead samples. "
                         "Only the sites the paper singles out are listed below; "
                         "the per-site relative abundances are in Figure 3B of the "
                         "paper and are NOT reproduced here because they were not "
                         "readable from the retrieved text."),
        "n_lysines_in_antibody": None,
        "n_lysines_note": ("total lysine count of trastuzumab not stated in the "
                           "source; count it from the sequence with seqqc when the "
                           "lysine route is implemented (order of magnitude ~80)"),
        "named_sites": {
            "H65":  "high abundance (>=0.5% of population); sits within 10 A of the "
                    "Protein A binding site; its conjugation drops ~40% on-bead, "
                    "which the authors read as steric hindrance",
            "H330": "differed significantly (p<=0.05) between on-bead and off-bead",
            "H396": "seen only in the off-bead sample",
            "H76":  "seen only in the on-bead sample",
            "H136": "seen only in the on-bead sample",
            "H342": "seen only in the on-bead sample; also within 10 A of the "
                    "Protein A site but <0.5% abundance",
            "H252": "within 10 A of the Protein A binding site, low abundance",
            "H321": "within 10 A of the Protein A binding site, low abundance",
        },
        "site_naming": "H = heavy chain, L = light chain (the paper uses this prefix)",
        "heterogeneity_note": ("several sites appear in only 1 of 3 replicates, which "
                               "the authors cite as evidence of high sample "
                               "heterogeneity. Any per-site model must reproduce "
                               "that scatter, not a smooth ranking."),
        "source": "PMC6698837",
        "tier": "B",
    },
    # --- What this gives the model ------------------------------------------
    "use_as": [
        "zero-prior test: run the lysine branch on the protocol above and compare "
        "with mean DAR 3.5",
        "site-prior seed: the named sites are the beginning of an IgG1 lysine "
        "reactivity library (the same idea as the 8 cysteine sites)",
        "distribution check: the model must reproduce a 0-8 spread, not a point",
    ],
    "sources": ["PMC6698837", "SCIEX", "Sterling", "literature review PMC8919033"],
}

# =============================================================================
# 2. Mylotarg — the heterogeneity extreme (IgG4, lysine)
# =============================================================================
MYLOTARG = {
    "name": "gemtuzumab ozogamicin (Mylotarg)",
    "antibody": "hP67.6, anti-CD33",
    "isotype": "IgG4 kappa",
    "isotype_warning": ("IgG4, NOT IgG1. The current model is calibrated for IgG1 "
                        "only, so this cannot be used to validate it directly. It "
                        "is recorded as the shape lysine conjugation can take."),
    "conjugation_chemistry": {
        "reactive_group": "activated ester (bifunctional linker)",
        "target_residue": "lysine",
        "payload": "N-acetyl gamma calicheamicin",
    },
    "dar": {
        "mean_measured": None,
        "mean_range": [2.0, 3.0],               # FDA label: 2 to 3 mol per mol
        "distribution_range": [0, 6],
        "unconjugated_fraction": 0.50,
        "label_quote": ("approximately 50% of the antibody is loaded with 4-6 moles "
                        "calicheamicin per mole of antibody. The remaining 50% of the "
                        "antibody is not linked to the calicheamicin derivative."),
        "note": ("a bimodal product: half the molecules carry nothing and the loaded "
                 "half carries 4-6. Any lysine model that only outputs a mean has "
                 "failed here."),
        "source": "FDA Mylotarg label (accessdata.fda.gov, 2000)",
        "tier": "B",
    },
    "use_as": [
        "shape check: the distribution must be able to put a large mass at DAR 0",
    ],
}

# =============================================================================
# 3. NOT recorded on purpose
# =============================================================================
EXCLUDED = {
    "inotuzumab ozogamicin (Besponsa)": ("lysine conjugated, IgG4, but no DAR figure "
                                         "with a traceable source was found in this "
                                         "session -> deliberately not recorded"),
}

# =============================================================================
# 4. What is still missing before the lysine branch can be calibrated
# =============================================================================
MISSING = [
    "the full per-site abundance table (PMC6698837 Figure 3B) — needed to know the "
    "shape of the lysine reactivity distribution, not just its mean",
    "an apparent second-order rate constant for NHS-ester + amine at a stated pH "
    "(the cysteine route has k2 = 300 M^-1 s^-1; the lysine route has nothing yet)",
    "an NHS-ester hydrolysis rate at the process pH — the cysteine route sets "
    "k_hyd = 0 because maleimide hydrolysis is negligible there; for an activated "
    "ester that assumption is false and the number must come from measurement",
    "the N-terminus must be carried as a separate residue class (pKa ~8.0) rather "
    "than pooled with lysine (pKa ~10.5)",
]

# --- convenience access -------------------------------------------------------
ALL = {"T-DM1": TDM1, "Mylotarg": MYLOTARG}


if __name__ == "__main__":
    import json
    print(json.dumps(ALL, indent=2, ensure_ascii=False))
