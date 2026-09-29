"""adcsim — computational pre-screen for cysteine-conjugated ADCs.

The model answers one question — *given this antibody, this linker-payload and
these process conditions, what DAR distribution comes out?* — by splitting it
into six links.  Each link is answered with its own evidence and its own
module; nothing is allowed to borrow conclusions across links.  Refer to them
by these names — no numbering, no metaphors.

LINK → MODULE MAP (this is also the notebook reading order)

    link                           module                 entry point
    -----------------------------  ---------------------  -----------------------
    sequence QC                    seqqc.py               run / report

    disulfide reduction            (dar_v6_complete.py)   pred_shared / fit_K
    site accessibility             accessibility.py       score_site
                                   sites.py               detect_disulfides
    thiol activation               site_env.py            assess
    covalent conjugation           chemistry (config)     k2_ref in yaml
    combinatorial statistics       (dar_v6_complete.py)   Poisson-binomial

SUPPORTING MODULES

    payload_qc.py   linker-payload QC: is the reactive end exposed, and does
                    solubility cap the working concentration?
    alarms.py       threshold checks + the shared alarm record schema
                    (make_alarm) used by every other module.
    release.py      payload release kinetics (cathepsin B) — is release rate-
                    limiting?  (answer: no)
    bystander.py    bystander effect (Krogh cylinder) — peripheral module,
                    deliberately NOT merged into the main model.
    io.py           PDB/site-spec parsing shared by the structural modules.
    geometry.py     shared geometry + molecular-graph helpers.
    solvation.py    the single FreeSASA entry point (solvation.sasa).

TWO CONFIG FAMILIES (a common trip-up)

    config.yaml             → the structural pipeline (pipeline.py)
    adc_model_config.yaml   → the process model (dar_v6, payload_qc, site_env)

They are not interchangeable; see config.load_config.
"""

from .config import load_config

__version__ = "0.1.0"
__all__ = ["load_config", "__version__"]
