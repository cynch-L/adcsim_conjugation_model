from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a pipeline config and resolve its input/output paths to absolute.

    WHICH CONFIG IS THIS?  There are two config families in this repo and they
    are NOT interchangeable — this one serves the *structural* pipeline
    (pipeline.py, run via `python -m adcsim config.yaml`), whose top-level keys
    are antibody / inputs / outputs / disulfide / sites / chemistry / sampling /
    qc / accessibility / search.  See examples/trop_adc/config.yaml.

    The other family is adc_model_config.yaml (used by dar_v6_complete.py,
    payload_qc.py, site_env.py): it describes *process chemistry* (pH, TCEP
    equivalents, k2, anchors) and has none of the keys above.  Feeding it here
    raises KeyError by design — use the right one.
    """
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f)
    root = path.parent
    for key in ("antibody_pdb", "lp_sdf"):
        p = Path(cfg["inputs"][key])
        if not p.is_absolute():
            cfg["inputs"][key] = str((root / p).resolve())
    out = Path(cfg["outputs"]["dir"])
    if not out.is_absolute():
        cfg["outputs"]["dir"] = str((root / out).resolve())
    return cfg
