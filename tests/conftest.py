"""Shared fixtures and paths for the adcsim test suite.

The package lives under ``packages/`` (not installed), so ``sys.path`` is set up
here instead of relying on an editable install.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT))

# ---- Canonical inputs / outputs shared by the whole suite ----
PDB_PATH = ROOT / "examples/trop_adc/data/trastuzumab_boltz_model_0.pdb"
DAR_JSON = ROOT / "examples/trop_adc/results_hinge/dar_v6_complete.json"
CONFIG_YAML = ROOT / "examples/trop_adc/kaggle_md_8site/adc_model_config.yaml"
MAIN_SCRIPT = ROOT / "examples/trop_adc/kaggle_md_8site/dar_v6_complete.py"
SS_PROFILE = ROOT / "examples/trop_adc/results_hinge/ss_reduction_profile.json"

# Values locked by the current calibration. If these move, the model changed
# and the change must be deliberate -- not a side effect of refactoring.
LOCKED = {
    "K_open": 1.1869,
    "mean_dar": 4.587,
    "mode_dar": 4,
    "even_odd_ratio": 43.346,
    "loo_mae": 0.0547,
    "n_sites": 8,
    "n_bond": 4,
}


@pytest.fixture(scope="session")
def pdb_path():
    if not PDB_PATH.exists():
        pytest.skip(f"structure not present: {PDB_PATH}")
    return str(PDB_PATH)


@pytest.fixture(scope="session")
def dar_result():
    if not DAR_JSON.exists():
        pytest.skip(f"model output not present: {DAR_JSON}")
    return json.loads(DAR_JSON.read_text())


@pytest.fixture(scope="session")
def seqqc_result(pdb_path):
    from adcsim import seqqc

    return seqqc.assess(pdb_path)


@pytest.fixture(scope="session")
def model():
    """The main model script, loaded by path.

    It is a runnable script rather than an importable module (it computes at
    import time and lives outside ``packages/``), so tests reach it by path.
    """
    if not MAIN_SCRIPT.exists():
        pytest.skip(f"model script not present: {MAIN_SCRIPT}")
    spec = importlib.util.spec_from_file_location("dar_v6_complete", MAIN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def ease():
    """Per-disulfide reduction ease, as read off the structure."""
    if not SS_PROFILE.exists():
        pytest.skip(f"reduction profile not present: {SS_PROFILE}")
    prof = json.loads(SS_PROFILE.read_text())
    return {p["site_a"] + "--" + p["site_b"]: p["reduction_ease"] for p in prof["pairs"]}


@pytest.fixture(scope="session")
def pc(dar_result):
    """Per-site conjugation probability, taken from the model's own output."""
    return {s: v["p_c"] for s, v in dar_result["sites"].items()}
