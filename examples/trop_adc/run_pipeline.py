#!/usr/bin/env python3
"""Run the Trastuzumab example with flushes between stages."""
from pathlib import Path
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
os.chdir(Path(__file__).resolve().parent)

from adcsim.pipeline import run  # noqa: E402

if not list(Path("data").glob("*.pdb")):
    sys.exit("data/ is empty — copy or symlink the inputs first")

# Do not touch results/ (Boltz baseline). Hinge run writes results_hinge/.
run("config.yaml")
print("FLUSH: pipeline done")
report = Path("results_hinge") / "QC_REPORT.md"
if report.exists():
    shutil.copy2(report, Path("../..") / "adcsim_QC_report_hinge.md")
    print("copied report -> ../../adcsim_QC_report_hinge.md")
