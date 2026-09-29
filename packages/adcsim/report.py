"""Aggregated alarm-summary entry point for the ADC conjugation pre-screen.

This module is the single "alarm summary hub" that the rest of the project
never wired up: every QC layer already emits its own alarms through
``alarms.make_alarm`` (sequence QC, site environment / thiol activation,
linker-payload QC), and ``alarms.check_all`` already implements ~15 process
checks, but ``check_all`` had no caller.  ``report.py`` plugs that gap.

What it does
------------
1. Collects the three layer alarms (sequence QC / thiol activation / linker-payload QC).
2. Builds the ``state`` dict that ``alarms.check_all`` expects **only from fields
   that actually exist** in the main model output JSON, then runs ``check_all``.
3. Merges everything into one list, groups it by process step, and renders a
   human-readable Markdown report plus a machine-readable JSON dump.

Vocabulary: every alarm must carry the bare name of the step it belongs to.
No numbering, no metaphors — write out what the step answers.
-----------------------------------------------------------------------------
    sequence QC
    disulfide reduction
    site accessibility
    thiol activation
    covalent conjugation
    combinatorial statistics
    linker-payload QC

Design notes for notebook migration
-----------------------------------
Every ``# [N]`` comment marks a self-contained block that maps to one notebook
cell, so this file can be lifted into a Jupyter notebook with minimal edits.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# [1] Imports & helpers (kept from the original report.py scaffolding so that
#     pipeline.py / polish.py keep working).
from .alarms import check_all, summarize_alarms, OK, WARN, CRIT, INFO
from . import seqqc, site_env, payload_qc


def write_markdown(path: str, title: str, sections: list[tuple[str, str]]) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", f"_Generated {datetime.now().isoformat(timespec='seconds')}_", ""]
    for heading, body in sections:
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(body.rstrip())
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n")
    return path


def table(headers: list[str], rows: list[list]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join([head, sep, *body]) if rows else head + "\n" + sep + "\n| _(empty)_ |"


# [2] Canonical step vocabulary -------------------------------------------------

# Order used for grouping and rendering. One entry per link of the model.
STEP_ORDER = [
    "sequence QC",
    "disulfide reduction",
    "site accessibility",
    "thiol activation",
    "covalent conjugation",
    "combinatorial statistics",
    "linker-payload QC",
]

# The linker-payload QC layer emits ``step == "LP-QC"``; normalize it to the
# canonical name so the whole report speaks one vocabulary.
_STEP_ALIAS = {
    "LP-QC": "linker-payload QC",
}


def _normalize(alarm: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of an alarm with its ``step`` mapped to a canonical name."""
    a = dict(alarm)
    a["step"] = _STEP_ALIAS.get(a.get("step"), a.get("step"))
    return a


# [3] Input loading --------------------------------------------------------------

def load_dar(dar) -> Dict[str, Any]:
    """Accept either a path/str or an already-parsed dict and return the dict."""
    if isinstance(dar, (str, os.PathLike)):
        with open(dar) as fh:
            return json.load(fh)
    return dar


# [4] Build the check_all ``state`` dict from the main model output ----------------
#
# Hard rule: only copy fields that are actually present in the model output.
# Missing fields are omitted so the corresponding check is skipped — never
# fabricated.

def build_state(dar: Dict[str, Any]) -> Dict[str, Any]:
    """Construct the ``state`` dict for ``alarms.check_all`` from real model fields.

    Supported fields (see alarms.INPUT_CONTRACT):
        dist, mean_dar, dar_std, tcep_eq, bonds_opened, per_bond_open,
        reduction_completion, t_eff_s, thiolate_pct, ph, p_conj,
        kinetic_completion, payload_feed_ratio, target_dar
    """
    params = dar.get("params", {}) or {}
    reduction = dar.get("disulfide_reduction", {}) or {}
    primary = dar.get("primary", {}) or {}
    sites = dar.get("sites", {}) or {}

    state: Dict[str, Any] = {}

    # combinatorial statistics: DAR distribution + moments
    if "dist" in primary:
        state["dist"] = primary["dist"]
    if "mean_dar" in primary:
        state["mean_dar"] = primary["mean_dar"]
    if "std_dar" in primary:
        state["dar_std"] = primary["std_dar"]

    # disulfide reduction: reductant equivalents + bonds opened
    if "TCEP_cal_eq" in params:
        state["tcep_eq"] = params["TCEP_cal_eq"]
    if "bonds_opened" in reduction:
        state["bonds_opened"] = reduction["bonds_opened"]
    # per_bond_open, reduction_completion: NOT present in this model output ->
    # deliberately omitted (check skipped, no fabricated data).

    # site accessibility: effective exposure time per site (seconds)
    t_eff = [s["t_eff_s"] for s in sites.values() if isinstance(s, dict) and "t_eff_s" in s]
    if t_eff:
        state["t_eff_s"] = t_eff

    # thiol activation: thiolate fraction is stored as a 0..1 fraction in the
    # model output; alarms.check_all expects a percentage, so convert.
    ft = [s["f_thiolate"] for s in sites.values() if isinstance(s, dict) and "f_thiolate" in s]
    if ft:
        state["thiolate_pct"] = sum(ft) / len(ft) * 100.0

    # covalent conjugation: pH, conjugation probability, payload charge equivalents
    if "pH" in params:
        state["ph"] = params["pH"]
    if "P_CONJ" in params:
        state["p_conj"] = params["P_CONJ"]
    if "payload_eq" in params:
        state["payload_feed_ratio"] = params["payload_eq"]
    # kinetic_completion: NOT present in this model output -> omitted (skipped).
    # target_dar: NOT present in this model output -> omitted (skipped).

    return state


# [5] Collect the three layer alarms + run check_all ------------------------------

def _collect(
    pdb_path: str,
    dar,
    payload_sdf: Optional[str] = None,
    md_dir: str = "",
    working_uM: Optional[float] = None,
    site_env_opts: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Internal collector.

    Returns ``(alarms, diagnostics, state)``:
      * alarms      — merged list of every layer alarm + check_all alarms
      * diagnostics— per-step run status (ok / skipped / failed + reason)
      * state       — the ``state`` dict handed to ``check_all`` (for transparency)
    """
    dar = load_dar(dar)
    diagnostics: Dict[str, Dict[str, Any]] = {}
    alarms: List[Dict[str, Any]] = []

    # Authoritative site list for this run = the sites the main model used.
    sites = sorted((dar.get("sites", {}) or {}).keys())
    params = dar.get("params", {}) or {}

    # --- Layer 1: sequence QC ---------------------------------------------------
    try:
        seq_res = seqqc.assess(pdb_path, md_sites=sites or None)
        layer = [ _normalize(a) for a in seqqc.to_alarms(seq_res) ]
        alarms.extend(layer)
        diagnostics["sequence QC"] = dict(status="ok", n=len(layer))
    except Exception as exc:  # surface the real reason; do not fabricate alarms
        diagnostics["sequence QC"] = dict(status="failed", reason=f"{type(exc).__name__}: {exc}")

    # --- Layer 2: site environment / thiol activation ---------------------------
    if sites:
        try:
            opts = dict(site_env_opts or {})
            opts.setdefault("ph", params.get("pH", site_env.DEFAULT_PH))
            opts.setdefault("conj_time_h", params.get("time_h", 2.0))
            opts.setdefault("k2_ref", params.get("K2", 300.0))
            se = site_env.assess(
                pdb_path, sites, md_dir=md_dir,
                ph=opts["ph"], conj_time_h=opts["conj_time_h"], k2_ref=opts["k2_ref"],
            )
            layer = [ _normalize(a) for a in se.get("alarms", []) ]
            alarms.extend(layer)
            diagnostics["thiol activation"] = dict(status="ok", n=len(layer))
        except Exception as exc:
            diagnostics["thiol activation"] = dict(status="failed", reason=f"{type(exc).__name__}: {exc}")
    else:
        diagnostics["thiol activation"] = dict(
            status="skipped", reason="no site list available from the model output")

    # --- Layer 3: linker-payload QC --------------------------------------------
    try:
        if not payload_qc._HAS_RDKIT:
            diagnostics["linker-payload QC"] = dict(
                status="skipped", reason="rdkit not available in this environment")
        else:
            mols, _ = payload_qc.load_ensemble(payload_sdf)
            res = payload_qc.analyse(mols, temperature_C=4.0)
            # working concentration for the solubility check; prefer the real
            # process numbers when present, else leave solubility un-attached.
            if working_uM is None:
                if "payload_eq" in params and "mAb_uM" in params:
                    working_uM = params["payload_eq"] * params["mAb_uM"]
            if working_uM is not None:
                res = payload_qc.attach_solubility(res, float(working_uM))
            layer = [ _normalize(a) for a in payload_qc.to_alarms(res) ]
            alarms.extend(layer)
            diagnostics["linker-payload QC"] = dict(status="ok", n=len(layer))
    except Exception as exc:
        diagnostics["linker-payload QC"] = dict(
            status="failed", reason=f"{type(exc).__name__}: {exc}")

    # --- Main model checks: alarms.check_all ------------------------------------
    state = build_state(dar)
    try:
        layer = [ _normalize(a) for a in check_all(state) ]
        alarms.extend(layer)
        diagnostics["combinatorial / process checks (check_all)"] = dict(status="ok", n=len(layer))
    except Exception as exc:
        diagnostics["combinatorial / process checks (check_all)"] = dict(
            status="failed", reason=f"{type(exc).__name__}: {exc}")

    return alarms, diagnostics, state


def collect_alarms(
    pdb_path: str,
    dar,
    payload_sdf: Optional[str] = None,
    md_dir: str = "",
    working_uM: Optional[float] = None,
    site_env_opts: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Collect the three layer alarms + ``check_all`` alarms and merge into one list.

    This is the public, spec-named entry point; it returns the merged alarm list.
    """
    alarms, _, _ = _collect(pdb_path, dar, payload_sdf=payload_sdf, md_dir=md_dir,
                            working_uM=working_uM, site_env_opts=site_env_opts)
    return alarms


# [6] Grouping & summary ---------------------------------------------------------

def group_by_step(alarms: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {st: [] for st in STEP_ORDER}
    for a in alarms:
        out.setdefault(a.get("step"), []).append(a)
    # drop empty steps (keeps output focused on what actually ran)
    return {st: al for st, al in out.items() if al}


def run_report(
    pdb_path: str,
    dar,
    payload_sdf: Optional[str] = None,
    md_dir: str = "",
    working_uM: Optional[float] = None,
    site_env_opts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the full report dict.

    Returns at least: ``alarms``, ``summary`` (from alarms.summarize_alarms),
    ``by_step``.  Also includes ``diagnostics`` (which modules ran / failed and
    why) and ``state`` (the fields fed to check_all) for full transparency.
    """
    alarms, diagnostics, state = _collect(
        pdb_path, dar, payload_sdf=payload_sdf, md_dir=md_dir,
        working_uM=working_uM, site_env_opts=site_env_opts)
    summary = summarize_alarms(alarms)
    by_step = group_by_step(alarms)
    return dict(
        alarms=alarms,
        summary=summary,
        by_step=by_step,
        diagnostics=diagnostics,
        state=state,
    )


# [7] Rendering — Markdown (human readable) --------------------------------------

_LEVEL_BADGE = {OK: "OK", WARN: "WARN", CRIT: "CRIT", INFO: "INFO"}


def _alarm_row(a: Dict[str, Any]) -> List[str]:
    return [
        a.get("item", ""),
        f"{a.get('value', '')} {a.get('unit', '')}".strip(),
        _LEVEL_BADGE.get(a.get("level"), str(a.get("level"))),
        str(a.get("reference", "")),
        str(a.get("meaning", "")),
        str(a.get("action", "")),
    ]


def _cause_rows(a: Dict[str, Any]) -> List[List[str]]:
    rows = []
    for c in a.get("causes", []) or []:
        rows.append([
            c.get("name", ""),
            str(c.get("metric", "")),
            str(c.get("value", "")),
            str(c.get("ref", "")),
            str(c.get("verdict", "")),
        ])
    return rows


def render_markdown(
    report: Dict[str, Any],
    pdb_path: str,
    dar_path: str,
    md_dir: str = "",
    payload_sdf: Optional[str] = None,
) -> str:
    """Render a biologist-friendly Markdown report from a run_report() dict."""
    summary = report["summary"]
    by_step = report["by_step"]
    diagnostics = report["diagnostics"]
    state = report["state"]

    sections: List[Tuple[str, str]] = []

    # --- Top legend -----------------------------------------------------------
    legend = (
        "This report collects every warning the pipeline can raise about an "
        "antibody–drug conjugation run, grouped by the step of the process it "
        "concerns.  Each alarm answers three questions: **what** is off, **how "
        "we judge it** (value vs. reference range), and **what to do** about it.\n\n"
        "- **OK** — within the expected range.\n"
        "- **WARN** — drifting out of specification; review before scaling up.\n"
        "- **CRIT** — out of specification; the batch is at risk.\n"
        "- **INFO** — worth noting, threshold not triggered.\n\n"
        "Process steps in order: " + " → ".join(STEP_ORDER) + "."
    )
    sections.append(("How to read this report", legend))

    # --- Overall verdict ------------------------------------------------------
    worst = summary.get("worst", OK)
    n_crit = summary.get("n_crit", 0)
    n_warn = summary.get("n_warn", 0)
    verdict_lines = [
        f"- **Overall status:** {_LEVEL_BADGE.get(worst, worst)}",
        f"- **Critical alarms:** {n_crit}    **Warning alarms:** {n_warn}",
    ]
    triggered = summary.get("triggered", [])
    if triggered:
        verdict_lines.append("")
        verdict_lines.append("**Alarms that need attention:**")
        for a in triggered:
            verdict_lines.append(
                f"  - [{_LEVEL_BADGE.get(a['level'], a['level'])}] "
                f"{a['step']} — {a['item']} = {a.get('value','')} {a.get('unit','')} "
                f"(reference {a.get('reference','')})")
    else:
        verdict_lines.append("")
        verdict_lines.append("No alarm crossed a threshold — every checked quantity is within range.")
    sections.append(("Overall verdict", "\n".join(verdict_lines)))

    # --- Per-step detail -----------------------------------------------------
    for step in STEP_ORDER:
        al = by_step.get(step)
        if not al:
            continue
        rows = [_alarm_row(a) for a in al]
        body = table(
            ["Check", "Value", "Level", "Reference", "What it means", "What to do"],
            rows,
        )
        # Append evidence-chain (candidate causes) for any triggered alarm.
        cause_blocks = []
        for a in al:
            if a.get("causes"):
                crows = _cause_rows(a)
                if crows:
                    cause_blocks.append(
                        f"**Evidence chain — {a['item']}**\n\n"
                        + table(["Candidate cause", "Mechanism metric", "Value", "Reference", "Verdict"], crows)
                    )
        if cause_blocks:
            body += "\n\n" + "\n\n".join(cause_blocks)
        sections.append((step, body))

    # --- Data integrity / diagnostics -----------------------------------------
    diag_lines = [
        "Which checks actually ran, and why any did not.  Missing data is never "
        "invented — a skipped step is reported as skipped.",
        "",
    ]
    for label, info in diagnostics.items():
        st = info.get("status", "?")
        extra = f" ({info.get('n')} alarms)" if info.get("n") is not None else ""
        reason = info.get("reason")
        line = f"- **{label}:** {st}{extra}"
        if reason:
            line += f" — {reason}"
        diag_lines.append(line)
    diag_lines.append("")
    diag_lines.append("**Fields fed to the process check (check_all):**")
    if state:
        for k, v in state.items():
            if isinstance(v, list) and len(v) > 8:
                v = f"list[{len(v)}] e.g. {v[:3]} …"
            diag_lines.append(f"  - `{k}` = {v}")
    else:
        diag_lines.append("  - (none — model output did not contain the needed fields)")
    sections.append(("Data integrity & what was checked", "\n".join(diag_lines)))

    # --- Inputs ---------------------------------------------------------------
    inputs = [
        f"- Structure file: `{pdb_path}`",
        f"- Model output JSON: `{dar_path}`",
        f"- MD frame library: `{md_dir or '(none — static structure used)'}`",
        f"- Linker-payload structure: `{payload_sdf or '(default ensemble)'}`",
    ]
    sections.append(("Inputs used", "\n".join(inputs)))

    title = "ADC conjugation process — consolidated alarm report"
    md = f"# {title}\n\n"
    md += f"_Generated {datetime.now().isoformat(timespec='seconds')}_\n\n"
    md += f"_Sources: structure `{pdb_path}`, model output `{dar_path}`_\n\n"
    for heading, body in sections:
        md += f"## {heading}\n\n{body.rstrip()}\n\n"
    return md


# [8] Rendering — JSON (machine readable) ----------------------------------------

def render_json(report: Dict[str, Any], pdb_path: str, dar_path: str,
                md_dir: str = "", payload_sdf: Optional[str] = None) -> Dict[str, Any]:
    return dict(
        meta=dict(
            generated=datetime.now().isoformat(timespec="seconds"),
            pdb_path=os.path.abspath(pdb_path),
            dar_json=os.path.abspath(str(dar_path)),
            md_dir=md_dir or None,
            payload_sdf=payload_sdf or None,
        ),
        summary=report["summary"],
        diagnostics=report["diagnostics"],
        state=report["state"],
        by_step=report["by_step"],
        alarms=report["alarms"],
    )


# [9] CLI ------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Consolidated alarm report for cysteine-conjugated ADCs "
                    "(sequence QC + thiol activation + linker-payload QC + process checks).")
    ap.add_argument("--pdb", required=True, help="antibody structure file (PDB)")
    ap.add_argument("--dar-json", required=True, help="main model output JSON "
                                                      "(e.g. dar_v6_complete.json)")
    ap.add_argument("--payload-sdf", default=None,
                    help="linker-payload conformational ensemble (SDF). "
                         "Defaults to the project's MMAE ensemble.")
    ap.add_argument("--md-dir", default="",
                    help="optional MD frame library for site-environment sampling")
    ap.add_argument("--working-um", type=float, default=None,
                    help="payload working concentration (µM) for the solubility check; "
                         "default = payload_eq × mAb_uM from the model output")
    ap.add_argument("--out-md", default="",
                    help="write the Markdown report to this path")
    ap.add_argument("--out-json", default="",
                    help="write the machine-readable JSON report to this path")
    args = ap.parse_args(argv)

    report = run_report(
        args.pdb, args.dar_json,
        payload_sdf=args.payload_sdf or None,
        md_dir=args.md_dir or "",
        working_uM=args.working_um,
    )

    md = render_markdown(report, args.pdb, args.dar_json,
                         md_dir=args.md_dir or "", payload_sdf=args.payload_sdf)
    js = render_json(report, args.pdb, args.dar_json,
                     md_dir=args.md_dir or "", payload_sdf=args.payload_sdf)

    # Always print the Markdown report to stdout.
    print(md)

    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(md)
        print(f"\n[written] {args.out_md}")
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(js, indent=2))
        print(f"[written] {args.out_json}")

    # Non-zero exit when the batch is at risk (CRIT present).
    return 1 if report["summary"].get("worst") == CRIT else 0


if __name__ == "__main__":
    raise SystemExit(main())
