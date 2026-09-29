from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

from .accessibility import filter_sites, score_sites
from .beam import funnel
from .chemistry import classify_conjugation
from .config import load_config
from .io import load_antibody, load_lp_ensemble, write_adc_pdb
from .qc import measure_state
from .report import table, write_markdown
from .sampling import build_dar1_library, build_open_space
from .sites import (
    detect_disulfides,
    interchain_sites,
    list_cysteines,
    reduction_difficulty_profile,
    resolve_conjugation_sites,
    sasa_table,
    surface_lysines,
)


def _conjugation_type(chemistry: dict, lp_sdf: str) -> tuple[str, str, dict | None]:
    """Decide the conjugation chemistry from the config / LP warhead.

    Returns (conj_type, warhead_label, classifier_result_or_None).
    conj_type is one of "cysteine" | "lysine" | "unknown". When the config says
    "auto" the LP SDF is inspected for a warhead; otherwise the config value (or the
    default "cysteine") wins. This is the model's first decision: it picks which
    sites to go looking for on the antibody.
    """
    conj_type = str(chemistry.get("conjugation_type", "cysteine")).lower()
    if conj_type == "auto":
        cls = classify_conjugation(lp_sdf)
        return cls["conjugation_type"], cls["warhead"], cls
    return conj_type, chemistry.get("warhead"), None


def prune_library(library: dict, max_per_site: int) -> dict:
    pruned = {}
    for site, poses in library.items():
        ranked = sorted(poses, key=lambda p: p["min_dist"], reverse=True)
        pruned[site] = ranked[:max_per_site] if max_per_site else ranked
    return pruned


def run(cfg_path: str) -> dict:
    cfg = load_config(cfg_path)
    out = Path(cfg["outputs"]["dir"])
    out.mkdir(parents=True, exist_ok=True)

    model = load_antibody(cfg["inputs"]["antibody_pdb"])
    mols = load_lp_ensemble(cfg["inputs"]["lp_sdf"], remove_hs=False)
    stride = int(cfg["search"].get("conformer_stride", 1))
    mols = mols[::stride]
    sites_cfg = cfg.get("sites") or {}
    chemistry = cfg["chemistry"]
    sampling = cfg["sampling"]
    qc = cfg["qc"]
    search = cfg["search"]

    conj_type, conj_warhead, conj_class = _conjugation_type(
        chemistry, cfg["inputs"]["lp_sdf"]
    )
    cys = list_cysteines(model)
    # First decision: judge the conjugation chemistry from the LP, then find the
    # matching sites (and, for cysteine, the disulfide reduction-difficulty profile).
    resolved = resolve_conjugation_sites(
        model,
        conj_type,
        cfg["antibody"]["chain_types"],
        cfg["disulfide"],
        sasa_min=float((cfg.get("accessibility") or {}).get("min_lysine_sasa", 10.0)),
    )
    pairs = resolved.get("pairs", [])
    reduction_profile = resolved.get("reduction_profile")
    auto_sites = resolved["sites"]
    sites = sites_cfg.get("pool") or auto_sites

    # Lysine conjugation: sites are found, but the DAR funnel is maleimide/
    # cysteine-specific and is not implemented yet — stop after reporting.
    if conj_type == "lysine":
        payload = {
            "conjugation_type": "lysine",
            "warhead": conj_warhead,
            "classifier": conj_class,
            "n_lysines_accessible": len(auto_sites),
            "reactive_sites": auto_sites,
            "all_lysines": resolved.get("all_lysines"),
            "reduction_profile": None,
            "note": resolved["note"],
        }
        (out / "01_sites.json").write_text(json.dumps(payload, indent=2))
        (out / "run_summary.json").write_text(
            json.dumps(
                {
                    "conjugation_type": "lysine",
                    "sites": auto_sites,
                    "note": "lysine DAR funnel not yet implemented; sites + reduction note written",
                },
                indent=2,
            )
        )
        print(json.dumps(payload, indent=2))
        return payload

    sasa = sasa_table(model, sites)
    access_cfg = cfg.get("accessibility") or {}
    access_rows = score_sites(model, sites, sampling, qc, access_cfg, sasa)
    keep, dropped = filter_sites(
        access_rows, drop_failed=bool(access_cfg.get("drop_failed", True))
    )
    (out / "03_accessibility.json").write_text(
        json.dumps(
            {
                "sites_in": sites,
                "sites_keep": keep,
                "sites_dropped": dropped,
                "rows": access_rows,
            },
            indent=2,
        )
    )
    sites = keep
    if not sites:
        raise RuntimeError(
            "chemical accessibility gate dropped every Cys; "
            "relax accessibility.min_open_rays / min_cys_sasa or inspect 03_accessibility.json"
        )

    (out / "01_sites.json").write_text(
        json.dumps(
            {
                "n_cys": len(cys),
                "conjugation_type": conj_type,
                "warhead": conj_warhead,
                "classifier": conj_class,
                "n_disulfides": len(pairs),
                "pairs": [
                    {k: v for k, v in p.items() if k not in ("i", "j")}
                    for p in pairs
                ],
                "reduction_profile": reduction_profile,
                "reactive_sites": sites,
                "sasa": sasa,
                "accessibility": access_rows,
                "sites_dropped_by_access": dropped,
                "n_lp_conformers": len(mols),
            },
            indent=2,
        )
    )

    cache = out / "02_dar1_library.pkl"
    if cache.exists() and cfg["search"].get("reuse_library", True):
        library = pickle.loads(cache.read_bytes())
        rebuilt = False
    else:
        open_space = build_open_space(model, sites, sampling, qc)
        (out / "02_open_space.json").write_text(
            json.dumps(
                {
                    site: {
                        "n_front": int(len(info["front"])),
                        "n_back": int(len(info["back"])),
                    }
                    for site, info in open_space.items()
                },
                indent=2,
            )
        )
        max_side = search.get("max_poses_per_site_side")
        library = build_dar1_library(
            model,
            mols,
            open_space,
            chemistry,
            sampling,
            qc,
            max_poses_per_site_side=max_side,
        )
        cache.write_bytes(pickle.dumps(library, protocol=pickle.HIGHEST_PROTOCOL))
        rebuilt = True

    counts = {s: len(p) for s, p in library.items()}
    library = prune_library(library, int(search.get("max_poses_per_site", 400)))

    max_dar = min(int(search.get("max_dar", 8)), len(sites))
    dar = funnel(
        library,
        model,
        chemistry,
        qc,
        search,
        max_dar=max_dar,
    )
    ceiling = next((n for n in range(1, max_dar + 1) if not dar.get(n)), max_dar + 1)

    summary_rows = []
    best_qc = {}
    for n, states in sorted(dar.items()):
        if not states:
            summary_rows.append([n, 0, "EMPTY", "", "", "", ""])
            continue
        best = states[0]
        meas = measure_state(best, model, chemistry, qc)
        best_qc[n] = meas
        summary_rows.append(
            [
                n,
                len(states),
                "PASS" if meas["pass"] else "FAIL",
                ", ".join(best["sites"]),
                f"{meas['protein_min']:.3f}",
                f"{meas['lp_min']:.3f}" if meas["lp_min"] is not None else "n/a",
                f"{min(r['bond'] for r in meas['per_site']):.3f}–{max(r['bond'] for r in meas['per_site']):.3f}",
            ]
        )
        if meas["pass"]:
            pdb_path = out / f"ADC_DAR{n}.pdb"
            conjugates = [
                {"site": s, "pose": p} for s, p in zip(best["sites"], best["poses"])
            ]
            write_adc_pdb(
                cfg["inputs"]["antibody_pdb"],
                conjugates,
                str(pdb_path),
                reactive_atom=int(chemistry["reactive_atom"]),
                resname=chemistry.get("resname", "LPP"),
            )
            (out / f"ADC_DAR{n}_bonds.json").write_text(
                json.dumps(
                    {
                        "sites": best["sites"],
                        "s_c_distances": [
                            {"site": s, "s_c": r["bond"], "pass": True}
                            for s, r in zip(best["sites"], meas["per_site"])
                        ],
                    },
                    indent=2,
                )
            )

    last_ok = max((n for n, m in best_qc.items() if m["pass"]), default=0)

    csv_path = out / "dar_funnel.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["DAR", "n_states", "qc", "sites", "protein_min", "lp_min", "bond_range"]
        )
        w.writerows(summary_rows)

    site_rows = [
        [
            s["site"],
            f"{s['cys_sasa']:.2f}",
            s["neighbors_4A"],
            counts.get(s["site"], 0),
        ]
        for s in sasa
        if s["site"] in counts or s["site"] in dropped
    ]
    access_table = [
        [
            r["site"],
            r["verdict"],
            f"{r['cys_sasa']:.1f}",
            f"{r['sg_sasa']:.1f}",
            r["n_open_rays"],
            r["n_attack_rays"],
            f"{r['best_attack_angle_deg']:.0f}"
            if r["best_attack_angle_deg"] is not None
            else "n/a",
            r["reason"],
        ]
        for r in access_rows
    ]
    per_dar_detail = []
    for n, meas in best_qc.items():
        for r in meas["per_site"]:
            per_dar_detail.append(
                f"- DAR{n} {r['site']} side={r['side']} S–C={r['bond']:.3f} Å protein-min={r['protein_min']:.3f} Å"
            )

    report = write_markdown(
        str(out / "QC_REPORT.md"),
        "adcsim QC report — Trastuzumab × mc-vc-PAB-MMAE",
        [
            (
                "What this is",
                "Computational cysteine-conjugation spatial modelling. "
                "QC pass means the generated pose satisfies covalent geometry "
                "and steric cutoffs. It is **not** experimental DAR, affinity, "
                "or cytotoxicity evidence.",
            ),
            (
                "Chemistry",
                f"- Reactive atom (RDKit heavy index): **{chemistry['reactive_atom']}** "
                f"(PDB name {chemistry.get('reactive_atom_name', 'C91')})\n"
                f"- Exit atom: {chemistry['exit_atom']}\n"
                f"- Allowed S–C window: {chemistry['bond_min']}–{chemistry['bond_max']} Å\n"
                f"- Generation bond values: {chemistry.get('bond_values', [chemistry.get('bond_step')])}\n"
                f"- Protein–LP min: {qc['protein_lp_min']} Å (Cys-SG of the bonded site excluded)\n"
                f"- LP–LP min: {qc['lp_lp_min']} Å\n"
                f"- LP conformers used: {len(mols)} (stride={stride})\n"
                f"- Library rebuilt this run: {rebuilt}",
            ),
            (
                "Chemical accessibility (before DAR1 library)",
                table(
                    [
                        "site",
                        "verdict",
                        "Cys-SASA",
                        "SG-SASA",
                        "open rays",
                        "attack rays",
                        "best angle",
                        "note",
                    ],
                    access_table,
                )
                + (
                    f"\n\nDropped FAIL sites (not sampled): {', '.join(dropped)}"
                    if dropped
                    else "\n\nNo site dropped."
                )
                + "\nSG-SASA = 0 is common for hinge Cys and is not a FAIL by itself.",
            ),
            (
                "Reactive Cys pool",
                table(
                    ["site", "Cys-SASA (Å²)", "4Å neighbours", "0-clash poses (raw)"],
                    site_rows,
                ),
            ),
            (
                "DAR funnel",
                table(
                    ["DAR", "states kept", "QC", "sites", "protein-min", "LP-LP min", "S–C range"],
                    summary_rows,
                )
                + f"\n\n**Highest QC-passing DAR this run: DAR{last_ok}**\n"
                f"**Computational ceiling on this backbone: DAR{ceiling}** "
                "(first empty level). Do not loosen clash cutoffs to force more."
                if ceiling
                else f"\n\n**Highest QC-passing DAR this run: DAR{last_ok}**",
            ),
            ("Best pose detail", "\n".join(per_dar_detail) or "_(none)_"),
            (
                "What this cannot prove",
                "- Experimental conjugation occupancy\n"
                "- Binding affinity (PLIP / GNINA scores, if added later, are not Kd)\n"
                "- ADC developability, PK, or cytotoxicity",
            ),
        ],
    )

    result = {
        "out": str(out),
        "report": report,
        "conjugation_type": conj_type,
        "warhead": conj_warhead,
        "reduction_profile": reduction_profile,
        "highest_pass_dar": last_ok,
        "ceiling": ceiling,
        "pose_counts": {k: int(v) for k, v in counts.items()},
        "n_states": {int(k): len(v) for k, v in dar.items()},
        "sites_dropped_by_access": dropped,
    }
    (out / "run_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def run_access_only(cfg_path: str) -> dict:
    """Cheap gate: sites + SASA + chemical accessibility. No DAR1 library."""
    cfg = load_config(cfg_path)
    out = Path(cfg["outputs"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    model = load_antibody(cfg["inputs"]["antibody_pdb"])
    sites_cfg = cfg.get("sites") or {}
    chemistry = cfg["chemistry"]
    conj_type, conj_warhead, conj_class = _conjugation_type(
        chemistry, cfg["inputs"]["lp_sdf"]
    )
    cys = list_cysteines(model)
    resolved = resolve_conjugation_sites(
        model,
        conj_type,
        cfg["antibody"]["chain_types"],
        cfg["disulfide"],
        sasa_min=float((cfg.get("accessibility") or {}).get("min_lysine_sasa", 10.0)),
    )
    pairs = resolved.get("pairs", [])
    reduction_profile = resolved.get("reduction_profile")
    auto_sites = resolved["sites"]
    sites = sites_cfg.get("pool") or auto_sites

    # Lysine: report sites only (the accessibility gate below is cysteine-specific).
    if conj_type == "lysine":
        payload = {
            "conjugation_type": "lysine",
            "warhead": conj_warhead,
            "classifier": conj_class,
            "n_lysines_accessible": len(auto_sites),
            "reactive_sites": auto_sites,
            "all_lysines": resolved.get("all_lysines"),
            "reduction_profile": None,
            "note": resolved["note"],
        }
        (out / "01_sites.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        return payload
    sasa = sasa_table(model, sites)
    access_cfg = cfg.get("accessibility") or {}
    access_rows = score_sites(
        model, sites, cfg["sampling"], cfg["qc"], access_cfg, sasa
    )
    keep, dropped = filter_sites(
        access_rows, drop_failed=bool(access_cfg.get("drop_failed", True))
    )
    payload = {
        "n_cys": len(cys),
        "conjugation_type": conj_type,
        "warhead": conj_warhead,
        "classifier": conj_class,
        "n_disulfides": len(pairs),
        "reduction_profile": reduction_profile,
        "sites_in": sites,
        "sites_keep": keep,
        "sites_dropped": dropped,
        "rows": access_rows,
    }
    (out / "03_accessibility.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return payload


def main(argv=None):
    p = argparse.ArgumentParser(prog="adcsim")
    p.add_argument("config", help="path to config.yaml")
    p.add_argument(
        "--access-only",
        action="store_true",
        help="run chemical accessibility gate only (no DAR1 / funnel)",
    )
    p.add_argument(
        "--polish",
        action="store_true",
        help="constrained minimization (polish) on ADC_DAR*.pdb in outputs.dir",
    )
    args = p.parse_args(argv)
    if args.polish:
        from .polish import polish_dir

        cfg = load_config(args.config)
        polish_dir(cfg["outputs"]["dir"])
    elif args.access_only:
        run_access_only(args.config)
    else:
        run(args.config)


if __name__ == "__main__":
    main()
