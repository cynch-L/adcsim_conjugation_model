"""Constrained minimization = the polish step.

Same protocol for every representative ADC (DAR1–7), not only the ceiling.
This is not a filter: input is already QC-pass; output is a quieter pose.

Laptop path (no PyRosetta): freeze C91, MMFF/UFF the rest of each payload.
RunPod path: same freeze + protein side chains within 8 Å (needs ligand params).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .clash import min_distance, protein_clash_ok
from .report import table, write_markdown
from .solvation import solvation_collapse

BOND_MIN = 1.75
BOND_MAX = 1.90
PROTEIN_LP_MIN = 2.0
LP_LP_MIN = 2.0
C91_NAME = "C91"


def _serial(line: str) -> int:
    return int(line[6:11])


def _is_atom(line: str) -> bool:
    return line.startswith(("ATOM", "HETATM"))


def _is_heavy(line: str) -> bool:
    if not _is_atom(line):
        return False
    name = line[12:16].strip()
    elem = line[76:78].strip() if len(line) >= 78 else ""
    if elem == "H" or name.startswith("H"):
        return False
    return True


def _xyz(line: str) -> np.ndarray:
    return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])


def _resname(line: str) -> str:
    return line[17:20].strip()


def _resseq(line: str) -> int:
    return int(line[22:26])


def _atom_name(line: str) -> str:
    return line[12:16].strip()


def parse_conect(lines: list[str]) -> list[tuple[int, int]]:
    pairs = []
    for line in lines:
        if not line.startswith("CONECT"):
            continue
        nums = []
        i = 6
        while i + 5 <= len(line.rstrip()):
            chunk = line[i : i + 5].strip()
            if chunk:
                nums.append(int(chunk))
            i += 5
        if len(nums) < 2:
            continue
        a = nums[0]
        for b in nums[1:]:
            pairs.append((a, b))
    return pairs


def load_adc_pdb(path: str, lines: list[str] | None = None) -> dict:
    lines = list(lines) if lines is not None else Path(path).read_text().splitlines()
    atoms = {}
    for line in lines:
        if _is_atom(line):
            atoms[_serial(line)] = line
    ligands: dict[int, dict] = {}
    protein_xyz = []
    protein_serials = []
    sg_of = {}
    for serial, line in atoms.items():
        if _resname(line) == "LPP":
            rid = _resseq(line)
            ligands.setdefault(
                rid,
                {"serials": [], "c91": None, "lines": {}},
            )
            ligands[rid]["serials"].append(serial)
            ligands[rid]["lines"][serial] = line
            if _atom_name(line) == C91_NAME:
                ligands[rid]["c91"] = serial
        elif _is_heavy(line):
            protein_xyz.append(_xyz(line))
            protein_serials.append(serial)
            if _resname(line) == "CYS" and _atom_name(line) == "SG":
                sg_of[serial] = line
    pairs = parse_conect(lines)
    serial_set = {s for lig in ligands.values() for s in lig["serials"]}
    for a, b in pairs:
        if a in sg_of and b in serial_set:
            for lig in ligands.values():
                if b in lig["serials"]:
                    lig["sg_serial"] = a
                    lig["sg_line"] = sg_of[a]
        elif b in sg_of and a in serial_set:
            for lig in ligands.values():
                if a in lig["serials"]:
                    lig["sg_serial"] = b
                    lig["sg_line"] = sg_of[b]
    return {
        "path": path,
        "lines": lines,
        "atoms": atoms,
        "ligands": ligands,
        "protein_xyz": np.asarray(protein_xyz, dtype=float),
        "protein_serials": protein_serials,
        "conect": pairs,
    }


def ligand_xyz(lig: dict) -> np.ndarray:
    return np.asarray([_xyz(lig["lines"][s]) for s in lig["serials"]], dtype=float)


def measure_pdb(adc: dict) -> dict:
    ligs = sorted(adc["ligands"])
    per = []
    protein_mins = []
    for rid in ligs:
        lig = adc["ligands"][rid]
        xyz = ligand_xyz(lig)
        sg_xyz = _xyz(lig["sg_line"])
        c91 = _xyz(lig["lines"][lig["c91"]])
        bond = float(np.linalg.norm(c91 - sg_xyz))
        sg_serial = lig["sg_serial"]
        prot = np.asarray(
            [
                adc["protein_xyz"][i]
                for i, s in enumerate(adc["protein_serials"])
                if s != sg_serial
            ],
            dtype=float,
        )
        pmin = min_distance(xyz, prot)
        protein_mins.append(pmin)
        chain = lig["sg_line"][21]
        res = int(lig["sg_line"][22:26])
        per.append(
            {
                "lp": rid,
                "site": f"{chain}:CYS{res}",
                "s_c": bond,
                "protein_min": pmin,
            }
        )
    lp_mins = []
    xyzs = [ligand_xyz(adc["ligands"][rid]) for rid in ligs]
    for i in range(len(xyzs)):
        for j in range(i + 1, len(xyzs)):
            lp_mins.append(min_distance(xyzs[i], xyzs[j]))
    lp_min = min(lp_mins) if lp_mins else None
    prot_min = min(protein_mins) if protein_mins else None
    bond_ok = all(BOND_MIN <= r["s_c"] <= BOND_MAX for r in per)
    clash_ok = (prot_min is None or prot_min >= PROTEIN_LP_MIN) and (
        lp_min is None or lp_min >= LP_LP_MIN
    )
    return {
        "n_lp": len(ligs),
        "protein_min": prot_min,
        "lp_min": lp_min,
        "bond_ok": bond_ok,
        "clash_ok": clash_ok,
        "pass": bond_ok and clash_ok,
        "per_site": per,
    }


def _ligand_mol(lig: dict, conect: list[tuple[int, int]]) -> tuple[Chem.Mol, dict[int, int]]:
    serials = list(lig["serials"])
    index = {s: i for i, s in enumerate(serials)}
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(serials))
    for i, s in enumerate(serials):
        line = lig["lines"][s]
        elem = line[76:78].strip() if len(line) >= 78 else _atom_name(line)[0]
        if not elem:
            elem = "C"
        atom = Chem.Atom(elem)
        atom.SetPDBResidueInfo(Chem.AtomPDBResidueInfo(_atom_name(line)))
        rw.AddAtom(atom)
        x, y, z = _xyz(line)
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    seen = set()
    for a, b in conect:
        if a in index and b in index:
            key = tuple(sorted((index[a], index[b])))
            if key not in seen:
                rw.AddBond(key[0], key[1], Chem.BondType.SINGLE)
                seen.add(key)
    mol = rw.GetMol()
    mol.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(mol, catchErrors=True)
    return mol, index


def _fold_ligand(
    lig: dict,
    adc: dict,
    others_xyz: list[np.ndarray],
    n_random: int = 16,
    angles: tuple[float, ...] = (-120.0, -90.0, -60.0, -30.0, 30.0, 60.0, 90.0, 120.0),
) -> dict[int, np.ndarray] | None:
    """Fold one LPP onto the antibody surface by rotating linker torsions with
    C91 pinned, clash-free against protein + every other LPP.

    This is the "collapse onto the surface" step. Unlike MMFF (which is downhill-only local
    minimization), rotating torsions crosses the barriers that separate the
    extended rod from a surface-hugging compact coil.

    Returns updated serial->coords, or None if no fold improves on the
    incoming pose (caller keeps the original).
    """
    from .fold import compactness, linker_rotatable_bonds, rotate_heavy, _self_clash_ok

    mol, index = _ligand_mol(lig, adc["conect"])
    serials = list(lig["serials"])
    xyz0 = ligand_xyz(lig)  # mol-order == serials order
    heavy = [i for i, s in enumerate(serials)
             if (lig["lines"][s][76:78].strip() if len(lig["lines"][s]) >= 78 else "C") != "H"]
    if not heavy:
        return None
    c91_mol = index[lig["c91"]]
    try:
        anchor = heavy.index(c91_mol)
    except ValueError:
        return None

    sg_xyz = _xyz(lig["sg_line"])
    prot = np.asarray(
        [adc["protein_xyz"][i] for i, s in enumerate(adc["protein_serials"]) if s != lig["sg_serial"]],
        dtype=float,
    )

    def accept(xyz):
        hx = xyz[heavy]
        if not protein_clash_ok(hx, prot, PROTEIN_LP_MIN)[0]:
            return None
        if not _self_clash_ok(xyz, heavy, mol):
            return None
        for o in others_xyz:
            if min_distance(hx, o) < LP_LP_MIN:
                return None
        if float(np.linalg.norm(xyz[anchor] - xyz0[anchor])) > 0.05:
            return None
        return compactness(hx, sg_xyz)

    bonds = [b for b in linker_rotatable_bonds(mol, c91_mol)[:18]
             if b[0] in heavy and b[1] in heavy]
    base = accept(xyz0)
    if base is None or not bonds:
        return None
    best_xyz = xyz0
    best_reach = base["reach"]

    # greedy sequential collapse: keep any angle that shortens reach
    xyz = xyz0.copy()
    for (a, b) in bonds:
        current = accept(xyz)
        if current is None:
            break
        current_reach = current["reach"]
        for ang in angles:
            trial = rotate_heavy(xyz, heavy, mol, a, b, ang)
            item = accept(trial)
            if item is not None and item["reach"] < current_reach - 0.3:
                xyz = trial
                current_reach = item["reach"]
    c = accept(xyz)
    if c is not None and c["reach"] < best_reach:
        best_xyz, best_reach = xyz, c["reach"]

    # random multi-bond walks biased toward near-C91 bonds (U-turn the linker).
    # IMPORTANT: the intermediate states of a walk are *exploration steps*, not
    # candidate poses. A transient clash mid-walk is fine — the ligand may pass
    # through a bumpy path and still land in a clean compact well. So we do NOT
    # abort the walk on a clashing intermediate; we simply keep sampling and
    # test clash only at the *landing* pose. This makes the search span the
    # whole extended→folded interval and answers "does a 0-clash compact state
    # exist?" rather than "is the first compact step clean?".
    rng = np.random.default_rng(91)
    n_bonds = len(bonds)
    for _ in range(n_random):
        xyz = xyz0.copy()
        weights = np.array([3.0 if i < 8 else 1.0 for i in range(n_bonds)], dtype=float)
        weights /= weights.sum()
        n_move = int(rng.integers(4, max(5, min(10, n_bonds + 1))))
        chosen = rng.choice(n_bonds, size=min(n_move, n_bonds), replace=False, p=weights)
        for i in chosen:
            a, b = bonds[int(i)]
            ang = float(rng.choice(angles))
            xyz = rotate_heavy(xyz, heavy, mol, a, b, ang)
        item = accept(xyz)  # clash is judged ONLY at the landing pose
        if item is None:
            continue
        if item["reach"] < best_reach - 0.3:
            best_xyz, best_reach = xyz, item["reach"]

    # Exhaustive single-bond scan: for every linker torsion, sweep the full
    # angular grid and keep any clean pose that lands more compact. This is the
    # cheap systematic cover of the extended→spherical interval.
    for (a, b) in bonds:
        if a not in heavy or b not in heavy:
            continue
        for ang in np.arange(-180.0, 180.0, 20.0):
            trial = rotate_heavy(xyz0, heavy, mol, a, b, float(ang))
            item = accept(trial)
            if item is None:
                continue
            if item["reach"] < best_reach - 0.3:
                best_xyz, best_reach = trial, item["reach"]

    if best_reach >= base["reach"]:
        return None
    return {serials[i]: np.asarray(best_xyz[i]) for i in heavy}


def minimize_ligand(lig: dict, conect: list[tuple[int, int]], max_its: int = 400) -> dict[int, np.ndarray] | None:
    mol, index = _ligand_mol(lig, conect)
    mol_h = Chem.AddHs(mol, addCoords=True)
    c91_idx = index[lig["c91"]]
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
        if props is None:
            raise ValueError("no MMFF props")
        ff = AllChem.MMFFGetMoleculeForceField(mol_h, props)
    except Exception:
        try:
            ff = AllChem.UFFGetMoleculeForceField(mol_h)
        except Exception:
            return None
    if ff is None:
        return None
    ff.AddFixedPoint(c91_idx)
    try:
        ff.Minimize(maxIts=max_its)
    except Exception:
        return None
    conf = mol_h.GetConformer()
    moved = {}
    for serial, idx in index.items():
        p = conf.GetAtomPosition(idx)
        moved[serial] = np.array([p.x, p.y, p.z], dtype=float)
    c91_old = _xyz(lig["lines"][lig["c91"]])
    drift = float(np.linalg.norm(moved[lig["c91"]] - c91_old))
    if drift > 0.05:
        moved[lig["c91"]] = c91_old
    return moved


def apply_coords(adc: dict, updates: dict[int, np.ndarray]) -> list[str]:
    out = []
    for line in adc["lines"]:
        if _is_atom(line):
            serial = _serial(line)
            if serial in updates:
                x, y, z = updates[serial]
                line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
        out.append(line)
    return out


def polish_one(pdb_path: str, out_pdb: str, max_its: int = 400) -> dict:
    adc = load_adc_pdb(pdb_path)
    before = measure_pdb(adc)
    accepted: dict[int, np.ndarray] = {}
    n_solv = 0
    n_fold = 0
    n_mmff = 0
    engine = "none"

    ligs = list(adc["ligands"].values())

    # ── Step 0 — SOLVATION COLLAPSE each LPP independently ──
    # In solution, MMAE collapses via hydrophobic effect (clogP ~3–4).
    # We drive each payload toward minimal non-polar SASA, constrained only
    # by protein clash (LP-LP ignored — this is a *single-molecule* collapse).
    # C91 is pinned at the thioether; the bond beyond C91 rotates freely.
    # This gives each LPP its "initial solution-state compactness" before
    # the inter-LP crowding adjustments of Step 1.
    from rdkit import Chem
    from rdkit.Chem import AllChem

    prot_for_solv = np.asarray(
        [adc["protein_xyz"][i] for i in range(len(adc["protein_xyz"]))], dtype=float
    )

    # NOTE: clash test uses only heavy atoms. The old code hardcoded 94 (= the
    # vc-MMAE heavy-atom count), which silently breaks for any other payload.
    # RDKit AddHs appends hydrogens at the end, so heavy atoms are exactly the
    # first mol.GetNumAtoms() entries — we now derive that per ligand.

    for lig in ligs:
        if lig.get("c91") is None or "sg_line" not in lig:
            continue
        serials = list(lig["serials"])
        xyz_raw = np.array([_xyz(lig["lines"][s]) for s in serials], dtype=float)

        # Build RDKit Mol with Hs from this LP's atoms + CONECT
        mol, index = _ligand_mol(lig, adc["conect"])
        mol_h = Chem.AddHs(mol, addCoords=True)
        conf = mol_h.GetConformer()
        n_all = mol_h.GetNumAtoms()
        # AddHs already placed the Hs on reasonable positions; we only
        # overwrite the HEAVY atoms with the PDB coordinates, and
        # leave Hs (which RDKit placed) untouched so SASA has valid
        # positions for all atoms.
        xyz_full = np.zeros((n_all, 3), dtype=float)
        for i in range(n_all):
            pos = conf.GetAtomPosition(i)
            xyz_full[i] = (pos.x, pos.y, pos.z)
        for serial, idx in index.items():
            xyz_full[idx] = _xyz(lig["lines"][serial])
        for i in range(n_all):
            conf.SetAtomPosition(i, (float(xyz_full[i, 0]), float(xyz_full[i, 1]), float(xyz_full[i, 2])))
        c91_idx = index[lig["c91"]]

        n_heavy = mol.GetNumAtoms()   # heavy count of THIS payload, not a magic 94

        def _solv_clash_fn(xyz_h: np.ndarray, _n: int = n_heavy) -> bool:
            heavy = np.arange(min(len(xyz_h), _n), dtype=int)
            return protein_clash_ok(xyz_h[heavy], prot_for_solv, PROTEIN_LP_MIN)[0]

        best_xyz, sasa_before, sasa_after = solvation_collapse(
            xyz_full, mol_h, c91_idx, clash_fn=_solv_clash_fn, n_random=64,
        )

        # Map back heavy atoms to serials
        moved = {}
        for serial, idx in index.items():
            moved[serial] = best_xyz[idx]
        # Re-clamp C91 to original (solvation_collapse already does this, but belt-and-braces)
        moved[lig["c91"]] = _xyz(lig["lines"][lig["c91"]])

        trial = dict(accepted)
        trial.update({s: np.asarray(v) for s, v in moved.items()})
        trial_adc = load_adc_pdb(pdb_path, lines=apply_coords(adc, trial))
        if measure_pdb(trial_adc)["pass"]:
            accepted = trial
            n_solv += 1
            engine = "solv"

    # ── Step 1 — FOLD each LPP onto the surface (collapse onto the surface) ──
    # Each fold must stay clash-free vs protein AND vs every other LP.
    # Input is the solv-collapsed state (Step 0), not the original extended pose.
    adc_now = load_adc_pdb(pdb_path, lines=apply_coords(adc, accepted)) if accepted else adc
    for rid, lig_now in list(adc_now["ligands"].items()):
        if lig_now.get("c91") is None or "sg_line" not in lig_now:
            continue
        others_now = [
            np.asarray([_xyz(l["lines"][s]) for s in l["serials"]], dtype=float)
            for r, l in adc_now["ligands"].items() if r != rid
        ]
        moved = _fold_ligand(lig_now, adc_now, others_now)
        if moved is not None:
            trial = dict(accepted)
            trial.update(moved)
            trial_adc = load_adc_pdb(pdb_path, lines=apply_coords(adc, trial))
            if measure_pdb(trial_adc)["pass"]:
                accepted = trial
                n_fold += 1
                if engine == "none":
                    engine = "fold"
                else:
                    engine = "solv+fold"

    # Step 2 — MMFF/UFF smooth the accepted (folded) pose.
    # FIX: source the ligand coordinates from `adc_now`, rebuilt here from the
    # FULLY accepted conformation (`accepted` = Step 0 solv-collapse + Step 1
    # surface fold). The previous code iterated `ligs` — the pristine
    # coordinates read straight from the input file — so MMFF relaxation started
    # from the UN-modified pose and its relaxed result was written back over
    # `accepted`, silently discarding every Step-0 / Step-1 move that had
    # already passed QC. (Note: `adc_now` built at the top of Step 1 only
    # reflected Step 0, so we explicitly reload it from `accepted` here to also
    # carry the Step-1 folds.) Connection topology (`adc["conect"]`) is unchanged
    # — only the atom coordinates move — so we keep using `adc["conect"]`.
    adc_now = load_adc_pdb(pdb_path, lines=apply_coords(adc, accepted)) if accepted else adc
    for lig in adc_now["ligands"].values():
        if lig.get("c91") is None or "sg_line" not in lig:
            continue
        moved = minimize_ligand(lig, adc["conect"], max_its=max_its)
        if moved is None:
            continue
        trial = dict(accepted)
        trial.update(moved)
        trial_adc = load_adc_pdb(pdb_path, lines=apply_coords(adc, trial))
        if measure_pdb(trial_adc)["pass"]:
            accepted = trial
            n_mmff += 1
            if engine == "none":
                engine = "MMFF/UFF"
            else:
                engine = "fold+MMFF/UFF"

    after_lines = apply_coords(adc, accepted) if accepted else adc["lines"]
    Path(out_pdb).parent.mkdir(parents=True, exist_ok=True)
    Path(out_pdb).write_text("\n".join(after_lines) + "\n")
    after = measure_pdb(load_adc_pdb(out_pdb))
    return {
        "input": pdb_path,
        "output": out_pdb,
        "engine": engine,
        "n_solv": n_solv,
        "n_fold": n_fold,
        "n_mmff": n_mmff,
        "n_atoms_moved": len(accepted),
        "before": before,
        "after": after,
        "note": (
            f"solv-collapsed {n_solv}; folded {n_fold}; smoothed {n_mmff} LP with MMFF/UFF"
        ),
    }


DEFAULT_REPS = [
    "ADC_DAR1.pdb",
    "ADC_DAR2.pdb",
    "ADC_DAR3.pdb",
    "ADC_DAR4.pdb",
    "ADC_DAR5.pdb",
    "ADC_DAR6.pdb",
    "ADC_DAR7_omit_B229.pdb",
    "ADC_DAR7_omit_B232.pdb",
]


def polish_dir(in_dir: str, out_dir: str | None = None) -> dict:
    in_dir = Path(in_dir)
    out_dir = Path(out_dir) if out_dir else in_dir / "polished"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    results = []
    for name in DEFAULT_REPS:
        src = in_dir / name
        if not src.exists():
            continue
        dst = out_dir / name.replace(".pdb", "_min.pdb")
        rec = polish_one(str(src), str(dst))
        results.append(rec)
        b, a = rec["before"], rec["after"]
        rows.append(
            [
                name,
                b["n_lp"],
                rec.get("n_solv", 0),
                rec.get("n_fold", 0),
                rec.get("n_mmff", 0),
                f"{b['protein_min']:.3f}" if b["protein_min"] is not None else "n/a",
                f"{a['protein_min']:.3f}" if a["protein_min"] is not None else "n/a",
                f"{b['lp_min']:.3f}" if b["lp_min"] is not None else "n/a",
                f"{a['lp_min']:.3f}" if a["lp_min"] is not None else "n/a",
                "PASS" if a["pass"] else "FAIL",
                rec["engine"],
            ]
        )
    report = write_markdown(
        str(out_dir / "POLISH_REPORT.md"),
        "Fold + constrained minimization (polish) — all representative DARs",
        [
            (
                "What this is",
                "Polish has three steps. **Solv** first: each payload collapses toward "
                "minimal non-polar SASA (hydrophobic-driven fold, C91 pinned, "
                "clash-free vs protein only — no inter-LP check). This gives "
                "each MMAE its solution-state initial compactness. **Fold** then "
                "adjusts for inter-LP crowding (clash-free vs protein + all other "
                "payloads). **MMFF/UFF** smooths bond geometries.",
            ),
            (
                "Before → after",
                table(
                    [
                        "file",
                        "DAR",
                        "solv",
                        "fold",
                        "mmff",
                        "prot min before",
                        "prot min after",
                        "LP-LP before",
                        "LP-LP after",
                        "QC",
                        "engine",
                    ],
                    rows,
                ),
            ),
        ],
    )
    payload = {"report": report, "n": len(results), "results": results}
    (out_dir / "polish_summary.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )
    print(json.dumps({"report": report, "n": len(results)}, indent=2))
    return payload


def main(argv=None):
    p = argparse.ArgumentParser(prog="adcsim-polish")
    p.add_argument("--dir", required=True, help="directory with ADC_DAR*.pdb")
    p.add_argument("--out", default=None, help="output directory (default: DIR/polished)")
    args = p.parse_args(argv)
    polish_dir(args.dir, args.out)


# This entry point was missing: polish could only be reached lazily through
# pipeline, so it could not be run standalone or executed from a notebook.
# Added for consistency with the other modules.
if __name__ == "__main__":
    main()
