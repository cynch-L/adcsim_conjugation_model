from __future__ import annotations

from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from scipy.spatial import cKDTree


def load_antibody(pdb_path: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("ab", pdb_path)
    return structure[0]


def heavy_atoms(model, exclude_sg=None):
    """Return (atoms, coords). exclude_sg is (chain, resnum) of the covalent Cys SG."""
    atoms, coords = [], []
    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            for atom in residue:
                if atom.element == "H":
                    continue
                if (
                    exclude_sg is not None
                    and atom.name == "SG"
                    and chain.id == exclude_sg[0]
                    and residue.id[1] == exclude_sg[1]
                ):
                    continue
                atoms.append(atom)
                coords.append(atom.coord)
    return atoms, np.asarray(coords, dtype=float)


def ca_coords(model) -> np.ndarray:
    return np.asarray(
        [
            atom.coord
            for chain in model
            for residue in chain
            for atom in residue
            if atom.name == "CA"
        ],
        dtype=float,
    )


def get_sg(model, site: str) -> np.ndarray:
    chain, res = parse_site(site)
    return np.asarray(model[chain][res]["SG"].coord, dtype=float)


def parse_site(site: str) -> tuple[str, int]:
    """Parse a conjugation-site spec.

    Two spellings coexist in this repo and were not interchangeable:
      · "A:CYS223" — the pipeline / config.yaml style
      · "A-Cys223" — the seqqc / site_env / MD-frame-library style
    Feeding one into the other's code path used to raise ValueError, so accept
    both here and normalize.
    """
    s = site.strip()
    sep = ":" if ":" in s else ("-" if "-" in s else None)
    if sep is None:
        raise ValueError(f"bad site spec {site!r}: expected A:CYS223 or A-Cys223")
    chain, rest = s.split(sep, 1)
    digits = "".join(ch for ch in rest if ch.isdigit())
    if not digits:
        raise ValueError(f"bad site spec {site!r}: no residue number")
    return chain, int(digits)


def load_lp_ensemble(sdf_path: str, remove_hs: bool = False):
    """Load LP conformers.

    MUST default to removeHs=False. The original notebooks address the
    maleimide olefin carbons as RDKit indices 90 and 91 in the 199-atom
    (explicit-H) molecule. Removing Hs reindexes and would silently
    attach the wrong atom.
    """
    mols = [
        mol
        for mol in Chem.SDMolSupplier(sdf_path, removeHs=remove_hs)
        if mol is not None
    ]
    return mols


def mol_heavy_xyz(mol) -> tuple[list[int], np.ndarray]:
    conf = mol.GetConformer()
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    xyz = np.asarray(
        [
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in heavy
        ],
        dtype=float,
    )
    return heavy, xyz


def sg_serial(pdb_path: str, site: str) -> int:
    chain, resnum = parse_site(site)
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain:
                continue
            if line[17:20].strip() != "CYS":
                continue
            if int(line[22:26]) != resnum:
                continue
            if line[12:16].strip() != "SG":
                continue
            return int(line[6:11])
    raise ValueError(f"SG not found for {site} in {pdb_path}")


def load_md_frames(npz_path: str) -> np.ndarray:
    """Read an MD ``frames`` array and return coordinates in ÅNGSTRÖM.

    This project's MD trajectories store coordinates in NANOMETRES (the
    nearest-neighbour spacing of a protein frame is ~0.14 nm = 1.4 Å). We convert
    to Å (×10) unless the file carries an explicit ``units`` key ("nm" or
    "angstrom"). When there is no metadata we fall back to nearest-neighbour
    auto-detection: a median spacing < 0.5 means the file is in nm.

    This is the ONE place unit conversion happens. Both the structural pipeline
    and the site_env trajectory reader call it, so the nm/Å question can never be
    answered inconsistently again. (The example script replace_collapsed.py is a
    standalone module that cannot import this package, so it carries an identical
    inlined copy of the same logic — keep the two in sync.)
    """
    d = np.load(npz_path, allow_pickle=True)
    if "frames" not in d.files:
        raise KeyError(f"{npz_path} has no 'frames' array")
    raw = np.asarray(d["frames"], dtype=np.float64)
    if "units" in d.files:
        units = str(d["units"]).strip().lower()
        scale = 1.0 if units.startswith("a") else 10.0
    else:
        # Auto-detect from the median nearest-neighbour spacing of one frame.
        probe = raw[0]
        if probe.shape[0] < 2:
            scale = 1.0
        else:
            nn = cKDTree(probe).query(probe, k=2)[0][:, 1]
            median_nn = float(np.median(nn))
            scale = 10.0 if median_nn < 0.5 else 1.0
    return (raw * scale).astype(np.float32)


def write_adc_pdb(
    antibody_pdb: str,
    conjugates: list[dict],
    output_pdb: str,
    reactive_atom: int,
    resname: str = "LPP",
):
    """Write antibody + N linker-payloads + CONECT (LP bonds + SG–reactive_atom)."""
    with open(antibody_pdb) as f:
        lines = [x.rstrip("\n") for x in f if not x.startswith("END")]

    used = [
        int(line[6:11])
        for line in lines
        if line.startswith(("ATOM", "HETATM"))
    ]
    next_serial = max(used) + 1
    conect = []

    for lp_idx, item in enumerate(conjugates, start=1):
        site = item["site"]
        pose = item["pose"]
        chain, resnum = parse_site(site)

        sg = sg_serial(antibody_pdb, site)

        mol = pose["payload"]
        coords = pose["coords"]
        heavy_indices = pose["heavy_atom_indices"]

        atom_map = {}
        for pos, atom_idx in enumerate(heavy_indices):
            atom = mol.GetAtomWithIdx(int(atom_idx))
            serial = next_serial
            next_serial += 1
            atom_map[int(atom_idx)] = serial
            x, y, z = coords[pos]
            name = f"{atom.GetSymbol()}{atom_idx}"[:4]
            element = atom.GetSymbol()
            lines.append(
                # resname must be padded to 3 columns, otherwise every
                # following field shifts when the name is shorter than 3 chars
                f"HETATM{serial:5d} {name:<4s} {resname:<3s} A{1000 + lp_idx:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {element:>2s}"
            )

        for bond in mol.GetBonds():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            if a in atom_map and b in atom_map:
                conect.append((atom_map[a], atom_map[b]))

        if reactive_atom not in atom_map:
            raise ValueError(
                f"{site}: reactive_atom {reactive_atom} not in LP heavy atoms"
            )
        conect.append((sg, atom_map[reactive_atom]))

    for a, b in conect:
        lines.append(f"CONECT{a:5d}{b:5d}")
    lines.append("END")

    Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)
    with open(output_pdb, "w") as f:
        f.write("\n".join(lines) + "\n")
    return output_pdb
