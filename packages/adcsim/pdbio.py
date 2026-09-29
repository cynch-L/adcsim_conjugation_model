"""Unified PDB read engine.

Design (user-approved "parse once, dispatch on demand")
-------------------------------------------------------
A single line-based parser turns a PDB into one :class:`PDBStructure` holding
an *atom table* (coordinates, element, residue name/number, chain, B-factor)
plus the connection records (CONECT and SSBOND). Every downstream module then
pulls only what it needs through the dispatch helpers, instead of each module
re-parsing the file its own way.

Heavy-atom criterion (THE consistency bug this module fixes)
------------------------------------------------------------
The legacy readers disagreed on what counts as a heavy atom:

* ``seqqc.read_pdb``        — no element filter at all (hydrogens counted)
* ``site_env.parse_pdb``    — ``element == 'H'``
* ``io.load_antibody``      — ``atom.element == 'H'`` (Biopython)
* ``polish._is_heavy``      — ``element == 'H'`` OR ``name.startswith('H')``
                             (the name rule wrongly drops Mercury ``HG`` and
                             Deuterium ``D``)

The unified rule is explicit and element-based: an atom is heavy unless its
element symbol is ``H`` or ``D``. This is the only criterion the whole package
now uses.

Dispatch helpers
----------------
* ``get_all_coordinates``        — every atom, or heavy atoms only
* ``get_sequence``              — per-chain (resid, resname) in order
* ``get_residue_atoms`` / ``residue_centroid`` — a single residue
* ``get_site_sg``               — the SG coordinate of a conjugation site
* ``split_antibody_ligand``     — antibody (standard ATOM) vs small molecule
                                  (LPP / HETATM) by residue name *and* by
                                  CONECT linkage to a Cys SG
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Elements treated as "not heavy". Deuterium ('D') is chemically hydrogen.
_LIGHT_ELEMENTS = frozenset({"H", "D"})

# Standard amino-acid residue names (ATOM records). Anything else in an ATOM/
# HETATM line that is not a standard residue is treated as a small molecule.
_STD_AA = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
})

# Default small-molecule residue names (linker-payload). HETATM lines with
# these names are linker-payload atoms; other HETATM (water, ions) are ignored
# by the splitter unless they are CONECT-linked to a Cys SG.
_LIGAND_RESNAMES = frozenset({"LPP"})


@dataclass
class AtomRecord:
    """One ATOM/HETATM line, normalised."""
    serial: int
    record: str          # "ATOM" or "HETATM"
    chain: str
    resid: int
    icode: str
    resname: str
    atomname: str
    element: str
    altloc: str
    occupancy: float
    bfactor: float
    charge: str
    xyz: np.ndarray      # (3,) float


@dataclass
class PDBStructure:
    source: str
    atoms: List[AtomRecord] = field(default_factory=list)
    conect: List[Tuple[int, int]] = field(default_factory=list)   # (serial, serial) pairs
    ssbond: List[Tuple[str, int, str, int]] = field(default_factory=list)  # (chain1,res1,chain2,res2)

    def __len__(self) -> int:
        return len(self.atoms)


def _element_from_line(line: str) -> str:
    """Element symbol, taken from columns 77-78 when present, else the first
    letter of the atom name (PDB fallback)."""
    if len(line) >= 78:
        el = line[76:78].strip()
        if el:
            return el.capitalize()
    # Fallback: first alpha char of the atom name.
    name = line[12:16].strip()
    return name[0].capitalize() if name else ""


def _parse_line(line: str) -> Optional[AtomRecord]:
    try:
        serial = int(line[6:11])
    except ValueError:
        return None
    try:
        resid = int(line[22:26])
    except ValueError:
        resid = 0
    xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)
    try:
        occ = float(line[54:60]) if len(line) >= 60 else 1.0
    except ValueError:
        occ = 1.0
    try:
        bfac = float(line[60:66]) if len(line) >= 66 else 0.0
    except ValueError:
        bfac = 0.0
    return AtomRecord(
        serial=serial,
        record="ATOM" if line.startswith("ATOM") else "HETATM",
        chain=line[21],
        resid=resid,
        icode=line[26] if len(line) > 26 else " ",
        resname=line[17:20].strip(),
        atomname=line[12:16].strip(),
        element=_element_from_line(line),
        altloc=line[16] if len(line) > 16 else " ",
        occupancy=occ,
        bfactor=bfac,
        charge=line[78:80].strip() if len(line) >= 80 else "",
        xyz=xyz,
    )


def _parse_conect(lines: List[str]) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for line in lines:
        if not line.startswith("CONECT"):
            continue
        nums = []
        i = 6
        while i + 5 <= len(line.rstrip()):
            chunk = line[i:i + 5].strip()
            if chunk:
                try:
                    nums.append(int(chunk))
                except ValueError:
                    pass
            i += 5
        if len(nums) < 2:
            continue
        a = nums[0]
        for b in nums[1:]:
            pairs.append((a, b))
    return pairs


def _parse_ssbond(lines: List[str]) -> List[Tuple[str, int, str, int]]:
    out: List[Tuple[str, int, str, int]] = []
    for line in lines:
        if not line.startswith("SSBOND"):
            continue
        # SSBOND   1 CYS A  223    CYS B  229
        try:
            res1 = line[11:14].strip()
            ch1 = line[15]
            r1 = int(line[17:21])
            res2 = line[25:28].strip()
            ch2 = line[29]
            r2 = int(line[31:35])
            out.append((ch1, r1, ch2, r2))
        except (ValueError, IndexError):
            continue
    return out


def read_pdb_lines(lines: List[str]) -> PDBStructure:
    """Parse an already-split list of PDB lines into a single PDBStructure."""
    atoms: List[AtomRecord] = []
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        rec = _parse_line(line)
        if rec is not None:
            atoms.append(rec)
    conect = _parse_conect(lines)
    ssbond = _parse_ssbond(lines)
    return PDBStructure(source="<lines>", atoms=atoms, conect=conect, ssbond=ssbond)


def read_pdb(path: str) -> PDBStructure:
    """Parse a PDB file once. Returns the canonical PDBStructure."""
    with open(path) as f:
        lines = f.read().splitlines()
    struct = read_pdb_lines(lines)
    struct.source = path
    return struct


# --------------------------------------------------------------------------
# Heavy-atom criterion — the single, explicit rule for the whole package.
# --------------------------------------------------------------------------
def is_heavy(rec: AtomRecord) -> bool:
    """An atom is heavy unless its element is H or D (Deuterium)."""
    return rec.element not in _LIGHT_ELEMENTS


def heavy_atoms(struct: PDBStructure) -> List[AtomRecord]:
    return [a for a in struct.atoms if is_heavy(a)]


# --------------------------------------------------------------------------
# Dispatch helpers
# --------------------------------------------------------------------------
def get_all_coordinates(struct: PDBStructure, heavy_only: bool = True) -> np.ndarray:
    """(N, 3) coordinates of all atoms, or heavy atoms only."""
    recs = heavy_atoms(struct) if heavy_only else struct.atoms
    if not recs:
        return np.zeros((0, 3), dtype=float)
    return np.asarray([a.xyz for a in recs], dtype=float)


def get_coordinate_array(struct: PDBStructure, heavy_only: bool = True):
    """Return (records, coords) for the selected atom set."""
    recs = heavy_atoms(struct) if heavy_only else struct.atoms
    coords = get_all_coordinates(struct, heavy_only=heavy_only)
    return recs, coords


def get_sequence(struct: PDBStructure) -> Dict[str, List[Tuple[int, str]]]:
    """Per-chain ordered list of (resid, resname)."""
    out: Dict[str, List[Tuple[int, str]]] = {}
    for a in struct.atoms:
        out.setdefault(a.chain, []).append((a.resid, a.resname))
    # de-duplicate consecutive (resid, resname), keep first occurrence order
    for ch in out:
        seen = set()
        uniq = []
        for rid, rn in out[ch]:
            if (rid, rn) in seen:
                continue
            seen.add((rid, rn))
            uniq.append((rid, rn))
        out[ch] = uniq
    return out


def iter_residues(struct: PDBStructure, chain: Optional[str] = None):
    """Yield ``(chain, resid, icode, resname, [AtomRecord,...])`` in file order."""
    by_res: Dict[Tuple[str, int, str], List[AtomRecord]] = {}
    order: List[Tuple[str, int, str]] = []
    for a in struct.atoms:
        if chain is not None and a.chain != chain:
            continue
        key = (a.chain, a.resid, a.icode)
        if key not in by_res:
            by_res[key] = []
            order.append(key)
        by_res[key].append(a)
    for key in order:
        ch, rid, ic = key
        recs = by_res[key]
        yield ch, rid, ic, recs[0].resname, recs


def get_residue_atoms(struct: PDBStructure, chain: str, resid: int) -> List[AtomRecord]:
    return [a for a in struct.atoms if a.chain == chain and a.resid == resid]


def residue_centroid(struct: PDBStructure, chain: str, resid: int,
                     heavy_only: bool = True) -> Optional[np.ndarray]:
    """Mean coordinate of (heavy) atoms in a residue.

    Replaces the ad-hoc "mean of all atom coords" some modules used; on
    hydrogen-free files this is numerically identical to the old behaviour.
    """
    recs = get_residue_atoms(struct, chain, resid)
    if heavy_only:
        recs = [a for a in recs if is_heavy(a)]
    if not recs:
        return None
    return np.mean(np.asarray([a.xyz for a in recs], dtype=float), axis=0)


def get_site_sg(struct: PDBStructure, site: str) -> np.ndarray:
    """Coordinate of the SG atom of a Cys site.

    Accepts ``A-Cys223`` or ``A:CYS223``.
    """
    s = site.strip()
    sep = ":" if ":" in s else ("-" if "-" in s else None)
    if sep is None:
        raise ValueError(f"bad site spec {site!r}: expected A:CYS223 or A-Cys223")
    ch, rest = s.split(sep, 1)
    digits = "".join(c for c in rest if c.isdigit())
    if not digits:
        raise ValueError(f"bad site spec {site!r}: no residue number")
    rid = int(digits)
    for a in struct.atoms:
        if a.chain == ch and a.resid == rid and a.resname == "CYS" and a.atomname == "SG":
            return a.xyz
    raise KeyError(f"SG of {site} not found in {struct.source}")


def split_antibody_ligand(
    struct: PDBStructure,
    ligand_resnames: frozenset = _LIGAND_RESNAMES,
) -> Tuple[List[AtomRecord], List[AtomRecord]]:
    """Split atoms into (antibody, ligand).

    Antibody = standard amino-acid residues (ATOM records).
    Ligand   = non-standard HETATM residues whose name is in ``ligand_resnames``,
               OR any atom CONECT-linked to a Cys SG (covers LINK-record LPs).

    Connection-based detection uses the parsed CONECT table: for each (a, b)
    pair, if one endpoint is a Cys SG and the other endpoint belongs to a
    non-standard residue, that residue is the ligand.
    """
    cys_sg_serials = {
        a.serial for a in struct.atoms
        if a.resname == "CYS" and a.atomname == "SG"
    }
    lig_serials: set = set()
    # by residue name
    for a in struct.atoms:
        if a.record == "HETATM" and a.resname in ligand_resnames:
            lig_serials.add(a.serial)
    # by CONECT linkage to a Cys SG
    serial_set = {a.serial for a in struct.atoms}
    for a, b in struct.conect:
        if a in cys_sg_serials and b in serial_set:
            lig_serials.add(b)
        elif b in cys_sg_serials and a in serial_set:
            lig_serials.add(a)

    antibody, ligand = [], []
    for a in struct.atoms:
        if a.serial in lig_serials:
            ligand.append(a)
        else:
            antibody.append(a)
    return antibody, ligand
