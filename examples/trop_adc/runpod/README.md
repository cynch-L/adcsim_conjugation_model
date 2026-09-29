# Hinge remodel (RunPod) + stitch (laptop) + re-run all DAR

Hinge change **invalidates every previous DAR pose**. After a new backbone, adcsim must start at DAR1 again. Do not graft old LP poses onto the stitched antibody.

## What actually OOM'd

`init OK` then `Killed` = Linux OOM. The process died on `pose_from_pdb(full IgG)` (~10220 atoms), before FastRelax. The pod can run PyRosetta; it cannot hold the full antibody pose.

## Split the job

| Where | What | File |
|---|---|---|
| RunPod terminal | extract B217–238, MinMover on 22 residues | `hinge_remodel.py` |
| Laptop (no PyRosetta) | Kabsch-align fragment onto full IgG | `stitch_hinge.py` |
| Laptop | adcsim funnel DAR1→ceiling on **stitched** PDB | `config.hinge.yaml` |

Do **not** give `B217_238_local.pdb` or `B220_235_frag_best.pdb` to adcsim. Those are 22 residues.

## RunPod (terminal, not Jupyter)

Kill the old full-IgG job if it is still around:

```bash
pkill -f hinge_remodel.py || true
```

Copy the **new** `hinge_remodel.py` to `/workspace/trastuzumab_project/`. Prefer the fragment you already extracted:

```bash
cd /workspace/trastuzumab_project
python -u hinge_remodel.py \
  --pdb B217_238_local.pdb \
  --out_dir B220_235_rosetta \
  --n 5 --protocol min
```

If you only have the full PDB, the script extracts the fragment first (text I/O, cheap) then loads the 22-residue pose.

`--protocol min` = torsion wiggle + MinMover (low RAM).  
`--protocol relax` = FastRelax on the fragment only (still small).

Success log must contain `pose residues=22` (or ~22). If you see hundreds of residues, you loaded the full IgG again.

Download only:

```
/workspace/trastuzumab_project/B220_235_rosetta/B220_235_frag_best.pdb
```

## Laptop stitch

```bash
python stitch_hinge.py \
  --full data/trastuzumab_boltz_model_0.pdb \
  --frag B220_235_frag_best.pdb \
  --out  data/trastuzumab_B220_235_stitched.pdb
```

Anchor CA RMSD after fit should be small (≪ 1 Å). Then:

```bash
python run_pipeline.py   # or: python -m adcsim examples/trop_adc/config.hinge.yaml
```

`config.hinge.yaml` writes to `results_hinge/` and sets `reuse_library: false`.

## If RunPod still dies on the fragment

Then the machine is too small even for a 22-residue pose (unusual). Options:

1. `free -h` — if RAM < 8 GB, pick a larger pod.
2. Install MinSizeRel:  
   `python -m pip install pyrosetta --find-links https://west.rosettacommons.org/pyrosetta/quarterly/minsizerel`
3. Google Colab (academic PyRosetta) or a 16–32 GB CPU pod. Full-IgG FastRelax is the thing to avoid, not PyRosetta itself.

adcsim DAR sampling does **not** need PyRosetta and already ran on the laptop.

---

# Polish + cheap score card (last RunPod credit)

Budget ~40–50 min, ~10 min to tune. Runs on a **full-IgG pod** (the one that already
ran full FastRelax at 1326 residues — do NOT use the small fragment pod).

## What it does

1. `molfile_to_params` → `LPP.params` (~5 min)
2. 8 DAR representatives, constrained minimization (~30–40 min)
3. Cheap score card (<1 min)

Constrained min = pin thioether SG–C91 (1.82 Å harmonic) + free all MMAE torsions
+ protein side chains within 8 Å. `fa_atr` pulls MMAE toward the surface = folding,
no extra bias needed. This is **polish, not a filter**; inputs already QC-pass.

## Score card (cheap, reproducible, no pretending)

| metric | what | meaning |
|---|---|---|
| Rg / span / reach | MMAE compactness | smaller = more folded = more ADC-like |
| surface_contact | atoms within 2.4–4.5 Å of protein | does it stick |
| mmff_stress | vacuum MMFF single-point | too-tight fold penalized |
| dG_est | buried-area ΔG (left empty) | needs conformer delta, not faked |

## Run

```bash
bash install_pyrosetta.sh   # if not installed

python polish_and_score_note.py \
  --pdb_dir  /path/to/results_hinge \
  --mmad_sdf /path/to/MC-Val-Cit-PAB-MMAE_3D.sdf \
  --out_dir  /path/to/results_hinge/polished_runpod
```

## Notes

- Input PDB ligand residue names `A1001..A1006` are auto-renamed to `LPP` to match params.
- If one DAR times out, drop `mover.max_iter(120)` to 60–80.
