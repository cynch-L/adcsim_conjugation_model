# Provenance

This repository is an audit-driven extraction of a private local notebook
workspace (`<upstream-notebook-workspace>/`, notebooks from 2026-08 to 2026-09).
That workspace is not shipped here; only the artefacts it produced are.

## Path placeholders in archived artefacts

Older run outputs under `examples/trop_adc/results_hinge/` record absolute paths
from the machine that produced them. Those machine-specific prefixes were
replaced with placeholders so the repository can be published:

| Placeholder | Original meaning |
|---|---|
| `<repo>/` | the adcsim-project checkout root |
| `<workspace>/` | the parent working folder holding that checkout |
| `<upstream-notebook-workspace>/` | the private notebook workspace this repo was extracted from |
| `<local-downloads>/` | the machine's downloads folder |

They are documentation markers, not real paths — nothing resolves them at
runtime. Re-running the model writes the current machine's paths again.

## What was kept

- Boltz model `trastuzumab_boltz_model_0.pdb` (validated vs 4HKZ / 3D6G)
- 32 Cys → 16 disulfides → 8 inter-chain sites
- 79-conformer LP ensemble (`payload_ensemble_79.sdf`)
- Front/back hemisphere sampling (15° grid, 30° roll, 2–10 Å ray, 2.0 Å probe)
- Beam search with hard LP–LP and protein–LP cutoffs

## What changed on purpose

| Item | Old notebook | adcsim |
|---|---|---|
| Reactive carbon | C90 in some cells, C91 in docking template | **C91 (RDKit index 91) everywhere** |
| S–C distance | constructed fixed at 1.82 Å, QC tautology | 1.82 Å generation, **measured window 1.75–1.90 Å** |
| Clash exclusion | removed only the bonded SG (inconsistent) | remove bonded SG for protein–LP; LP–LP remains strict; ≥2.0 Å everywhere |
| Notebook kernel state | outputs depended on `globals()` | deterministic `config.yaml` + CLI |
| DAR continuation | forced through DAR6–8 | **stops on empty level**; hinge remodel provides the variant backbone |

## What was not ported (yet)

- PLIP interaction analysis (interaction counts ≠ affinity)
- GNINA (was never actually run locally)
- PyRosetta hinge remodel — **code is packaged in `examples/trop_adc/runpod/`, but must be executed on your licensed RunPod machine**

## Conflicts found in the original workspace

1. **Two coordinate systems**: `trastuzumab_boltz_model_0.pdb` (current) vs old conect PDBs (translated/rotated). adcsim always uses the Boltz frame.
2. **Two DAR generations**: old Kaggle DAR6–DAR8 use C91 but fail a 2.0 Å clash (min 0.75 Å); new docking cells use C90 and stop at DAR6. adcsim resolves the chemistry and reproduces the DAR ceiling.
3. `A223_pass_pose.pkl` was 0 bytes — never trusted; the funnel recreates poses directly.
