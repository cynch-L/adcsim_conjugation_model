Put (or symlink) these files here before running:

- `trastuzumab_boltz_model_0.pdb` — Boltz-1 full IgG (original backbone)
- `B220_235_rosetta_best.pdb` — PyRosetta FastRelax of B220–235 on the same frame (2026-09-08 RunPod). Current `config.yaml` input.
- `payload_ensemble_79.sdf` — 79 ETKDG/MMFF conformers of mc-vc-PAB-MMAE
- `MC-Val-Cit-PAB-MMAE_3D.sdf` — original 3D ligand (optional; used only for chemistry checks)

In the original workspace the PDB/SDF were symlinks into a private local
directory (`<upstream-notebook-workspace>/`).
`B220_235_rosetta_best.pdb` is a real copy (also archived at
`<upstream-notebook-workspace>/data/antibody/hinge_remodel/`).
For GitHub, copy the files (or Git LFS) instead of committing machine-specific symlinks.

Do not use `B217_238_local.pdb` (22-residue fragment) as `antibody_pdb`.
