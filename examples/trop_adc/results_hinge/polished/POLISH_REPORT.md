# Fold + constrained minimization (polish) — all representative DARs

_Generated 2026-09-10T17:42:03_

## What this is

Polish has three steps. **Solv** first: each payload collapses toward minimal non-polar SASA (hydrophobic-driven fold, C91 pinned, clash-free vs protein only — no inter-LP check). This gives each MMAE its solution-state initial compactness. **Fold** then adjusts for inter-LP crowding (clash-free vs protein + all other payloads). **MMFF/UFF** smooths bond geometries.

## Before → after

| file | DAR | solv | fold | mmff | prot min before | prot min after | LP-LP before | LP-LP after | QC | engine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADC_DAR1.pdb | 1 | 1 | 1 | 1 | 3.105 | 3.105 | n/a | n/a | PASS | fold+MMFF/UFF |
| ADC_DAR2.pdb | 2 | 2 | 2 | 0 | 2.421 | 2.364 | 2.426 | 2.426 | PASS | solv+fold |
| ADC_DAR3.pdb | 3 | 3 | 2 | 1 | 2.378 | 2.383 | 2.359 | 2.341 | PASS | fold+MMFF/UFF |
| ADC_DAR4.pdb | 4 | 4 | 2 | 1 | 2.339 | 2.339 | 2.359 | 2.341 | PASS | fold+MMFF/UFF |
| ADC_DAR5.pdb | 5 | 5 | 4 | 0 | 2.272 | 2.272 | 2.343 | 2.343 | PASS | solv+fold |
| ADC_DAR6.pdb | 6 | 6 | 5 | 0 | 2.020 | 2.020 | 2.343 | 2.343 | PASS | solv+fold |
| ADC_DAR7_omit_B229.pdb | 7 | 7 | 5 | 1 | 2.020 | 2.020 | 2.322 | 2.322 | PASS | fold+MMFF/UFF |
| ADC_DAR7_omit_B232.pdb | 7 | 7 | 5 | 0 | 2.007 | 2.007 | 2.062 | 2.040 | PASS | solv+fold |

