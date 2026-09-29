# Fold + constrained minimization (polish) — all representative DARs

_Generated 2026-09-23T00:37:11_

## What this is

Polish has three steps. **Solv** first: each payload collapses toward minimal non-polar SASA (hydrophobic-driven fold, C91 pinned, clash-free vs protein only — no inter-LP check). This gives each MMAE its solution-state initial compactness. **Fold** then adjusts for inter-LP crowding (clash-free vs protein + all other payloads). **MMFF/UFF** smooths bond geometries.

## Before → after

| file | DAR | solv | fold | mmff | prot min before | prot min after | LP-LP before | LP-LP after | QC | engine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| _(empty)_ |

