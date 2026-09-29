# adcsim QC report — Trastuzumab × mc-vc-PAB-MMAE

_Generated 2026-09-10T21:02:05_

## What this is

Computational cysteine-conjugation spatial modelling. QC pass means the generated pose satisfies covalent geometry and steric cutoffs. It is **not** experimental DAR, affinity, or cytotoxicity evidence.

## Chemistry

- Reactive atom (RDKit heavy index): **91** (PDB name C91)
- Exit atom: 85
- Allowed S–C window: 1.75–1.9 Å
- Generation bond values: [1.82]
- Protein–LP min: 2.0 Å (Cys-SG of the bonded site excluded)
- LP–LP min: 2.0 Å
- LP conformers used: 100 (stride=1)
- Library rebuilt this run: True

## Chemical accessibility (before DAR1 library)

| site | verdict | Cys-SASA | SG-SASA | open rays | attack rays | best angle | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A:CYS223 | PASS | 35.8 | 1.3 | 16 | 12 | 79 | solvent-accessible thiol with an open approach cone |
| B:CYS223 | PASS | 14.2 | 0.0 | 20 | 20 | 84 | solvent-accessible thiol with an open approach cone |
| A:CYS229 | PASS | 54.2 | 7.7 | 34 | 22 | 70 | solvent-accessible thiol with an open approach cone |
| B:CYS229 | PASS | 36.0 | 10.3 | 25 | 14 | 73 | solvent-accessible thiol with an open approach cone |
| A:CYS232 | PASS | 58.7 | 15.4 | 88 | 66 | 71 | solvent-accessible thiol with an open approach cone |
| B:CYS232 | PASS | 21.6 | 5.1 | 19 | 8 | 72 | solvent-accessible thiol with an open approach cone |
| C:CYS214 | PASS | 14.2 | 0.0 | 15 | 5 | 70 | solvent-accessible thiol with an open approach cone |
| D:CYS214 | PASS | 3.4 | 1.3 | 39 | 9 | 70 | solvent-accessible thiol with an open approach cone |

No site dropped.
SG-SASA = 0 is common for hinge Cys and is not a FAIL by itself.

## Reactive Cys pool

| site | Cys-SASA (Å²) | 4Å neighbours | 0-clash poses (raw) |
| --- | --- | --- | --- |
| A:CYS223 | 35.78 | 9 | 17 |
| B:CYS223 | 14.17 | 8 | 9 |
| A:CYS229 | 54.20 | 10 | 120 |
| B:CYS229 | 36.03 | 9 | 3 |
| A:CYS232 | 58.70 | 6 | 230 |
| B:CYS232 | 21.62 | 6 | 1 |
| C:CYS214 | 14.25 | 14 | 0 |
| D:CYS214 | 3.43 | 9 | 8 |

## DAR funnel

| DAR | states kept | QC | sites | protein-min | LP-LP min | S–C range |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 160 | PASS | A:CYS232 | 3.051 | n/a | 1.820–1.820 |
| 2 | 80 | PASS | A:CYS232, A:CYS229 | 2.494 | 3.158 | 1.820–1.820 |
| 3 | 80 | PASS | A:CYS232, A:CYS229, A:CYS223 | 2.282 | 3.158 | 1.820–1.820 |
| 4 | 80 | PASS | A:CYS232, A:CYS229, A:CYS223, D:CYS214 | 2.278 | 3.158 | 1.820–1.820 |
| 5 | 0 | EMPTY |  |  |  |  |

**Highest QC-passing DAR this run: DAR4**
**Computational ceiling on this backbone: DAR5** (first empty level). Do not loosen clash cutoffs to force more.

## Best pose detail

- DAR1 A:CYS232 side=back S–C=1.820 Å protein-min=3.051 Å
- DAR2 A:CYS232 side=back S–C=1.820 Å protein-min=2.582 Å
- DAR2 A:CYS229 side=back S–C=1.820 Å protein-min=2.494 Å
- DAR3 A:CYS232 side=back S–C=1.820 Å protein-min=2.582 Å
- DAR3 A:CYS229 side=back S–C=1.820 Å protein-min=2.494 Å
- DAR3 A:CYS223 side=back S–C=1.820 Å protein-min=2.282 Å
- DAR4 A:CYS232 side=back S–C=1.820 Å protein-min=2.582 Å
- DAR4 A:CYS229 side=back S–C=1.820 Å protein-min=2.494 Å
- DAR4 A:CYS223 side=back S–C=1.820 Å protein-min=2.282 Å
- DAR4 D:CYS214 side=front S–C=1.820 Å protein-min=2.278 Å

## What this cannot prove

- Experimental conjugation occupancy
- Binding affinity (PLIP / GNINA scores, if added later, are not Kd)
- ADC developability, PK, or cytotoxicity

