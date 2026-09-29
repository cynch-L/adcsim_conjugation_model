# Inverted funnel — extended MMAE → surface-folded MMAE

Computational steric model only. Not experimental DAR, occupancy, or cytotoxicity.

## What changed

No new vacuum ETKDG. Each already-placed DAR1 pose keeps **C91 pinned** at the S–C bond. Linker / MMAE torsions rotate so the chain can U-turn onto the Fc. Clash cutoffs stay 2.0 Å (protein–LP and LP–LP). Sites are recombined independently (not parent-locked beam).

Compactness is a **range**, not a conformer count:

| site | extended poses | folded poses (pre-dedup) | reach (Å) | Rg (Å) |
|---|---:|---:|---|---|
| A223 | 24 | 107 | 18.8–42.8 | 9.3–12.3 |
| B223 | 24 | 115 | 25.0–42.7 | 8.9–12.4 |
| A229 | 24 | 198 | 19.8–40.4 | 8.3–12.0 |
| B229 | 4 | 15 | 20.5–40.5 | 9.8–11.8 |
| A232 | 24 | 163 | 18.4–40.4 | 9.1–12.0 |
| B232 | 3 | 15 | 22.7–41.4 | 9.9–11.8 |
| C214 | 2 | 10 | 28.8–45.2 | 10.9–13.1 |
| D214 | 9 | 27 | 23.6–42.7 | 10.5–12.4 |

Folded poses outnumber the 79 extended seeds. That is expected: torsion space is continuous.

## Inverted funnel rates

**Pairwise necessary condition** (every site pair has ≥1 clash-free pose pair):

| DAR | extended library | folded library |
|---|---|---|
| 1–7 | 100% except subsets that keep B229+B232 | 100% |
| 8 | **0%** (B229×B232 = 0/12) | **100%** (B229×B232 = 5 pairs, max 2.26 Å) |

Pairwise 100% is **not** formation probability. It only says no pair is a hard block.

**n-body** (one pose per site, all at once, LP–LP ≥ 2.0):

| DAR | folded n-body |
|---|---|
| 8 | **0 / 1 = 0%** (42k nodes from the 5 pairs; 0 hits) |
| 8, both-compact (reach < 28 Å) | **0 pairs** even before filling the other six sites |

The 5 “compatible” B229×B232 pairs all keep **B229 extended (~40 Å)** while B232 is shorter. Those pairs then block A223 / A232. When both B229 and B232 are actually folded, they clash again.

## How to read “formation %”

This run’s percentage is:

`antibody opening × payload compactness × steric cutoff`

It is **not** chemical yield. Independent collapse-then-combine still cannot let two payloads fold *around each other*. Coordinated co-folding of B229+B232, or MD on the Fc, would be the next relaxation — still not wet-lab DAR.

## Files

- `folded_library.pkl` — C91-pinned surface-folded pose library
- `dar8_fold_summary.json` — inverted pairwise rates
- `dar8_from_pairs.json` — fill-from-the-5-pairs search (empty)
- `dar8_compact_pairs.json` — both-compact B229×B232 = 0
