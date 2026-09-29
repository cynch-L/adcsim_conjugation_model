# ADC conjugation process — consolidated alarm report

_Generated 2026-09-23T15:47:36_

_Sources: structure `examples/trop_adc/data/trastuzumab_boltz_model_0.pdb`, model output `examples/trop_adc/results_hinge/dar_v6_complete.json`_

## How to read this report

This report collects every warning the pipeline can raise about an antibody–drug conjugation run, grouped by the step of the process it concerns.  Each alarm answers three questions: **what** is off, **how we judge it** (value vs. reference range), and **what to do** about it.

- **OK** — within the expected range.
- **WARN** — drifting out of specification; review before scaling up.
- **CRIT** — out of specification; the batch is at risk.
- **INFO** — worth noting, threshold not triggered.

Process stages in order: sequence QC → disulfide reduction → site accessibility → thiol activation → covalent conjugation → combinatorial statistics → linker-payload QC.

## Overall verdict

- **Overall status:** CRIT
- **Critical alarms:** 2    **Warning alarms:** 6

**Alarms that need attention:**
  - [CRIT] sequence QC — conformation confidence at conjugation sites = 48.9 pLDDT (reference )
  - [WARN] thiol activation — B-Cys223 local charge deviation = 0.5064 pKa units (reference ±0.5)
  - [WARN] thiol activation — B-Cys229 local charge deviation = 0.6684 pKa units (reference ±0.5)
  - [WARN] thiol activation — C-Cys214 local charge deviation = 0.8574 pKa units (reference ±0.5)
  - [WARN] thiol activation — D-Cys214 local charge deviation = 0.6653 pKa units (reference ±0.5)
  - [CRIT] linker-payload QC — Working concentration / aqueous solubility = 15.3 x (>1 means the part exceeding solubility) (reference ≤ 2 (placeholder threshold, needs calibration with your own DLS/solubility data))
  - [WARN] linker-payload QC — Computed partition coefficient logP = 5.08 (Crippen) (reference ≤ 5 as a warning reference (needs calibration with your own data))
  - [WARN] combinatorial statistics — High DAR (≥6) fraction = 42.42 % (reference ≤ 25.0)

## sequence QC

| Check | Value | Level | Reference | What it means | What to do |
| --- | --- | --- | --- | --- | --- |
| conjugatable site count | 8 sites | OK |  | 4 interchain disulfides -> 8 conjugatable thiols, consistent with the model's premise | No action needed |
| site list matches MD frame library | True | OK |  | Conjugatable sites inferred from sequence vs. sites in the existing MD frame library | Inconsistency means downstream simulations ran on the wrong sites |
| heavy-chain isotype | IgG1 | OK |  | Hinge motif matches IgG1, consistent with the model's IgG1-only scope | No action needed |
| Fc N-glycosylation sequon (chain A) | 1 sites | OK |  | Located at [300]; conservative glycosylation, the model's default case | No action needed |
| Fc N-glycosylation sequon (chain B) | 1 sites | OK |  | Located at [300]; conservative glycosylation, the model's default case | No action needed |
| degradation liability far from conjugation site | 22 sites | OK |  | Present but not adjacent to a conjugation site; limited impact on conjugation itself | No action needed |
| conformation confidence at conjugation sites | 48.9 pLDDT | CRIT |  | Mean local pLDDT across the 8 conjugatable sites = 48.9 (A-Cys223:54.0, A-Cys229:42.7, A-Cys232:42.0, B-Cys223:49.1, B-Cys229:43.0, B-Cys232:43.1, C-Cys214:59.0, D-Cys214:58.3) | Lower pLDDT means the structure is more uncertain; conformational sampling must perturb the low-confidence regions more strongly |

## disulfide reduction

| Check | Value | Level | Reference | What it means | What to do |
| --- | --- | --- | --- | --- | --- |
| Mass conservation (opened bonds ≤ reductant molar equivalents) | 2.304 bonds (cap 2.5) | OK | ≤ 2.5 | Opened bonds exceed the reductant molar amount = TCEP-disulfide 1:1 conservation is violated | **This is a model bug, not a process problem** — check the code; do not tune the process |
| Reductant utilization (opened bonds ÷ equivalents) | 0.922 — | OK | ≥ 0.85 | Reductant wasted: not going to interchain bonds | Check TCEP freshness / whether intrachain bonds consumed it / raise equivalents or extend time |

## site accessibility

| Check | Value | Level | Reference | What it means | What to do |
| --- | --- | --- | --- | --- | --- |
| Effective exposure time of the most buried site | 1541.6 s | OK | ≥ 300.0 | This site is essentially buried inside the protein | This site is unsuitable as a conjugation site; or pick a mutant with a more open conformation |
| Site exposure time, max / min | 3.31 fold | OK | ≤ 10.0 | Site heterogeneity is extreme, so the product will inevitably be inhomogeneous | Unrelated to process parameters; it is a property of the antibody itself / site selection |

## thiol activation

| Check | Value | Level | Reference | What it means | What to do |
| --- | --- | --- | --- | --- | --- |
| A-Cys223 ΔpKa sensitive to dielectric constant | 0.6109 pKa units | INFO | < 0.3 | With epsilon_eff ranging from 20 to 80 the conclusion changes, indicating this site sits very close to charged groups | Do not cite its absolute pKa; use it only for relative comparison between sites. |
| A-Cys229 ΔpKa sensitive to dielectric constant | 0.6881 pKa units | INFO | < 0.3 | With epsilon_eff ranging from 20 to 80 the conclusion changes, indicating this site sits very close to charged groups | Do not cite its absolute pKa; use it only for relative comparison between sites. |
| B-Cys223 local charge deviation | 0.5064 pKa units | WARN | ±0.5 | Net charge around the site is significant; negative charge raises pKa, activation slows down | This site's reactivity will differ from the others; if site-occupancy spectra are measured, prioritize comparing against this one. |
| B-Cys223 ΔpKa sensitive to dielectric constant | 0.7595 pKa units | INFO | < 0.3 | With epsilon_eff ranging from 20 to 80 the conclusion changes, indicating this site sits very close to charged groups | Do not cite its absolute pKa; use it only for relative comparison between sites. |
| B-Cys229 local charge deviation | 0.6684 pKa units | WARN | ±0.5 | Net charge around the site is significant; negative charge raises pKa, activation slows down | This site's reactivity will differ from the others; if site-occupancy spectra are measured, prioritize comparing against this one. |
| B-Cys229 ΔpKa sensitive to dielectric constant | 1.0027 pKa units | INFO | < 0.3 | With epsilon_eff ranging from 20 to 80 the conclusion changes, indicating this site sits very close to charged groups | Do not cite its absolute pKa; use it only for relative comparison between sites. |
| C-Cys214 local charge deviation | 0.8574 pKa units | WARN | ±0.5 | Net charge around the site is significant; negative charge raises pKa, activation slows down | This site's reactivity will differ from the others; if site-occupancy spectra are measured, prioritize comparing against this one. |
| C-Cys214 ΔpKa sensitive to dielectric constant | 1.286 pKa units | INFO | < 0.3 | With epsilon_eff ranging from 20 to 80 the conclusion changes, indicating this site sits very close to charged groups | Do not cite its absolute pKa; use it only for relative comparison between sites. |
| D-Cys214 local charge deviation | 0.6653 pKa units | WARN | ±0.5 | Net charge around the site is significant; negative charge raises pKa, activation slows down | This site's reactivity will differ from the others; if site-occupancy spectra are measured, prioritize comparing against this one. |
| D-Cys214 ΔpKa sensitive to dielectric constant | 0.9979 pKa units | INFO | < 0.3 | With epsilon_eff ranging from 20 to 80 the conclusion changes, indicating this site sits very close to charged groups | Do not cite its absolute pKa; use it only for relative comparison between sites. |
| Thiolate fraction | 3.065 % | OK | ≥ 1.0 | Too few activated thiols; the reaction cannot get going | Raise pH (or check whether pKa is abnormally high) |
| pH within process window | 7.0 — | OK | 6.5–8.5 | Too low and the reaction stalls; too high and payload hydrolysis and side reactions accelerate | Bring pH back into the window |

## covalent conjugation

| Check | Value | Level | Reference | What it means | What to do |
| --- | --- | --- | --- | --- | --- |
| Conjugation probability | 0.995 — | OK | ≥ 0.98 | Mathematically guarantees a large fraction of odd-numbered DAR (bond opened but only one payload attached) | Check payload freshness / pH / hydrolysis / site sterics |
| payload margin (charge equivalents − thiols opened) | 1.39 equiv (need 4.61) | OK | ≥ 0.7 | payload is fed sub-stoichiometrically; when the margin is too thin it becomes the new limiting reagent | Raise the payload charge in proportion to equivalents; this is a feed-ratio design issue, not a conjugation-probability issue |

## combinatorial statistics

| Check | Value | Level | Reference | What it means | What to do |
| --- | --- | --- | --- | --- | --- |
| Odd-numbered DAR fraction | 2.25 % | OK | ≤ 5.0 | A bond opened but only one payload attached — can only be the conjugation step, unrelated to reduction | This is the fastest attribution: first check payload freshness and feed |
| High DAR (≥6) fraction | 42.42 % | WARN | ≤ 25.0 | High DAR correlates with aggregation propensity and toxicity risk | Directly lower the equivalents — this is the only effective knob; no other adjustment needed |
| Low DAR (≤1) fraction | 3.33 % | OK | ≤ 10.0 | Too much naked antibody, which competes with the drug for the target | Read the diagnostic table: low opened bonds → check reduction; enough opened bonds → check conjugation |
| DAR distribution standard deviation | 1.965 — | OK | ≤ 2.0 | Product is too heterogeneous | Unrelated to process parameters; caused by excessive site-to-site variation |

**Evidence chain — High DAR (≥6) fraction**

| Candidate cause | Mechanism metric | Value | Reference | Verdict |
| --- | --- | --- | --- | --- |
| too many reduction equivalents fed | opened bonds ÷ equivalents | 0.9218 | ≤ 0.85 | undetermined |
| conjugation too complete | odd-numbered DAR fraction | 2.25 | ≥ 1% | excluded |
| site accessibility out of control | site exposure-time ratio | 3.31 | ≤ 10× | undetermined |

## linker-payload QC

| Check | Value | Level | Reference | What it means | What to do |
| --- | --- | --- | --- | --- | --- |
| Reactive-end availability | 1.0 (population fraction) | OK | ≥ 0.7 | fraction of conformers where the payload buries its maleimide end inside its own hydrophobic core in aqueous phase; the lower it is, the more molecules must first unfold before they can react | raise the organic co-solvent fraction (cap 10–20% v/v) / switch to a more hydrophilic linker / use DLS or solubility measurements to confirm whether clustering truly occurs |
| Exposed nonpolar area (intermolecular aggregation driving force) | 198.7 Å² | OK | ≤ 200 (placeholder threshold, must be calibrated with your own DLS/solubility data) | the hydrophobic area a single molecule presents outward when spread in solution; the larger it is, the more readily several molecules stick together. **This is the intermolecular-aggregation driving force, not intramolecular self-burial** — the two mechanisms differ, do not conflate them. | raise the organic co-solvent fraction (cap 10–20% v/v), lower the payload concentration, add a surfactant (e.g. PS20), or use DLS to look at the particle-size distribution directly |
| Working concentration / aqueous solubility | 15.3 x (>1 means the part exceeding solubility) | CRIT | ≤ 2 (placeholder threshold, needs calibration with your own DLS/solubility data) | working concentration 402 µM, most-optimistic estimated aqueous solubility only 26.3 µM (melting point unknown, liquid/amorphous assumption, already the upper bound). The excess exists as aggregates or precipitate, **does not participate in conjugation** —— this is the mechanism behind "feed ratio was sufficient yet it will not conjugate", not freshness nor pH. | raise the organic co-solvent fraction (cap 10–20% v/v) / lower the payload feed concentration and use a longer time / switch to a more hydrophilic linker / use DLS to inspect particle size or turbidity, or post-centrifugation supernatant concentration to confirm how much actually dissolved |
| Computed partition coefficient logP | 5.08 (Crippen) | WARN | ≤ 5 as a warning reference (needs calibration with your own data) | the higher the logP, the more it tends to self-aggregate in aqueous phase rather than spread out | when logP > 5, always check the reactive-end availability, not just the feed ratio |

## Data integrity & what was checked

Which checks actually ran, and why any did not.  Missing data is never invented — a skipped stage is reported as skipped.

- **sequence QC:** ok (7 alarms)
- **thiol activation:** ok (10 alarms)
- **linker-payload QC:** ok (4 alarms)
- **combinatorial / process checks (check_all):** ok (12 alarms)

**Fields fed to the process check (check_all):**
  - `dist` = list[9] e.g. [0.0316, 0.0017, 0.1733] …
  - `mean_dar` = 4.586
  - `dar_std` = 1.965
  - `tcep_eq` = 2.5
  - `bonds_opened` = 2.3045
  - `t_eff_s` = [4568.5, 1541.6, 5100.1, 1541.6, 3261.1, 3261.1, 3261.1, 3261.1]
  - `thiolate_pct` = 3.065
  - `ph` = 7.0
  - `p_conj` = 0.995
  - `payload_feed_ratio` = 6.0

## Inputs used

- Structure file: `examples/trop_adc/data/trastuzumab_boltz_model_0.pdb`
- Model output JSON: `examples/trop_adc/results_hinge/dar_v6_complete.json`
- MD frame library: `(none — static structure used)`
- Linker-payload structure: `(default ensemble)`

