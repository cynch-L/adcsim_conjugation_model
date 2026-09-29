# Cys-conjugation playbook — generic decision protocol

A **decision protocol** for steric-feasibility pre-screening of
**IgG1 × maleimide × random inter-chain cysteine conjugation**,
before any wet-lab conjugation.

It abstracts the Trastuzumab × vc-MMAE run into a `new antibody × new
linker × new payload` pipeline. The value is **not** the MMAE numbers —
it is the decision branches that decide *how* to model each new system.

> Scope: steric / geometric ceiling + relative DAR tendency.
> It does NOT predict ΔG of binding, solubility, or absolute conjugation
> yield. Every output is a computational estimate and must be checked
> against HIC / LC-MS.

---

## 0. Inputs → outputs

| in | required |
|---|---|
| antibody 3D | IgG1 (or IgG1-like 4-chain) full structure; if only sequence, fold first |
| linker–payload 3D | one conformer, explicit H, reliable atom ordering (reactive C + exit fixed) |
| conjugation chemistry | maleimide–Cys (Michael addition), reactive olefin carbon index |

| out | meaning |
|---|---|
| 8 sites → usable set | which inter-chain Cys are chemically + sterically approachable |
| DAR ceiling | highest DAR that can be placed clash-free |
| P(DAR) tendency | relative propensity of each DAR, main peak |
| per-DAR representative | polished structures for the scorecard |

---

## 1. Site layer — generic for IgG1, no re-tuning

IgG1 has **4 inter-chain disulfides → 8 reactive Cys** (2 in hinge, 1 per
light chain). Steps:

1. find all Cys SG–SG pairs, split intra (< 2.5 Å) vs inter (< 3.0 Å, H–H or H–L)
2. keep inter-chain only — intra-chain disulfides are structural and not reduced
3. per linear chain glue the *native* hinge/CL Cys (in trastuzumab: HC 223/229/232, LC 214); do **not** scan arbitrary engineered Cys (that is a different, site-specific protocol)

**None of this depends on the payload.** Reuse as-is for any IgG1.

---

## 2. Chemical accessibility gate — generic

Drop sites that cannot be attacked, before any placement:

| check | generic cut | meaning |
|---|---|---|
| Cys SASA | ≥ 5 Å² | sulfhydryl not buried |
| SG open rays | ≥ 4 | solvent reachable cone |
| attack angle | 70–140° (Bürgi–Dunitz ~107°) | maleimide approach geometry |

A buried SG passes site-layer pairing but fails here. Same thresholds for
any maleimide payload.

---

## 3. Payload conformation criterion — **BRANCH 1, re-judge per payload**

This is the first place the MMAE run is *not* transferable.

**Decision input** (per payload): log P, formal charge, rotatable-bond count.

| branch | trigger | model choice |
|---|---|---|
| **a. hydrophobic** | log P high, no charge (MMAE) | solution state is *collapsed-dominant*; build ensemble, keep collapsed-first |
| **b. hydrophilic / charged** | low log P, charged terminus (MMAF), PEG linker | collapse is weak; build *extended-dominant* ensemble + add electrostatics approximation |

**Do not guess** the branch. Run the free-state probe:

- ETKDGv3 `useRandomCoords=True`, ~600 embeds
- measure span / Rg distribution + non-polar SASA per atom
- compare collapsed vs extended: if non-polar SASA drops < ~15 % on
  folding, the collapse driver is weak → branch **b**

The probe converts "is this payload hydrophobic" from an assumption into a
measured number.

### maleimide-end exposure filter — **generic, always on**

Unconstrained collapse can bury the reactive maleimide carbon (C91) inside
the globule — physically unconjugatable, and it produces false negatives.
Filter every conformer: maleimide-ring SASA / extended reference ≥ 0.7.
This applies to hydrophobic *and* hydrophilic payloads.

---

## 4. Linker geometry criterion — **BRANCH 2, re-judge per linker**

| branch | trigger | folding method |
|---|---|---|
| **flexible** | cleavable dipeptide (vc), rotatable bonds ≥ 15 (mc-vc-PAB-MMAE) | fold-by-torsion → collapse feasible |
| **rigid** | non-cleavable short (mc), few rotatable bonds (< 8) | collapse not meaningful; keep mostly extended, skip fold |

The rotatable-bond count is the practical proxy for "does this linker have
a foldable path".

---

## 5. Assembly — three complementary estimators (generic)

1. **beam funnel** → greedy lower bound (what "easily" assembles)
2. **independent recombination** → exhaustive upper bound (ceiling)
3. **sequential MC + hinge flexibility (σ)** → P(DAR) tendency (relative)

The **hinge-breathe σ** is the one non-rigid degree of freedom that decides
high DAR. Set it by a sensitivity sweep (0 / +2 / +4 Å on the hinge SG),
and — ideally — calibrate against MD RMSF of the actual hinge. The DAR
tendency shape (main peak, even-odd, tail) is robust to σ; the exact tail
height is not.

**Even-DAR advantage** drops out of the physics: each A/B cysteine pair
shares one pocket (inter-SG ~2 Å), so "one ball per pair" is geometrically
forced, reproducing the observed even-peak without a fitted bias term.

---

## 6. Polish — apply folding method from Branch 2+3

- collapse-feasible payload → `fold` (torsion crossing, judge final pose
  0-clash only, **not** the path) → `MMFF` (downhill smooth) → re-QC
- rigid linker / charged payload → skip fold, MMFF only

Judgement rule (fixed): a DAR holds if **any** conformer in the
extended↔collapsed interval is 0-clash. Intermediate collisions are not
disqualifying — intermediate states are not candidates.

---

## 7. Scorecard + experimental check — generic

| metric | what it tells |
|---|---|
| Rg / span / reach | collapse degree |
| surface contact | payload–protein burial |
| CDR distance | target-binding interference |
| MMFF stress | local geometry (low resolution — down-weight) |

**Always** close the loop against a real HIC / LC-MS DAR histogram when one
exists for the antibody–payload pair (e.g. Trastuzumab–vc-MMAE: main peak
DAR 4, avg 4.0–4.5, Singh 2016).

---

## 8. Migration checklist — every new system

Run these four before trusting any number:

- [ ] **ensemble rebuilt** for the new payload (Branch 1 probe + MI filter)
- [ ] **linker geometry judged** (Branch 2 rotatable-bond count)
- [ ] **electrostatics decided** — charged payload needs it; pure geometry over-estimates the ceiling
- [ ] **hinge σ swept / MD-calibrated** for high-DAR tail

`trastuzumab × MMAF` passes the *protocol* but fails the *transfer* of
numbers: MMAF is branch 1b (charged, not collapse-driven) + a typically
rigid mc linker (branch 2), so its ceiling must shift and its ensemble must
be rebuilt. The framework is reusable; the parameters are not.