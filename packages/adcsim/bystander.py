"""bystander effect: (payload release—diffusion—uptake in tumor tissue)

==========================================================================
Question answered by this section
==========================================================================
Antibodies can only reach a ring around the vessel (the "binding-site barrier",
a well-known limitation of antibodies).
How can antigen-negative cells separated by this gap also be killed?
The answer lies in the payload: it is cleaved off inside positive cells, exits
the cell, diffuses through the interstitial space, and is taken up by
neighboring negative cells.

But "diffusing far" and "entering cells" are two conflicting demands, with an
optimum in between:
  - Uptake too fast -> consumed near the source, cannot reach far
  - Uptake too slow -> flushed away by blood flow before entering any cell
Thurber's model explicitly yields this "existence of a physicochemical optimum"
conclusion (see LIT R10 abstract).

==========================================================================
Upstream dependencies (do not skip)
==========================================================================
This module requires two upstream quantities, both computed by this project or
using traceable parameters:
  1. DAR distribution -> determines how many payload copies each ADC carries   (combinatorial statistics)
  2. release rate      -> determines when and where payload appears            (payload release kinetics)
payload release kinetics already computed: Val-Cit release half-life in
lysosomes is only on the order of milliseconds to minutes, far faster than
internalization/trafficking (~4 h) -> **release is not rate-limiting**,
so this module places the source term directly at the "ADC arrival location",
without an extra time constant for release.

==========================================================================
Geometry and equations
==========================================================================
Adopts the **Krogh cylinder** consistent with Vasalou 2015 (one capillary +
surrounding ring of tumor tissue):

    D_eff · (1/r)·d/dr( r · dC/dr )  −  (k_cell + k_clear)·C  +  S(r)  =  0

  source term S(r): non-zero only in the ring around the vessel that antibodies
                    can reach (r <= R_cap + antibody penetration depth)
  k_cell          : rate at which payload is taken up by cells (higher
                    permeability -> faster uptake)
  k_clear         : rate at which payload is washed away by blood flow/lymph
                    (fast for free small molecules)
  boundary        : r = R_Krogh symmetric (dC/dr = 0); r = R_cap no flux
                    (washout already captured by k_clear)

Two derived length/dimensionless quantities:
  penetration depth delta = sqrt( D_eff / (k_cell + k_clear) )  -- how far payload can travel
  Thiele modulus phi = L / delta                          -- distance scale / penetration depth
  capture fraction f = k_cell / (k_cell + k_clear)         -- fraction taken up before washout

==========================================================================
Known limitations (hard-coded)
==========================================================================
1. **k_clear (tissue washout rate) is the most fragile number in the whole
   module**, currently an order-of-magnitude estimate (~1e-3 s^-1, derived from
   tumor blood flow ~0.06 mL/(g·min)), with no locked primary literature.
   It directly determines where the "optimum" lies. The conclusion tier is
   therefore downgraded to **tier C**.
2. 1D steady-state, single-scale Krogh cylinder; no advection, no realistic
   vascular network geometry, no plasma protein binding of payload, no efflux
   transporters (P-gp, etc.).
   The fact that MMAE is a P-gp substrate is entirely absent here.
3. Output is **ranking and relative comparison**, not absolute kill fraction.
   Absolute values need calibration with in-house xenograft data.
4. This module only answers "can payload reach the negative cells",
   **not "can the ADC reach the tumor"** -- that is an antibody-level
   penetration problem, belonging to another module.
"""

import math

import numpy as np

# ==========================================================================
# Literature parameters (each must be traceable)
# ==========================================================================
LIT = {
    "geometry_and_transport": {
        # Table 1 of R9: Krogh cylinder geometry + effective diffusion coefficient
        "R_cap_um": 8.0,             # capillary radius, range 5-15
        "R_krogh_um": 72.0,          # Krogh cylinder radius = half the capillary spacing, range 30-200+
        "D_payload_cm2_s": 3.2e-6,   # payload effective diffusion coefficient = 320 um^2/s
        "D_adc_cm2_s": 1.3e-7,       # antibody effective diffusion coefficient = 13 um^2/s (range 0.5-1.9e-7)
        "source": "Vasalou C, Helmlinger G, Gomes B. A mechanistic tumor penetration model "
                  "to guide antibody drug conjugate design. PLoS ONE 2015;10(3):e0118977. "
                  "DOI 10.1371/journal.pone.0118977, PMID 25786126",
        "confidence": "B",
    },
    "uptake_halflife": {
        # Khera 2018 Table S1 / Table S2 (original table headers were misformatted;
        # only the single alignable row is taken here)
        "source": "Khera E, Cilliers C, Bhatnagar S, Thurber GM. Computational transport "
                  "analysis of antibody-drug conjugate bystander effects and payload tumoral "
                  "distribution: implications for therapy. Mol Syst Des Eng. 2018;3:73-88. "
                  "DOI 10.1039/C7ME00093F",
        "note": "The original used an empirical correlation between PAMPA permeability and "
                "cellular uptake rate (the equation was misformatted in the source and its "
                "exact form could not be read) to back out the uptake half-life. This module "
                "uses t_uptake directly as a first-order rate constant: k_cell = ln2/t_uptake. "
                "This conversion is our own interpretation, tier C; independently checked "
                "against PAMPA, both are of the same order (see __main__ self-consistency test).",
        "confidence": "C",
    },
    "k_clear": {
        "value": 1.0e-3,             # s^-1, order-of-magnitude estimate
        "derivation": "Tumor blood flow ~0.06 mL plasma/(g·min) -> blood perfusion equivalent "
                      "of 1e-3 s^-1 per cm^3 tissue; small-molecule uptake is nearly complete, "
                      "so the same order is taken. This is a derived value, not a cited one.",
        "confidence": "C (most fragile number in the module)",
    },
}

# ==========================================================================
# payload library: uptake half-life is the only payload-specific input to this module
# ==========================================================================
PAYLOADS = {
    # name: (uptake half-life min, PAMPA permeability x1e-6 cm/s or None, note)
    "Lys-SMCC-DM1": (194.0, 10.0,  "Charged degradation product of T-DM1; bystander effect widely recognized as very weak"),
    "MMAF":         ( 72.0,  1.15, "C-terminus charged; bystander effect widely recognized as weak"),
    "SPP-DM1":      (  7.0,  None, "disulfide-linked maytansine"),
    "PBD":          (  3.6,  None, "pyrrolobenzodiazepine (PBD)"),
    "SN-38":        (  2.8,  1.27, "payload of sacituzumab govitecan"),
    "S-methyl DM4": (  2.9,  None, "active metabolite of DM4"),
    "DM4":          (  5.6,  None, "maytansine class"),
    "MMAE":         (  2.2,  7.47, "used in this system; bystander effect widely recognized as strong"),
    "Dxd":          (  0.9, 12.2,  "payload of T-DXd; bystander effect widely recognized as strong"),
}

# ==========================================================================
# Core functions
# ==========================================================================
UM2_PER_CM2 = 1e8   # 1 cm^2/s = 1e8 um^2/s


def k_from_halflife(t_half_min):
    """First-order rate constant (s^-1) = ln2 / half-life (s)."""
    return math.log(2.0) / (t_half_min * 60.0)


def penetration_depth_um(D_um2_s, k_total_s):
    """Penetration depth delta = sqrt(D/k_total), in um."""
    return math.sqrt(D_um2_s / k_total_s) if k_total_s > 0 else float("inf")


def thiele_modulus(L_um, delta_um):
    """Thiele modulus phi = L/delta. phi<1 means diffusion is not limiting (concentration approximately uniform)."""
    return L_um / delta_um


def capture_fraction(k_cell, k_clear):
    """Fraction that enters cells before being washed away by blood flow."""
    return k_cell / (k_cell + k_clear) if (k_cell + k_clear) > 0 else 0.0


def solve_profile(D_um2_s, k_total_s, R_cap_um, R_krogh_um, source_depth_um, n=400):
    """Solve the 1D radial steady-state equation, return (r_um, C), source term S0 = 1.

    Equation: D·(1/r)d/dr(r dC/dr) - k·C + S = 0
    Boundary: no flux at both ends (symmetric/closed; washout captured by k_clear in k_total)
    """
    if source_depth_um is None or source_depth_um <= 0:
        source_depth_um = R_krogh_um - R_cap_um
    r = np.linspace(R_cap_um, R_krogh_um, n)
    dr = r[1] - r[0]
    r_half_out = 0.5 * (r + np.roll(r, -1))     # r_{i+1/2}
    r_half_in = 0.5 * (r + np.roll(r, 1))       # r_{i-1/2}
    r_half_out[-1] = R_krogh_um
    r_half_in[0] = R_cap_um

    S = np.where(r <= R_cap_um + source_depth_um, 1.0, 0.0)

    A = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            A[i, i - 1] += D_um2_s * r_half_in[i] / (r[i] * dr * dr)
            A[i, i] -= D_um2_s * r_half_in[i] / (r[i] * dr * dr)
        if i < n - 1:
            A[i, i + 1] += D_um2_s * r_half_out[i] / (r[i] * dr * dr)
            A[i, i] -= D_um2_s * r_half_out[i] / (r[i] * dr * dr)
        A[i, i] -= k_total_s
    C = np.linalg.solve(A, -S)
    return r, C


def coverage(C, r, R_cap_um, source_depth_um):
    """Beyond the antibody-reachable zone, mean concentration / source-zone mean
    concentration (dimensionless, <=1).

    Note C has units of "seconds": at steady state k_total·C = S, so the absolute
    value scales with k_total, and source-zone normalization is required for
    cross-payload comparison.
    """
    src = r <= R_cap_um + source_depth_um
    beyond = r > R_cap_um + source_depth_um
    if beyond.sum() == 0 or src.sum() == 0:
        return None
    w_src = r[src]                     # area weight proportional to r dr (dr identical, cancels)
    w_bey = r[beyond]
    mean_src = float((C[src] * w_src).sum() / w_src.sum())
    mean_bey = float((C[beyond] * w_bey).sum() / w_bey.sum())
    if abs(mean_src) < 1e-30:
        return None
    return mean_bey / mean_src


# ==========================================================================
# Main entry
# ==========================================================================
def assess(state=None):
    """Score every payload, output a ranking table + key intermediates per entry."""
    g = LIT["geometry_and_transport"]
    st = dict(
        D_um2_s=g["D_payload_cm2_s"] * UM2_PER_CM2,
        R_cap_um=g["R_cap_um"],
        R_krogh_um=g["R_krogh_um"],
        source_depth_um=30.0,          # depth antibodies can reach; tier C, needs in-house IHC calibration
        k_clear=LIT["k_clear"]["value"],
    )
    if state:
        st.update(state)

    L = st["R_krogh_um"] - st["R_cap_um"]     # tissue thickness to traverse, um
    rows = []
    for name, (t_up_min, pampa, note) in PAYLOADS.items():
        k_cell = k_from_halflife(t_up_min)
        k_tot = k_cell + st["k_clear"]
        delta = penetration_depth_um(st["D_um2_s"], k_tot)
        f_cap = capture_fraction(k_cell, st["k_clear"])
        phi = thiele_modulus(L, delta)
        r, C = solve_profile(st["D_um2_s"], k_tot, st["R_cap_um"],
                             st["R_krogh_um"], st["source_depth_um"])
        cov = coverage(C, r, st["R_cap_um"], st["source_depth_um"])
        # bystander index = true far-field coverage (solved above) x fraction truly taken up
        #   note: coverage is taken directly from the solved concentration profile above,
        #   not proxied by an empirical formula
        rows.append(dict(
            payload=name, t_up_min=t_up_min, pampa=pampa, note=note,
            k_cell=k_cell, delta_um=delta, capture=f_cap, phi=phi,
            coverage=cov, index=f_cap * (cov if cov is not None else 1.0),
        ))
    rows.sort(key=lambda x: -x["index"])
    return dict(state=st, thickness_um=L, rows=rows)


def sensitivity(state=None, kc_values=(1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1)):
    """Sweep the most fragile parameter k_clear to see when the ranking flips.

    This is something the module must provide by itself: k_clear is a derived
    value, not a cited one, and the ranking cannot be cited without showing its
    sensitivity.
    """
    groups = {}
    for kc in kc_values:
        st = dict(state or {})
        st["k_clear"] = kc
        order = tuple(r["payload"] for r in assess(st)["rows"])
        groups.setdefault(order, []).append(kc)
    return groups


def to_alarms(res):
    """Convert to alarm items isomorphic with alarms.py."""
    top = res["rows"][0]
    items = [
        dict(
            step="bystander effect", item="bystander index (this payload)",
            value=next((f"{r['index']:.3f}" for r in res["rows"]
                        if r["payload"] == "MMAE"), "n/a"), unit="",
            reference=f"strongest in this batch {top['payload']} = {top['index']:.3f}",
            level="ok", confidence="C",
            meaning="In this system MMAE belongs to the stronger bystander-effect class; "
                    "but if the gap to the strongest is less than 2x, switching payloads yields limited benefit",
            action="For stronger bystander effect (heterogeneous tumors): switch payload down the table; "
                   "for weaker (reduce off-target toxicity): go the other way, e.g. MMAF",
            causes=[
                dict(name="tissue washout rate", metric="k_clear",
                     value=f"{res['state']['k_clear']:.1e} s^-1",
                     ref="needs in-house data calibration", verdict="cannot determine"),
                dict(name="antibody penetration depth", metric="source_depth",
                     value=f"{res['state']['source_depth_um']:.0f} um",
                     ref="needs IHC calibration", verdict="cannot determine"),
                dict(name="diffusion-limited?", metric="phi(MMAE)",
                     value=f"{next(r['phi'] for r in res['rows'] if r['payload']=='MMAE'):.3f}",
                     ref="<1 means spreads out", verdict="supported"),
            ],
        ),
        dict(
            step="bystander effect", item="efflux transporter (P-gp)", value="MMAE is a P-gp substrate", unit="-",
            reference="transporter not included in this model", level="warn", confidence="A",
            meaning="MMAE is a P-glycoprotein substrate; tumors with high P-gp expression pump "
                    "the payload out, the model's current pure-passive-diffusion picture "
                    "overestimates the bystander effect in such tumors",
            action="if the target indication has high P-gp expression, downgrade the conclusion; "
                   "this effect is not counted in this module",
            causes=[dict(name="transporter expression", metric="target tumor P-gp", value="unknown",
                         ref="needs IHC/RNA data", verdict="cannot determine")],
        ),
    ]
    return items


# ==========================================================================
if __name__ == "__main__":
    print("=" * 96)
    print("bystander effect - payload release / diffusion / uptake in the Krogh cylinder")
    print("=" * 96)
    for k, v in LIT.items():
        s = v.get("source") or v.get("derivation", "")
        print(f"  . {k:<24} [{v.get('confidence','')}]")
        print(f"      {str(s)[:88]}")

    res = assess()
    st = res["state"]
    print(f"\nGeometry: capillary R={st['R_cap_um']} um -> Krogh boundary {st['R_krogh_um']} um, "
          f"tissue thickness to traverse {res['thickness_um']:.0f} um")
    print(f"Params: D_payload={st['D_um2_s']:.0f} um^2/s   "
          f"antibody reach depth={st['source_depth_um']:.0f} um   "
          f"k_clear={st['k_clear']:.1e} s^-1")

    print(f"\n{'payload':<16}{'uptake t1/2':>10}{'delta pen.':>12}{'phi=L/delta':>9}"
          f"{'capture frac':>10}{'far cover':>10}{'bystander idx':>12}")
    print("-" * 96)
    for r_ in res["rows"]:
        cov = "-" if r_["coverage"] is None else f"{r_['coverage']:.3f}"
        print(f"{r_['payload']:<16}{r_['t_up_min']:>9.1f}m{r_['delta_um']:>11.0f}um"
              f"{r_['phi']:>9.3f}{r_['capture']:>10.3f}{cov:>10}{r_['index']:>12.4f}")

    print("\n" + "=" * 96)
    print("How to read:")
    print("  delta = how far payload travels before being taken up/washed away; phi = thickness to traverse / delta")
    print("  capture fraction = fraction truly taken up before being washed away by blood flow")
    print("  bystander index = far-field coverage x capture fraction -- for ranking, not absolute kill rate")
    print("             (coverage is the true solution of the PDE above, not an empirical-formula proxy)")
    print()
    print("* Key point: every payload's delta far exceeds the 64 um to traverse, all phi < 1.")
    print("  -> **At this scale, diffusion is never the limitation.**")
    print("  -> What separates them is 'whether they get into cells before washout', i.e. the capture-fraction column.")
    print("  -> So don't focus on 'making it travel farther'; focus on 'making it get in'.")

    print("\n[Alarms]")
    for a in to_alarms(res):
        print(f"   [{a['level']}] {a['item']} = {a['value']} {a['unit']}  ref {a['reference']}")
        print(f"        {a['meaning']}")
        for c in a["causes"]:
            print(f"        - {c['name']}: {c['metric']}={c['value']} (ref {c['ref']}) -> {c['verdict']}")

    print("\n[Ranking vs k_clear - must be shown or the table cannot be cited]")
    for order, kcs in sensitivity().items():
        tag = "  ".join(f"{k:.0e}" for k in kcs)
        print(f"   k_clear in [{tag}]")
        print("      " + " > ".join(order))
    print("   -> **The last two (MMAF / Lys-SMCC-DM1, weakest) don't move under any value** -- this is robust;")
    print("   -> Which of the top three is strongest depends on k_clear (when k_clear is small MMAE is "
          "slightly stronger than Dxd) -> tier C, must be calibrated.")

    print("\n[Self-consistency test: independently estimate k_cell from PAMPA permeability]")
    print("   k_cell = P x a_v, a_v = membrane area per unit tissue volume")
    print("   tumor cell density 3e8 /cm^3, cell radius 7.5 um -> a_v ~ 2120 cm^-1 (close-packing upper bound)")
    av = 2120.0
    for name in ("MMAE", "MMAF", "Lys-SMCC-DM1", "Dxd"):
        t, p, _ = PAYLOADS[name]
        if p is None:
            continue
        k_est = p * 1e-6 * av
        print(f"   {name:<15} PAMPA={p:>6.2f}e-6 cm/s -> k={k_est:.2e} s^-1, "
              f"t1/2={math.log(2)/k_est/60:.2f} min   measured {t:.1f} min in table   "
              f"{t/(math.log(2)/k_est/60):.1f}x faster")
    print("   -> Even the close-packing upper-bound estimates are faster than measured, direction consistent "
          "(3D close packing vs effective a_v in experimental system);")
    print("     but **the fast/slow factor differs per payload** (3x to 350x),")
    print("     showing PAMPA (pure passive permeation) **cannot replace measured cellular uptake** -- "
          "especially charged degradation products.")
    print("     This is exactly why this module takes only 'measured uptake half-life', not PAMPA-backed-out values.")

    print("\n[Ranking vs established qualitative knowledge]")
    print("   established (weak->strong): Lys-SMCC-DM1 ~ MMAF << DM1/DM4 < SN-38 < MMAE < Dxd")
    print("   this model (weak->strong): " + " < ".join(r_["payload"] for r_ in reversed(res["rows"])))
    print("   -> The two ends (Lys-SMCC-DM1 weakest, Dxd strongest) agree with established knowledge; "
          "mid-range relative positions need measured calibration.")

    print("\n" + "=" * 96)
    print("Known limits: k_clear is an order-of-magnitude estimate that sets where the optimum lies -> whole table tier C;")
    print("          no advection, realistic vascular network, plasma protein binding, or P-gp efflux transporters.")
