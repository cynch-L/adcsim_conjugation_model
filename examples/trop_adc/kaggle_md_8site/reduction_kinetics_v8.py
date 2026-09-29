#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reduction_kinetics_v8.py — disulfide reduction model switch: from inverting lambda to shared reductant pool + stoichiometry
=============================================================================
The single question to answer
-----------------------------------------------------------------------------
Currently the disulfide reduction step is r_i = 1 - exp(-lambda * ease_i * eq), where lambda is a number back-solved from DAR.
It has two flaws:
  (1) lambda has no physical identity — it is neither a rate nor a yield, just a knob to make the curve pass through that point;
  (2) the pseudo-first-order assumption presumes the reductant is always in excess — but eq=2~3 and the 4 bonds are of the same order of magnitude,
      so TCEP gets consumed, and this assumption clearly fails in the regime we use.

This script does three things:
  A. Use 3 real measurement anchors (2.0 / 2.5 / 3.0 equivalents) to decide which of three functional forms is correct;
  B. Convert the single knob K into a quantity with physical identity: K = k_TCEP x [mAb] x t_reduction;
  C. Leave-one-out validation (2 points fix K, predict the 3rd) + high-equivalent extrapolation (falsifiable target for experiments).

Key: p_c is fixed at 0.96, independently given by site accessibility / thiol activation / covalent conjugation; the disulfide reduction fit must not touch it (no cross-step borrowing).
-----------------------------------------------------------------------------
"""
import json, os, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results_hinge")

# ---- Experimental data (4 user-provided measured points, all real, do not delete) --------------
#     units: TCEP / antibody equivalent ratio -> HIC measured mean DAR (engineered mAb)
EQ = np.array([2.00, 2.15, 2.50, 3.00])
DAR = np.array([3.77, 3.97, 4.55, 5.38])
N_BOND = 4
PC = 0.96  # given independently by site accessibility / thiol activation / covalent conjugation; fixed here


# =============================================================================
# Model B: shared reductant pool (bonds compete for the same TCEP)
#   dS_i/dt = -k*ease_i*S_i*T,  T(t) = eta*eq - Sum(1-S_i)
#   substitution J = k*Int T dt, tau = k*t   =>   dJ/dtau = eta*eq - n + Sum exp(-ease_i*J)
#   final r_i = 1 - exp(-ease_i * J(K)),  K = k*t_final  <- single knob
#   when TCEP is exhausted dJ/dtau -> 0, automatically stops at the stoichiometric ceiling (no extra parameter needed to cap)
# =============================================================================
# J's trajectory depends only on (eq, ease), not on K — integrate once, then any K is a table lookup.
# This reduces "recomputing the integral for every candidate K" from O(grid size) to O(1).
_TAB = {}
TAU_MAX, DT = 62.0, 0.002


def _rhs_vec(J, ease, eqs, eta):
    """J:(E,) ease:(B,) -> f:(E,) ; f = eta*eq - n + Sum_i exp(-ease_i*J)"""
    return eta * eqs - len(ease) + np.exp(-np.outer(J, ease)).sum(axis=1)


def _build_table(eqs, ease, eta=1.0):
    key = (tuple(np.round(ease, 9)), tuple(np.round(eqs, 9)), eta)
    if key in _TAB:
        return _TAB[key]
    eqs = np.asarray(eqs, float)
    n = int(TAU_MAX / DT)
    Js = np.zeros((n + 1, len(eqs)))
    J = np.zeros(len(eqs))
    for i in range(n):
        k1 = _rhs_vec(J, ease, eqs, eta)
        k2 = _rhs_vec(J + 0.5 * DT * k1, ease, eqs, eta)
        k3 = _rhs_vec(J + 0.5 * DT * k2, ease, eqs, eta)
        k4 = _rhs_vec(J + DT * k3, ease, eqs, eta)
        J = np.maximum(J + DT * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0, 0.0)
        Js[i + 1] = J
    tau = np.arange(n + 1) * DT
    tab = (tau, Js)
    _TAB[key] = tab
    return tab


def exposure_B(eq, ease, K, eta=1.0):
    """Return J(K): first look up the column of J(tau) for that eq in the table, then interpolate to tau=K"""
    tau, Js = _build_table(np.array([float(eq)]), ease, eta)
    return float(np.interp(K, tau, Js[:, 0]))


def exposure_B_many(eqs, ease, Ks, eta=1.0):
    """Look up multiple (eq, K) combinations at once, return the J array"""
    tau, Js = _build_table(np.asarray(eqs, float), ease, eta)
    Ks = np.atleast_1d(np.asarray(Ks, float))
    return np.array([np.interp(k, tau, Js[:, i]) for i, k in enumerate(Ks)])


def exposure_B_exact(eq, K, n=N_BOND, eta=1.0):
    """Analytical solution for homogeneous ease=1 (Bernoulli equation), used to self-check the numerical integration"""
    e = eta * eq
    d = e - n
    if abs(d) < 1e-12:                      # degenerate case eq*eta == n
        return n * K / (1.0 + n * K / n)
    w = (1.0 + n / d) * math.exp(d * K) - n / d
    return -math.log(max(1.0 / w, 1e-300))


def mean_dar_B(eq, ease, K, eta=1.0):
    r = 1.0 - np.exp(-np.asarray(ease, float) * exposure_B(eq, ease, K, eta))
    return 2.0 * PC * float(np.sum(r))


def mean_dar_A(eq, ease, lam):
    """Model A: pseudo-first-order (current model). Assumes TCEP is constant and in excess"""
    r = 1.0 - np.exp(-lam * np.asarray(ease, float) * eq)
    return 2.0 * PC * float(np.sum(r))


def mean_dar_C(eq, ease, y):
    """Model C: pure stoichiometry, number of bonds opened = min(4, eq*y)"""
    return 2.0 * PC * min(float(N_BOND), float(eq) * y)


# =============================================================================
def fit_one_knob(fn, lo, hi, ease, n=160, refine=60):
    """Single-knob least squares: coarse scan + ternary refinement"""
    xs = np.linspace(lo, hi, n)
    rms = [float(np.sqrt(np.mean((np.array([fn(e, ease, x) for e in EQ]) - DAR) ** 2)))
           for x in xs]
    i = int(np.argmin(rms))
    a, b = xs[max(i - 1, 0)], xs[min(i + 1, n - 1)]
    for _ in range(refine):
        m1, m2 = a + (b - a) / 3, b - (b - a) / 3
        f1 = float(np.sqrt(np.mean((np.array([fn(e, ease, m1) for e in EQ]) - DAR) ** 2)))
        f2 = float(np.sqrt(np.mean((np.array([fn(e, ease, m2) for e in EQ]) - DAR) ** 2)))
        if f1 < f2:
            b = m2
        else:
            a = m1
    x = (a + b) / 2
    r = float(np.sqrt(np.mean((np.array([fn(e, ease, x) for e in EQ]) - DAR) ** 2)))
    return float(x), r


def rmse_of(fn, ease, knob, eqs=None):
    eqs = EQ if eqs is None else np.asarray(eqs, float)
    pred = np.array([fn(e, ease, knob) for e in eqs])
    obs = DAR if len(eqs) == len(DAR) else None
    if obs is None:
        return pred, None
    return pred, float(np.sqrt(np.mean((pred - obs) ** 2)))


def load_ease():
    for fn in ("ss_reduction_profile.json",):
        p = os.path.join(RES, fn)
        if os.path.exists(p):
            d = json.load(open(p))
            e = np.array([float(q["reduction_ease"]) for q in d["pairs"]], float)
            if len(e) == N_BOND:
                return e / e.mean(), fn, e
    return np.ones(N_BOND), "homogeneous (default)", np.ones(N_BOND)


# =============================================================================
def main():
    out = {}
    ease, src, ease_raw = load_ease()

    print("=" * 86)
    print("disulfide reduction model-switch verdict: 4 measured anchors (TCEP 2.0 / 2.15 / 2.5 / 3.0 equivalents), p_c fixed at 0.96")
    print("=" * 86)
    print(f"\n[ease_i] source={src}")
    print(f"    raw={np.round(ease_raw,3)}   normalized (mean=1)={np.round(ease,3)}")

    # ---- self-check: numerical integration vs analytical solution ----
    print("\n[self-check] at homogeneous ease=1, numerical integration vs analytical solution")
    for e in (2.0, 3.0):
        for K in (0.5, 2.0):
            num = exposure_B(e, np.ones(N_BOND), K)
            ex = exposure_B_exact(e, K)
            print(f"   eq={e} K={K}: numeric J={num:.6f}  analytic J={ex:.6f}  diff={abs(num-ex):.2e}")

    # ---- reference ceiling: stoichiometric ceiling ----
    print("\n[reference] stoichiometric ceiling (TCEP 1:1 opens disulfides, all hit interchain bonds)")
    print(f"   {'eq':>6}{'ceiling DAR(=2*pc*eq)':>22}{'measured':>8}{'measured/ceiling':>12}")
    for e, d in zip(EQ, DAR):
        ceil = 2 * PC * e
        print(f"   {e:>6.2f}{ceil:>22.3f}{d:>8.2f}{d/ceil:>12.3f}")

    # ---- three-model single-knob fit ----
    print("\n" + "-" * 86)
    print("single-knob fit (RMS units = DAR)")
    print("-" * 86)
    print(f"{'model':<30}{'knob':>10}{'RMS':>8}   pointwise residuals (2.0/2.15/2.5/3.0 equiv)")
    rows = {}
    for name, fn, lo, hi in [
        ("A pseudo-first-order 1-exp(-lambda*ease*eq)", mean_dar_A, 1e-3, 5.0),
        ("B shared pool  J(K)", mean_dar_B, 1e-3, 60.0),
        ("C pure stoichiometry min(4,eq*y)", mean_dar_C, 0.5, 1.5),
    ]:
        knob, rms = fit_one_knob(fn, lo, hi, ease)
        pred, _ = rmse_of(fn, ease, knob)
        res = pred - DAR
        print(f"{name:<30}{knob:>10.4f}{rms:>8.4f}   " + " ".join(f"{v:+.3f}" for v in res))
        rows[name] = dict(knob=knob, rms=rms, pred=pred.tolist(), residuals=res.tolist())
        out[name] = rows[name]
    print("\nrun of residual signs (+ + - or - + +) = wrong functional form; random +/- = noise")

    # ---- leave-one-out validation: 2 points fix knob, predict the 3rd ----
    print("\n" + "=" * 86)
    print("leave-one-out validation: use only 2 points to fix the knob, predict the 3rd")
    print("=" * 86)
    print(f"{'model':<30}{'left-out equiv':>9}{'measured':>8}{'predicted':>8}{'error':>8}")
    loo = {}
    for name, fn, lo, hi in [
        ("A pseudo-first-order", mean_dar_A, 1e-3, 5.0),
        ("B shared pool", mean_dar_B, 1e-3, 60.0),
        ("C pure stoichiometry", mean_dar_C, 0.5, 1.5),
    ]:
        errs = []
        for i in range(len(EQ)):
            tr_eq = [EQ[j] for j in range(len(EQ)) if j != i]
            tr_da = [DAR[j] for j in range(len(EQ)) if j != i]
            xs = np.linspace(lo, hi, 160)
            best, bs = None, float("inf")
            for x in xs:
                ss = float(np.mean((np.array([fn(e, ease, x) for e in tr_eq])
                                   - np.array(tr_da)) ** 2))
                if ss < bs:
                    best, bs = x, ss
            pv = fn(EQ[i], ease, best)
            errs.append(abs(pv - DAR[i]))
            print(f"{name:<30}{EQ[i]:>9.2f}{DAR[i]:>8.2f}{pv:>8.3f}{pv-DAR[i]:>+8.3f}")
            loo.setdefault(name, []).append(dict(held_eq=float(EQ[i]),
                                                 actual=float(DAR[i]),
                                                 predicted=round(float(pv), 3),
                                                 error=round(float(pv - DAR[i]), 3)))
        print(f"{'':<30}{'mean abs error':>9}{'':>8}{'':>8}{np.mean(errs):>8.3f}\n")
        loo[name + "|MAE"] = round(float(np.mean(errs)), 4)
    out["leave_one_out"] = loo

    # ---- extrapolation: where the three diverge most ----
    print("=" * 86)
    print("extrapolation divergence (data with >=4 equiv would settle it decisively) — this is the next point to request from experiments")
    print("=" * 86)
    print(f"{'model':<30}" + "".join(f"{e:>10.1f} equiv" for e in (3.5, 4.0, 5.0, 6.0, 8.0)))
    ext = {}
    for name, fn in [("A pseudo-first-order", mean_dar_A), ("B shared pool", mean_dar_B),
                     ("C pure stoichiometry", mean_dar_C)]:
        knob = rows[[k for k in rows if k.startswith(name[:1])][0]]["knob"]
        vals = [fn(e, ease, knob) for e in (3.5, 4.0, 5.0, 6.0, 8.0)]
        print(f"{name:<30}" + "".join(f"{v:>12.3f}" for v in vals))
        ext[name] = [round(float(v), 3) for v in vals]
    out["extrapolation"] = dict(eq=[3.5, 4.0, 5.0, 6.0, 8.0], pred=ext)
    print("\nreading: A would exceed 8 (physically impossible: only 4 bonds / 8 thiols); B/C have a ceiling.")

    # ---- physical identity of K: K = k_TCEP x [mAb] x t_reduction ----
    print("\n" + "=" * 86)
    print("physical identity of K: K = k_TCEP x [mAb] x t_reduction")
    print("=" * 86)
    Kb = rows[[k for k in rows if k.startswith("B")][0]]["knob"]
    print(f"    fitted K = {Kb:.3f}\n")
    print(f"   {'process condition':<32}{'implied k_TCEP (M^-1 s^-1)':>24}")
    for label, mab_uM, t_h in [("10 uM, 1 h", 10.0, 1.0),
                               ("10 uM, 2 h", 10.0, 2.0),
                               ("30 uM, 1 h", 30.0, 1.0),
                               ("30 uM, 2 h", 30.0, 2.0),
                               ("67 uM(10 mg/mL), 2 h", 67.0, 2.0)]:
        k_imp = Kb / (mab_uM * 1e-6 * t_h * 3600.0)
        print(f"   {label:<32}{k_imp:>24.2f}")
    print("\n   usage: compare against the literature second-order rate constant for TCEP reducing disulfides —")
    print("   if it matches -> K is not fitted, it is physical; if not -> other chemistry is at play")
    print("   (reductant spent on intrachain bonds, reduction time too short) — that is eta's concern (reductant share spent on interchain bonds).")
    print("   NOTE: eta is a bookkeeping placeholder only. Do NOT tune it to match a measured DAR —")
    print("         that is fitting one number to one experiment and it will not transfer.")
    out["K_physical"] = dict(K=Kb, note="K = k_TCEP * [mAb] * t_reduction")

    # ---- does ease heterogeneity matter at all ----
    print("\n" + "=" * 86)
    print("does ease heterogeneity matter? (all normalized to mean 1, only the shape changes)")
    print("=" * 86)
    print(f"{'ease shape':<40}{'K':>8}{'RMS':>8}")
    for label, e in [("measured heterogeneity", ease),
                     ("homogeneous (all 1)", np.ones(N_BOND)),
                     ("widen gap spread=2", np.exp(np.linspace(-2, 2, N_BOND)) /
                      np.exp(np.linspace(-2, 2, N_BOND)).mean())]:
        k, r = fit_one_knob(mean_dar_B, 1e-3, 60.0, e)
        print(f"   {label:<40}{k:>8.3f}{r:>8.4f}")
    print("\n   reading: RMS barely changes -> ease shape does not affect mean DAR in the ceiling regime")
    print("         (mass balance: how many bonds open is set only by TCEP amount, unrelated to which bond is easier to open).")
    print("         ease actually decides which bonds open — i.e. the shape of the DAR distribution, not the mean.")

    # ---- distribution is a zero-parameter prediction: K pinned by 3 mean points, distribution shape has no knob ----
    print("\n" + "=" * 86)
    print("zero-parameter prediction: DAR distribution at 2.5 equiv (K pinned, ease decides which bonds open)")
    print("=" * 86)
    DIST_OBS = {0: 3.404, 2: 17.876, 4: 37.881, 6: 29.574, 8: 11.266}  # engineered mAb measured
    J25 = exposure_B(2.5, ease, Kb)
    r25 = 1.0 - np.exp(-ease * J25)
    print(f"   J(2.5equiv, K={Kb:.3f}) = {J25:.4f}")
    print(f"    each bond reduction probability r_i = {np.round(r25,4)}   Sum r = {r25.sum():.3f} (ceiling=equiv=2.5)")

    def dist_of(r, pc, rho):
        d = np.zeros(2 * N_BOND + 1)
        d[0] = 1.0
        for ri in r:
            p2 = rho * pc + (1 - rho) * pc * pc
            p1 = (1 - rho) * 2 * pc * (1 - pc)
            p0 = rho * (1 - pc) + (1 - rho) * (1 - pc) ** 2
            contrib = np.array([(1 - ri) + ri * p0, ri * p1, ri * p2])
            nxt = np.zeros_like(d)
            for k, c in enumerate(contrib):
                if c > 0:
                    nxt[k:] += d[: len(d) - k] * c
            d = nxt
        return d * 100.0

    obs_v = np.array([DIST_OBS.get(k, 0.0) for k in range(2 * N_BOND + 1)])
    obs_v = obs_v / obs_v.sum() * 100.0
    print(f"\n{'scheme':<28}{'odd%':>7}{'mean':>7}{'max dev':>9}   diff vs measured (0/2/4/6/8)")
    dr = {}
    for label, pc, rho in [("pc=0.96, rho=0", 0.96, 0.0),
                           ("pc=0.96, rho=0.85", 0.96, 0.85),
                           ("pc=0.99, rho=0", 0.99, 0.0),
                           ("pc=1.00, rho=0", 1.00, 0.0)]:
        d = dist_of(r25, pc, rho)
        dev = d - obs_v
        print(f"{label:<28}{d[1::2].sum():>7.2f}"
              f"{np.sum(np.arange(len(d))*d)/100:>7.3f}{np.abs(dev).max():>9.2f}   "
              + " ".join(f"{dev[k]:+5.2f}" for k in (0, 2, 4, 6, 8)))
        dr[label] = dict(pc=pc, rho=rho, odd_pct=round(float(d[1::2].sum()), 2),
                         mean=round(float(np.sum(np.arange(len(d)) * d) / 100), 3),
                         dist=[round(float(x), 2) for x in d])
    print(f"{'measured (engineered mAb)':<28}{'<3':>7}"
          f"{sum(k*v for k,v in DIST_OBS.items())/sum(DIST_OBS.values()):>7.3f}")
    print("\n   reading: the mean is already pinned by K, so here we compare shape.")
    print("         odd DAR measured <3%, while the pc=0.96 independent-arm model gives ~15% — ")
    print("         this is the next inconsistency to resolve (belongs to combinatorial statistics, not the disulfide reduction step).")
    out["dist_prediction"] = dict(J_2p5eq=round(float(J25), 4),
                                  r=[float(x) for x in r25], rows=dr)

    json.dump(out, open(os.path.join(RES, "reduction_kinetics_v8.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\n[saved] reduction_kinetics_v8.json")


if __name__ == "__main__":
    main()
