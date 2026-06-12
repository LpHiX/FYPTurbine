"""
plots_round2.py — P2/P3/P4 from the 2026-06-12 analysis plan (exploratory;
promote winners into thesis_figures.py).

P2  eta_re_fit.png         peak overall eta per run vs Re_u, the Reynolds
                           step-up exponent FITTED from our own 5 speeds
                           (fig_eta_extrap currently assumes a=0.1/0.15/0.2),
                           extrapolated to 20k with the fit band.
P3  eta_vs_ns_context.png  efficiency vs nondim specific speed: this work vs
                           every low-ns pump in the lit review. Filled =
                           experimental, open = CFD. The "is 19% bad?" killer.
P4  cav_collapse.png       cavitation coefficient sigma1 = 2g*NPSH3/u1^2 vs
                           flow coefficient phi2 for the 3 usable speeds:
                           if it collapses, NPSHr ~ N^2 similarity holds
                           nondimensionally (3rd collapse after psi-phi,
                           eta-phi). Prints suction specific speeds.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import epump_io as ep
import thesis_figures as tf

NU_W = 1.002e-6                      # water kinematic viscosity [m2/s]
R2 = tf.D2 / 2
D1 = 0.0215                          # blade leading-edge diameter [m]


def re_u(rpm):
    return np.pi * tf.D2 * np.asarray(rpm, float) / 60 * R2 / NU_W


# --------------------------------------------------------------------------- #
# P2 — eta vs Re, exponent fitted from our own runs
# --------------------------------------------------------------------------- #
def peak_eta_per_run():
    """Binned peak overall efficiency per canonical run (same recipe as
    fig_eta_phi: median eta in 16 Q-bins, take the best bin)."""
    out = []
    for lbl, e in tf.exp_runs().items():
        d = e["d"]; t = d["t"]; m = (t >= 0.5) & (t <= 20)
        Q = d["q"][m] / 1000
        H = ep.head_m(d["pout"][m] - d["pin"][m])
        rpm = d["rpm"][m]; w = rpm * 2 * np.pi / 60; tq = d["tq"][m]
        Phyd = tf.RHO * tf.G * Q * H; Psh = tq * w
        good = (rpm > 2000) & (Psh > 0) & (H > 0) & (Q > 0)
        Q, eo = Q[good], (Phyd / Psh)[good]
        qb = np.linspace(0, np.nanpercentile(Q * 1000, 98), 16)
        idx = np.digitize(Q * 1000, qb)
        med = [(np.median(eo[idx == i]), (idx == i).sum())
               for i in range(1, len(qb)) if (idx == i).sum() >= 5]
        if not med:
            continue
        eta_pk = max(m0 for m0, _ in med)
        out.append(dict(lbl=lbl, N=e["N"], eta=eta_pk))
    return sorted(out, key=lambda r: r["N"])


def p2_eta_re():
    rows = peak_eta_per_run()
    N = np.array([r["N"] for r in rows])
    eta = np.array([r["eta"] for r in rows])
    Re = re_u(N)

    # loss-ratio model: (1-eta) = (1-eta_ref) (Re_ref/Re)^a, fitted in log space
    a, c = np.polyfit(np.log(Re), np.log(1 - eta), 1)
    a = -a
    resid = np.log(1 - eta) - (c - a * np.log(Re))
    a_sig = (np.sqrt(np.sum(resid ** 2) / max(len(N) - 2, 1) /
                     np.sum((np.log(Re) - np.log(Re).mean()) ** 2)))

    def eta_of(rpm, ai):
        return 1 - np.exp(c) * re_u(rpm) ** (-ai) * (np.exp(a * np.log(Re).mean()) /
                                                     np.exp(ai * np.log(Re).mean()))

    # cleaner: re-anchor at the top point
    Re0, e0 = Re[-1], eta[-1]
    def eta_extrap(rpm, ai):
        return 1 - (1 - e0) * (Re0 / re_u(rpm)) ** ai

    nn = np.linspace(3000, 21000, 200)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(N, eta * 100, "o", ms=5, color="C0", zorder=5,
            label="measured peak $\\eta$ per run")
    ax.plot(nn, eta_extrap(nn, a) * 100, "-", color="C0",
            label=rf"fit $(1-\eta)\propto Re^{{-{a:.2f}}}$")
    ax.fill_between(nn, eta_extrap(nn, max(a - a_sig, 0)) * 100,
                    eta_extrap(nn, a + a_sig) * 100, color="C0", alpha=0.15,
                    label=rf"$a={a:.2f}\pm{a_sig:.2f}$")
    for ai, lsty in ((0.1, ":"), (0.2, "--")):
        ax.plot(nn, eta_extrap(nn, ai) * 100, lsty, color="gray", lw=1,
                label=rf"assumed $a={ai}$ (old fig)")
    ax.axvline(tf.N_DES, color="k", lw=.5)
    ax.axhline(tf.ETA_DESIGN_PCT, color="gray", ls=":",
               label=f"design ~{tf.ETA_DESIGN_PCT:.0f}%")
    ax.set(xlabel="shaft speed [rpm]", ylabel="overall efficiency [%]",
           ylim=(0, 40))
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("eta_re_fit.png", dpi=200)

    e20_lo = eta_extrap(20000, max(a - a_sig, 0)) * 100
    e20_hi = eta_extrap(20000, a + a_sig) * 100
    print("\nP2 -- eta(Re):")
    for r in rows:
        print(f"  {r['lbl']:>4} {r['N']:6.0f} rpm  Re_u {re_u(r['N']):.2e}  "
              f"peak eta {r['eta']*100:5.1f}%")
    print(f"  fitted exponent a = {a:.3f} +/- {a_sig:.3f} "
          f"(Moody-type 0.1-0.25 expected)")
    print(f"  eta @ 20k rpm = {eta_extrap(20000, a)*100:.1f}%  "
          f"(band {e20_lo:.1f}-{e20_hi:.1f}%)   [prior claim 26-29%]")
    print("  wrote eta_re_fit.png")


# --------------------------------------------------------------------------- #
# P3 — efficiency vs specific speed, literature context
# --------------------------------------------------------------------------- #
def p3_context():
    # (omega_s nondim, eta, label, experimental?, dx, dy) — sources verified in
    # the 2026-06-12 lit passes. omega_s = n_q / 52.9.
    LIT = [
        (0.171, 0.63, "Danieli PE (CFD)",        False, 5, -4),
        (0.205, 0.63, "Danieli FE-1 (CFD)",      False, 5,  2),
        (0.170, 0.52, "Chabannes FE-2 (exp)",    True, -68, -2),
        (0.132, 0.67, "Kim (CFD)",               False, 5,  0),
        (0.089, 0.315, "Knyazeva Barske (CFD)",  False, 5,  5),
        (0.090, 0.30, "Olimstad FE (exp)",       True,  5, -13),
    ]
    ME_WS, ME_ETA = 0.121, 0.19
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for ws, e, lbl, isexp, dx, dy in LIT:
        ax.plot(ws, e * 100, "o" if isexp else "o", ms=7,
                mfc="C0" if isexp else "none", mec="C0")
        ax.annotate(lbl, (ws, e * 100), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7)
    ax.plot(ME_WS, ME_ETA * 100, "*", ms=15, color="C3", zorder=6)
    ax.annotate("this work\n(exp, ~7.5k rpm)", (ME_WS, ME_ETA * 100),
                textcoords="offset points", xytext=(-62, -14), fontsize=7,
                color="C3", ha="center")
    ax.annotate("", xy=(ME_WS, 26.4), xytext=(ME_WS, ME_ETA * 100 + 1.2),
                arrowprops=dict(arrowstyle="->", color="C3", ls="--"))
    ax.annotate("Re-corrected to 20k:\n26.4% (fitted $a$=0.10)", (ME_WS, 26.4),
                textcoords="offset points", xytext=(10, 4), fontsize=7, color="C3")
    # legend proxies
    ax.plot([], [], "o", mfc="C0", mec="C0", label="experimental")
    ax.plot([], [], "o", mfc="none", mec="C0", label="CFD")
    ax.plot([], [], "*", ms=11, color="C3", label="this work")
    ax.text(0.97, 0.04,
            "also measured, $\\omega_s$ unpublished:\n"
            "Barske 1960: 30$\\to$60%   Snell: 50-60%\nShao cryo: 35-45%",
            transform=ax.transAxes, fontsize=6.5, ha="right",
            bbox=dict(fc="white", ec="0.8"))
    ax.set(xlabel=r"specific speed $\omega_s$ [-]",
           ylabel="overall/hydraulic efficiency [%]",
           xlim=(0.06, 0.24), ylim=(0, 80))
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig("eta_vs_ns_context.png", dpi=200)
    print("\nP3 -- wrote eta_vs_ns_context.png "
          "(check label positions; CFD-vs-exp gap is the visual argument)")


# --------------------------------------------------------------------------- #
# P4 — nondimensional cavitation collapse
# --------------------------------------------------------------------------- #
def p4_cav_collapse():
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    print("\nP4 -- cavitation collapse:")
    all_nss = []
    for lbl in tf.CAV_RUNS:
        e = tf.exp_runs()[lbl]
        cav = ep.analyse_cav(e["tag"], e["d"])
        N = cav["rpm"]
        u1 = np.pi * D1 * N / 60
        u2 = np.pi * tf.D2 * N / 60
        pts = [(s["q_ref"], s["npshr"], s.get("nss", np.nan)) for s in cav["steps"]
               if s.get("broke") and np.isfinite(s["npshr"]) and s["npshr"] > 0]
        if not pts:
            continue
        q, npshr, nss = map(np.array, zip(*pts))
        phi2 = (q / 1000) / (np.pi * tf.D2 * tf.B2 * u2)
        sig1 = 2 * tf.G * npshr / u1 ** 2
        ax.plot(phi2, sig1, "o", ms=5, label=f"{lbl} ~{N:.0f} rpm")
        all_nss += list(nss[np.isfinite(nss)])
        for qq, nn, ss in pts:
            print(f"  {lbl:>4} {N:5.0f} rpm  Q {qq:.3f} l/s  NPSH3 {nn:5.2f} m  "
                  f"nss {ss:6.0f}" if np.isfinite(ss) else
                  f"  {lbl:>4} {N:5.0f} rpm  Q {qq:.3f} l/s  NPSH3 {nn:5.2f} m")
    if all_nss:
        print(f"  suction specific speed (metric nq-form) at breakdown: "
              f"median {np.median(all_nss):.0f}, range "
              f"{min(all_nss):.0f}-{max(all_nss):.0f}")
    ax.set(xlabel=r"flow coefficient $\phi_2$",
           ylabel=r"$\sigma_1 = 2g\,\mathrm{NPSH_3}/u_1^2$")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("cav_collapse.png", dpi=200)
    print("  wrote cav_collapse.png (collapse = NPSHr~N^2 similarity holds)")


if __name__ == "__main__":
    try:
        tf.use_style()
    except Exception:
        pass
    p2_eta_re()
    p3_context()
    p4_cav_collapse()
