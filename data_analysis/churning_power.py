"""
churning_power.py — P1 from the 2026-06-12 analysis plan.

At shutoff (Q ~ 0) the hydraulic power is zero, so shaft power = disk/blade
churning + seal/bearing parasitics. Extracting tau*omega at Q~0 from the five
H-Q ramps therefore measures the churning loss DIRECTLY at five speeds —
the corroboration path for the fitted disk_mult ~ 2.5 (SK-08 discipline box):
if the measured shutoff power sits ~2.5x over Barske's disk correlation
across speeds, the multiplier graduates from "tuned" to "measured".

Outputs: console table + churning_power.png (exploratory; promote into
thesis_figures.py as fig_churning if the result holds).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import epump_io as ep
import thesis_figures as tf   # pump() geometry, MR/MP shaft-only parasitic, LOGDIR

Q_SHUTOFF_LPS = 0.02   # "at shutoff" flow threshold [l/s]; BEP ~ 0.22 l/s
T_RAMP_HI = 22.0       # only the H-Q ramp window (healthy suction; the cav
                       # matrix sweeps NPSH down, which could alter churning)
RPM_MIN = 2000


def barske_disk_power(rpm):
    """Barske empirical disk/churning correlation, pump as-built geometry."""
    p = tf.pump()
    return (1956 * ep.RHO * p.visc ** 0.2 * (np.asarray(rpm, float) / 1000) ** 2.8
            * (p.d_2 ** 4.6 + 4.6 * p.d_1 ** 3.6 * p.b_1))


def shutoff_points():
    rows = []
    for tag, path in ep.find_runs(tf.LOGDIR).items():
        d = ep.load(path)
        t, q, rpm, tq = d["t"], d["q"], d["rpm"], d["tq"]
        spun = rpm[rpm > RPM_MIN]
        if not spun.size:
            continue
        Nmed = np.nanmedian(spun)
        m = ((t <= T_RAMP_HI) & (rpm > 0.85 * Nmed) & (rpm < 1.15 * Nmed)
             & (q < Q_SHUTOFF_LPS) & np.isfinite(tq))
        if m.sum() < 30:   # relax once if the ramp barely touches shutoff
            m = ((t <= T_RAMP_HI) & (rpm > 0.85 * Nmed) & (rpm < 1.15 * Nmed)
                 & (q < 0.03) & np.isfinite(tq))
        if m.sum() < 15:
            print(f"  {tag}: only {m.sum()} shutoff samples - skipped")
            continue
        w = rpm[m] * np.pi / 30
        P = tq[m] * w
        P_med = float(np.nanmedian(P))
        P_sem = 1.253 * float(np.nanstd(P)) / np.sqrt(m.sum())
        N = float(np.nanmedian(rpm[m]))
        P_par = float(np.interp(N, tf.MR, tf.MP))   # measured seal+bearing
        rows.append(dict(tag=tag, rpm=N, n=int(m.sum()),
                         P_shaft=P_med, P_sem=P_sem, P_par=P_par,
                         P_churn=P_med - P_par,
                         P_barske=float(barske_disk_power(N))))
    return sorted(rows, key=lambda r: r["rpm"])


def main():
    rows = shutoff_points()
    if len(rows) < 3:
        print("Not enough shutoff points.")
        return

    N = np.array([r["rpm"] for r in rows])
    Pc = np.array([r["P_churn"] for r in rows])
    Pb = np.array([r["P_barske"] for r in rows])
    sem = np.array([r["P_sem"] for r in rows])

    a, loga0 = np.polyfit(np.log(N), np.log(Pc), 1)
    ratio = Pc / Pb

    print(f"\n{'tag':>16} {'rpm':>6} {'n':>5} {'P_shaft':>8} {'P_par':>6} "
          f"{'P_churn':>8} {'P_barske':>8} {'ratio':>6}")
    for r in rows:
        print(f"{r['tag']:>16} {r['rpm']:6.0f} {r['n']:5d} {r['P_shaft']:8.1f} "
              f"{r['P_par']:6.1f} {r['P_churn']:8.1f} {r['P_barske']:8.1f} "
              f"{r['P_churn']/r['P_barske']:6.2f}")
    print(f"\nfit: P_churn = {np.exp(loga0):.3e} * n^{a:.2f}   "
          f"(Barske exponent 2.8)")
    print(f"ratio to Barske: mean {ratio.mean():.2f}, "
          f"range {ratio.min():.2f}-{ratio.max():.2f}   (fitted disk_mult was {tf.DISK_MULT})")
    # extrapolations to design speed
    print(f"at 20k rpm: power-law fit -> {np.exp(loga0) * 20000 ** a:.0f} W, "
          f"mean-ratio x Barske -> {ratio.mean() * barske_disk_power(20000):.0f} W")

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    nn = np.linspace(N.min() * 0.9, N.max() * 1.15, 100)
    ax.errorbar(N, Pc, yerr=2 * sem, fmt="o", ms=4, color="C0", zorder=5,
                label="measured shutoff churning")
    ax.plot(nn, np.exp(loga0) * nn ** a, "-", color="C0",
            label=rf"fit $P \propto n^{{{a:.2f}}}$")
    ax.plot(nn, barske_disk_power(nn), "--", color="k", label="Barske disk model")
    ax.plot(nn, tf.DISK_MULT * barske_disk_power(nn), ":", color="C3",
            label=rf"${tf.DISK_MULT}\times$ Barske (fitted from $\eta$)")
    ax.plot(nn, np.interp(nn, tf.MR, tf.MP), "-.", color="gray", lw=1,
            label="shaft-only parasitic (meas.)")
    ax.set(xlabel="shaft speed [rpm]", ylabel="power [W]",
           xscale="log", yscale="log")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("churning_power.png", dpi=200)
    print("\nwrote churning_power.png")


if __name__ == "__main__":
    main()
