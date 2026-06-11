"""Scratch preview: H-Q with the Lock model's own vapour cutoff rendered.

Left: as-built throat D3=3.8mm. Right: effective throat D3=3.0mm.
Theory domain extended past breakdown so the H_3<=H_vp cliff is visible.
Video throat-clear flows marked as vertical dotted lines.
NOT a thesis figure - delete freely.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\Martin\active\FYPTurbine\data_analysis")
sys.path.insert(0, r"C:\Users\Martin\active\FYPTurbine")
import thesis_figures as tf
from uncertainties import unumpy as unp

runs = tf.exp_runs()
clears = {c["run"]: c for c in tf.hq_throat_clear()}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, d3mm in zip(axes, (3.8, 3.0)):
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(runs)))
    for (lbl, e), c in zip(runs.items(), cols):
        H = tf.u_head(e["H"], e["Hsem"])
        ax.plot(e["q"], unp.nominal_values(H), "o", ms=3, color=c,
                label=f"{lbl} ~{e['N']:.0f} rpm")
        qg = np.linspace(1e-6, 0.7e-3, 1200)  # to 0.7 l/s, well past any cutoff
        res = tf.pump().analyse_lock(qg, RPM=e["N"], D_3=d3mm / 1000,
                                     D_inlet=tf.DINLET, K_factor=tf.K_FIT,
                                     eta_losses=0.194, p_inlet=tf.P_INLET)
        Hs = np.asarray(res["H_static"], float)
        broke = np.asarray(res["H_3"], float) <= (3171.0 - tf.P_INLET) / (tf.RHO * tf.G)
        icut = int(np.argmax(broke)) if broke.any() else None
        ax.plot(qg[~broke] * 1000, Hs[~broke], "-", color=c, lw=1.4)
        if icut:
            qcut = qg[icut] * 1000
            ax.plot([qcut, qcut], [Hs[icut - 1], 0], "-", color=c, lw=1.4)
            ax.plot(qg[broke] * 1000, np.zeros(broke.sum()), "-", color=c, lw=1.0, alpha=0.5)
            ax.annotate(f"{qcut:.2f}", (qcut, Hs[icut - 1] * 0.5), fontsize=7,
                        color=c, ha="left", rotation=90)
        if lbl in clears:
            ax.axvline(clears[lbl]["q_clear_lps"], color=c, ls=":", lw=1.0, alpha=0.8)
    ax.set_xlabel("Q [l/s]")
    ax.set_xlim(0, 0.7)
    ax.set_title(f"$D_3$ = {d3mm} mm " + ("(as-built cutwater)" if d3mm == 3.8 else "(effective throat)"))
    ax.grid(alpha=0.25)
axes[0].set_ylabel("H [m]")
axes[0].legend(fontsize=7, loc="lower left")
fig.suptitle("H-Q with Lock's vapour cutoff rendered (solid drop = model H$_3$=H$_{vp}$; dotted = video throat-clear)",
             fontsize=9)
fig.tight_layout()
out = r"C:\Users\Martin\active\FYPTurbine\data_analysis\_scratch_lock_cutoff.png"
fig.savefig(out, dpi=150)
print("saved", out)
for lbl in runs:
    if lbl in clears:
        print(lbl, "video clear:", round(clears[lbl]["q_clear_lps"], 3), "l/s")
