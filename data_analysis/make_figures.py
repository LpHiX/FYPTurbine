"""
make_figures.py — campaign QA: per-run overview time-series + numeric summary.

SUPERSEDED for thesis figures by thesis_figures.py (2026-06-10), which
build_thesis.py --figs now calls. Keep this for raw-campaign QA/provenance
(the per-run overviews and AI_analysis_summary.md), not for thesis output.

Imports the SAME analysis core as the interactive notebook (epump_io.py) so the
figures in the thesis are byte-for-byte reproducible from raw H5 with one run:

    python make_figures.py

Writes PNGs + AI_analysis_summary.md to OUT. This is the deterministic
"how did you get this number" artefact — keep it in sync with the campaign you
are submitting.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import epump_io as ep

# --------------------------------------------------------------------------- #
# Campaign config — edit per submission
# --------------------------------------------------------------------------- #
LOGDIR = r"D:\Projects\propbackend_logs\2026-06-02\hotfirelog"
OUT = r"D:\Projects\propbackend_logs\2026-06-02\analysis"
os.makedirs(OUT, exist_ok=True)

# per-run annotation: the ONE place you log run metadata for this campaign.
# esc = intended ESC throttle %, hq = has a discharge-valve H-Q ramp to analyse.
# Keep in sync with the notebook's RUNS dict.
RUNS = {
    # cavitation-only (no H-Q ramp)
    "20260602_142543": dict(esc=25, hq=False, note="cav only; motor demand ~30, ~3480 rpm"),
    "20260602_143646": dict(esc=25, hq=False, note="cav only, ~2460 rpm"),
    # full H-Q + cavitation, full-length valid runs
    "20260602_150821": dict(esc=30, hq=True),
    "20260602_152153": dict(esc=35, hq=True),
    "20260602_153853": dict(esc=40, hq=True),
    "20260602_155535": dict(esc=45, hq=True),
    "20260602_162513": dict(esc=45, hq=True),
    "20260602_163135": dict(esc=50, hq=True),
    "20260602_164329": dict(esc=55, hq=True),
    "20260602_153306": dict(esc=40, hq=True, note="only ~85s — check it captured the full ramp"),
    # short / aborted full-test attempts (excluded from H-Q)
    "20260602_150429": dict(esc=25, hq=False, note="aborted ~20s, no rpm"),
    "20260602_150507": dict(esc=25, hq=False, note="aborted ~16s, no rpm"),
    "20260602_150530": dict(esc=30, hq=False, note="aborted ~11s"),
    "20260602_175102": dict(esc=30, hq=False, note="short ~29s"),
    "20260602_182637": dict(esc=30, hq=False, note="short ~27s"),
    "20260602_182738": dict(esc=30, hq=False, note="short ~17s"),
    "20260602_182820": dict(esc=30, hq=False, note="short ~14s"),
    # dead-head motor ramps (no flow sweep -> not an H-Q ramp)
    "20260602_182852": dict(esc=None, hq=False, note="dead-head motor ramp 30-85"),
    "20260602_183317": dict(esc=None, hq=False, note="dead-head motor ramp 30-85 att2"),
    "20260602_183935": dict(esc=None, hq=False, note="dead-head motor ramp 30-85 att2"),
}
HQ_RUNS = {tag for tag, m in RUNS.items() if m.get("hq")}

# --------------------------------------------------------------------------- #
data = ep.load_runs(LOGDIR)

# 1) Overview time-series per run
for tag, d in data.items():
    fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    ax[0].plot(d["t"], ep.smooth(d["rpm"]), color="#1f77b4"); ax[0].set_ylabel("RPM")
    ax[0].set_title(f"{tag}  |  {d['notes']}")
    ax[1].plot(d["t"], d["pin"], label="pt_in", color="#d62728")
    ax[1].plot(d["t"], d["pout"], label="pt_out", color="#2ca02c")
    ax[1].plot(d["t"], d["ppump"], label="pt_pump (rails @3bar)", color="#7f7f7f", lw=0.8)
    ax[1].axhline(0, color="k", lw=0.4); ax[1].set_ylabel("bar(g)"); ax[1].legend(loc="upper right", fontsize=8)
    ax[2].plot(d["t"], ep.smooth(d["q"]), color="#ff7f0e"); ax[2].set_ylabel("Q (l/s)")
    ax[3].plot(d["t"], d["out_dem"], label="outlet dem", color="#17becf")
    ax[3].plot(d["t"], d["in_dem"], label="inlet dem", color="#e377c2")
    ax[3].set_ylabel("servo deg"); ax[3].set_xlabel("t (s)"); ax[3].legend(loc="upper right", fontsize=8)
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"overview_{tag}.png"), dpi=110)
    plt.close(fig)

# 2) H-Q curves overlaid
fig, ax = plt.subplots(figsize=(9, 6))
hq_summary = {}
for tag in sorted(HQ_RUNS):
    if tag not in data:
        continue
    r = ep.analyse_hq(tag, data[tag])
    if r is None:
        continue
    hq_summary[tag] = r
    line, = ax.plot(r["qbin"], r["Hbin"], "-o", lw=2, ms=4,
                    label=f"{tag}  ~{r['rpm']:.0f} rpm  (H0={r['H_shutoff']:.1f} m, "
                          f"Qrunout={r['q_runout']:.3f} l/s)")
    ax.scatter(r["qn"], r["Hn"], s=4, alpha=0.12, color=line.get_color())
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("Flow rate Q (l/s, affinity-normalised to Nref)")
ax.set_ylabel("Developed head H = (pt_out - pt_in) (m)")
ax.set_title("Pump H-Q characteristic (affinity-normalised; H~N^2, Q~N)\n"
             "pt_out = true pump discharge; curve crosses 0 at runout flow")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "HQ_curves.png"), dpi=120); plt.close(fig)

# 2b) Shutoff head vs N^2
figv, axv = plt.subplots(figsize=(7, 5.5))
Ns = np.array([hq_summary[t]["rpm"] for t in hq_summary])
H0 = np.array([hq_summary[t]["H_shutoff"] for t in hq_summary])
if len(Ns):
    axv.scatter(Ns ** 2, H0, s=60, zorder=3, color="#d62728")
    for t in hq_summary:
        axv.annotate(t, (hq_summary[t]["rpm"] ** 2, hq_summary[t]["H_shutoff"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=8)
    if len(Ns) >= 2:
        k = np.sum(H0) / np.sum(Ns ** 2)
        xx = np.linspace(0, (Ns.max() * 1.05) ** 2, 50)
        axv.plot(xx, k * xx, "--", color="#1f77b4",
                 label=f"H0 = {k:.3e}*N^2  (head coeff const)")
    axv.set_xlabel("N^2 (rpm^2)"); axv.set_ylabel("Shutoff head H0 (m)")
    axv.set_title("Shutoff head vs N^2 (valid: Q=0 => zero valve loss)")
    axv.grid(alpha=0.3); axv.legend()
    figv.tight_layout(); figv.savefig(os.path.join(OUT, "shutoff_vs_N2.png"), dpi=120)
plt.close(figv)

# 3) Cavitation breakdown curves (one figure per run)
cav_summary = {}
for tag, d in data.items():
    cav = ep.analyse_cav(tag, d)
    cav_summary[tag] = cav
    if not cav["steps"]:
        continue
    fig, ax = plt.subplots(figsize=(9, 6))
    for s in cav["steps"]:
        line, = ax.plot(s["cen"], s["Hmed"], "-o", ms=3, lw=1.5)
        c = line.get_color()
        ax.plot(s["n"], s["H"], ".", ms=1.5, alpha=0.12, color=c)
        if s.get("lowhead"):
            lab = f"out={s['level']:.0f}  Q~{s['q_ref']:.3f}  low-head pt (H_ref={s['H_ref']:.1f} m)"
        elif s["broke"]:
            lab = f"out={s['level']:.0f}  Q~{s['q_ref']:.3f}  NPSHr={s['npshr']:.2f} m"
            ax.axvline(s["npshr"], ls="--", lw=0.8, alpha=0.7, color=c)
        else:
            lab = f"out={s['level']:.0f}  Q~{s['q_ref']:.3f}  no breakdown (>{s['npsha_min']:.2f} m)"
        ax.plot([], [], color=c, label=lab)
    ax.set_xlabel("NPSHa (m)"); ax.set_ylabel("Developed head H (m)")
    ax.set_title(f"Cavitation breakdown  {tag}  ~{cav['rpm']:.0f} rpm")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"cavitation_{tag}.png"), dpi=120); plt.close(fig)

# ---- numeric summary to markdown ----
lines = ["# AI Pump Test Analysis\n",
         f"Water: rho={ep.RHO} kg/m3, Pv={ep.PV} bar, Patm={ep.PATM} bar.\n",
         "### Sensor map (established from throttle response)",
         "- pt_pump = suction UPSTREAM of inlet throttle (tank-side ref).",
         "- pt_in   = pump suction eye, DOWNSTREAM of inlet throttle -> NPSH ref.",
         "- pt_out  = TRUE pump discharge (between volute and outlet valve).",
         "",
         "NPSHa = (pt_in + Patm - Pv)/(rho g), velocity head neglected.",
         "Developed head H = (pt_out - pt_in); valid across the whole flow range.",
         "ESC has no speed loop -> RPM drifts ~8% over the ramp, so H-Q is",
         "affinity-normalised to Nref (H~N^2, Q~N). The curve legitimately crosses",
         "H=0 at the runout flow (pressurised feed pushes flow past the pump there).\n",
         "## H-Q characteristic (affinity-normalised)\n",
         "| Run | Nref (rpm) | Shutoff head H0 (m) | Runout flow Q (l/s) | H0/N^2 |",
         "|-----|-----|--------------------|---------------------|--------|"]
for tag, r in hq_summary.items():
    qr = f"{r['q_runout']:.3f}" if np.isfinite(r['q_runout']) else "n/a"
    lines.append(f"| {tag} | {r['rpm']:.0f} | {r['H_shutoff']:.1f} | {qr} | {r['H_shutoff']/r['rpm']**2:.3e} |")

lines += ["\n## Cavitation — NPSHr at 3% head drop\n"]
for tag, cav in cav_summary.items():
    if not cav["steps"]:
        continue
    lines.append(f"\n### {tag}  (~{cav['rpm']:.0f} rpm)")
    lines.append("| Outlet deg | Q_ref (l/s) | H_ref (m) | NPSHa_min (m) | NPSHr@3% (m) | Nss (rpm,m3/s,m) |")
    lines.append("|------------|-------------|-----------|---------------|--------------|------------------|")
    for s in cav["steps"]:
        nss = ""
        if s.get("lowhead"):
            nr = "low-head pt (n/a)"
        elif s["broke"]:
            nr = f"{s['npshr']:.2f}"
            nss = f"{s['nss']:.2f}" if np.isfinite(s.get("nss", np.nan)) else ""
        else:
            nr = f"<{s['npsha_min']:.2f} (no breakdown)"
        lines.append(f"| {s['level']:.0f} | {s['q_ref']:.3f} | {s['H_ref']:.1f} | {s['npsha_min']:.2f} | {nr} | {nss} |")

with open(os.path.join(OUT, "AI_analysis_summary.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("DONE. Outputs in", OUT)
print("\n".join(lines))
