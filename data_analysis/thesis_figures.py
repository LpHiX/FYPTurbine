"""
thesis_figures.py — THE single generator for every pipeline thesis figure.

    python thesis_figures.py            # build all figures -> report/figs/*.pdf
    python thesis_figures.py --list     # show figure names
    python thesis_figures.py psi_phi npshr_vs_q   # build selected only

Called by build_thesis.py --figs. Supersedes make_figures.py (kept as the raw
campaign QA/overview artefact) and the exploratory notebooks
(theory_vs_experiment.ipynb / turbine_analysis.ipynb), which remain the
playground — THIS file is what the thesis actually uses.

House style via thesis_pipeline/figstyle.py (solid = tuned/spline,
dashed = original theory, markers = experimental data with error bars).

Uncertainty model (plain RSS, systematic B + random P in quadrature):
  - P (random) is REAL: per-bin scatter (SEM of the median) from epump_io.
  - B (systematic) values live in UNC below and are PLACEHOLDERS —
    >>> TODO(Martin): replace every UNC entry from datasheets/calibration. <<<
  - Error bars are drawn at UNC['coverage'] * sigma (2 -> ~95%).
  - NOTE: B is common-mode (shifts a whole curve, doesn't scatter points),
    so per-point B+P bars overstate point-to-point scatter; caption should
    say bars are total (systematic + random) uncertainty.
  - Propagation through derived quantities (psi, phi, eta) is done with the
    `uncertainties` package (linear propagation, no hand partials).

Figure register mapping (AI SK-FIGURES.md):
  3.1  theory_hq        -> theory_HQ_tuned.pdf
  7.1  psi_phi          -> results_psi_phi.pdf
  7.2  eta_phi          -> results_eta_flow.pdf
  7.2b eta_extrap       -> results_eta_extrap_20k.pdf
  (—)  hq_extrap        -> results_HQ_extrap_20k.pdf   (objective-3 figure)
  7.3  npshr_vs_q       -> results_npshr_vs_q.pdf
  (—)  coupled_hq       -> results_coupled_HQ.pdf      (coupled-run proof)
  7.5  coupled_torque   -> results_coupled_torque.pdf
  8.1  turbine_starting -> turbine_starting.pdf
  4.4  turbine_eta_uc0  -> turbine_theory.pdf
"""
from __future__ import annotations

import json
import os
import sys
import types

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "thesis_pipeline"))

# turbine.py imports rocketcea at module level; the inert-gas path never uses it.
_m = types.ModuleType("rocketcea"); _m2 = types.ModuleType("rocketcea.cea_obj_w_units")
_m2.CEA_Obj = object
sys.modules.setdefault("rocketcea", _m)
sys.modules.setdefault("rocketcea.cea_obj_w_units", _m2)

import h5py
from uncertainties import unumpy as unp

import epump_io as ep
from figstyle import use_style, save, plot_data, plot_theory, plot_tuned
from prop_components.barskepump import BarskePump
from prop_components.turbine import Turbine, SupersonicStartingGoldman
from mech_components.bearing import Bearing
from mech_components.mechanicalseal import MechanicalSeal

# =========================================================================== #
# CONFIG — the only block you should ever need to edit
# =========================================================================== #
LOGDIR = r"D:\Projects\propbackend_logs\2026-06-02\hotfirelog"

# canonical pump H-Q runs (55% excluded: constantly hit motor ESC limits)
RUNS = {
    "30%": "20260602_150821",
    "35%": "20260602_152153",
    "40%": "20260602_153853",
    "45%": "20260602_162513",   # good video
    "50%": "20260602_163135",   # good video
}
CAV_RUNS = ["45%", "50%"]       # NPSHr reliable only here (video-confirmed)
ANNOT_DIR = os.path.join(_HERE, "cav_annotations")

# coupled turbine-driven run
COUPLED_H5 = r"D:\Projects\propbackend_logs\2026-06-05\hotfirelog\test_20260605_233937_HotfireLog.h5"
COUPLED_T = (1776.0, 1840.0)    # s analysis window

# pump geometry (barskesizer.ipynb / as-built)
D2, B2 = 0.058522, 0.003554     # impeller tip diameter / outlet width [m]
# D3: TRUE throat = 3.0 mm, CAD section-view confirmed 2026-06-11. The 3.8 mm
# cone cutwater sits DOWNSTREAM of the real minimum (semicircle into the
# annular casing). 3.8 was never the minimum area; do not revert.
D3_MM, DINLET_MM = 3.0, 19.54   # true diffuser throat / eye diameter [mm]
D3, DINLET = D3_MM / 1000, DINLET_MM / 1000
P_INLET = 3.0e5                 # inlet static pressure for throat/cutoff [Pa]
RHO, G = 998.0, 9.81

# Lock model constants. STATUS LABELS MATTER (Martin 2026-06-10): K/eta_losses
# are least-squares FITS, not physics; disk_mult=2.5 is an EMPIRICAL churning
# multiplier that still needs physical justification. Figures must always show
# the default model alongside, and legends must say "fitted".
# 2026-06-11 BOUNDED RE-FIT with true D3=3.0 fixed (_scratch_refit_lock.py):
# K1 in [0,1] -> 0.53 (prerotation; Lock's study 0.15-0.20, excess = LE
# rounding + effective-radius h0 optimism, both in Lock's paper);
# zeta in [0.194, 0.6] -> 0.42, INTERIOR (entry-separation augmented nozzle
# loss, Lock p.733). The old eta_losses=1.1 railed fit is RETIRED — it was
# compensating for the wrong throat diameter. rms 2.0 m over plateau/droop.
K_DEFAULT, ETAL_DEFAULT = 0.17, 0.194    # literature defaults
K_FIT, ETAL_FIT = 0.53, 0.42             # bounded re-fit 2026-06-11, D3=3.0
SEAL_F = 0.014                           # fitted seal friction (water film)
DISK_MULT = 2.5                          # fitted churning multiplier (see above)
ETA_DESIGN_PCT = 23.4                    # design-point overall efficiency [%]

# measured shaft-only parasitic torque (rpm, Nm) -> power interp
_MEAS = [(3722, 0.0160), (4011, 0.0099), (4101, 0.0107),
         (4159, 0.0103), (6739, 0.0112), (8699, 0.0151)]
MR = np.array([m[0] for m in _MEAS])
MP = np.array([m[1] * m[0] * np.pi / 30 for m in _MEAS])

# turbine design point (turbinesizer.ipynb)
TRB = dict(P_W=2600, RPM_DES=20000, D_MEAN_MM=95, MDOT_DES=0.0428,
           BETA_DEG=15, DOA=0.15, P_E=1.1e5, R_GAS=287.0, GAM=1.4,
           T01_DES=300.0, N_NOZ=3)
MACH_LOWER, MACH_UPPER = 1.05, 1.5      # MoC blade surface Machs (turbinemoc)
T01_TEST = 308.0                        # K measured turbine air temperature

N_DES = 20000                            # pump design speed
DESIGN_Q_LPS, DESIGN_H_M = 0.3, 203.9    # pump design point

# =========================================================================== #
# UNC — systematic (B) 1-sigma placeholders.
# >>> TODO(Martin): every value below is a GUESS. Replace from datasheets /
#     calibration records, then delete this banner. <<<
# =========================================================================== #
UNC = dict(
    p_bar=0.005,     # TODO(Martin): PT systematic [bar]. Placeholder 0.5% FS of 12 bar.
    q_rel=0.010,     # TODO(Martin): flowmeter, fraction of reading. Placeholder 1%.
    tq_nm=0.001,     # TODO(Martin): torque systematic [Nm] incl. tare drift band.
    rpm=5.0,        # TODO(Martin): tacho systematic [rpm].
    d2_m=0.05e-3,     # TODO(Martin): as-printed vs CAD tip diameter [m] (SLA).
    b2_m=0.05e-3,     # TODO(Martin): as-printed vs CAD outlet width [m] (SLA).
    coverage=2.0,    # error bars drawn at coverage*sigma (2 ~ 95%)
)
# derived: head systematic from two independent PTs, in metres of water
H_SYS_M = np.sqrt(2.0) * UNC["p_bar"] * 1e5 / (RHO * G)
NPSHA_SYS_M = UNC["p_bar"] * 1e5 / (RHO * G)   # single PT (pt_in)


def _cover(sig):
    return UNC["coverage"] * np.asarray(sig, float)


def u_head(H_m, sem, sys=True):
    """Binned head -> uarray; sys=False -> random P only (per-point bars)."""
    return unp.uarray(H_m, np.hypot(np.asarray(sem, float),
                                    H_SYS_M if sys else 0.0))


def u_flow(q_lps, sem=0.0, sys=True):
    rel = UNC["q_rel"] if sys else 0.0
    return unp.uarray(q_lps, np.hypot(np.asarray(sem, float),
                                      rel * np.abs(np.asarray(q_lps, float))))


def u_psi_phi(q_lps, H_m, N_rpm, Hsem=0.0, qsem=0.0, sys=True):
    """Gulich (phi2, psi) as uarrays. sys=True includes geometry/instrument
    systematics (use for the sys_badge); sys=False is random P only (use for
    per-point bars)."""
    d2 = unp.uarray(D2, UNC["d2_m"] if sys else 0.0)
    b2 = unp.uarray(B2, UNC["b2_m"] if sys else 0.0)
    n = unp.uarray(N_rpm, UNC["rpm"] if sys else 0.0)
    u2 = np.pi * d2 * n / 60.0
    H = u_head(H_m, Hsem, sys=sys)
    Q = u_flow(q_lps, qsem, sys=sys) / 1000.0  # l/s -> m3/s
    psi = 2 * G * H / u2 ** 2
    phi = Q / (np.pi * d2 * b2 * u2)
    return phi, psi


# =========================================================================== #
# shared objects / loaders (lazy, cached)
# =========================================================================== #
_cache = {}


def pump():
    if "pump" not in _cache:
        tb = Bearing(d=10, D=22, series=619, visc=1.0, C_0_kN=1.27, submerged=False)
        bb = Bearing(d=10, D=22, series=619, visc=1.0, C_0_kN=1.27, submerged=True)
        seal = MechanicalSeal(OD_mm=19.5, ID_mm=15, BD_mm=14, F_sp=100, f=SEAL_F)
        _cache["pump"] = BarskePump(0.3, 20e5, 1000, 1.002e-6, 20000, tb, bb, seal)
    return _cache["pump"]


def lock(Q, RPM, K, eL):
    return pump().analyse_lock(Q, RPM=RPM, D_3=D3, D_inlet=DINLET,
                               K_factor=K, eta_losses=eL, p_inlet=P_INLET)


def exp_runs():
    """{label -> dict(q,H,Hsem,qsem,N,d,tag)} binned H-Q for the canonical runs."""
    if "exp" not in _cache:
        out = {}
        paths = ep.find_runs(LOGDIR)
        for lbl, tag in RUNS.items():
            d = ep.load(paths[tag])
            r = ep.analyse_hq(tag, d)
            if r is None:
                continue
            out[lbl] = dict(q=np.asarray(r["qbin"]), H=np.asarray(r["Hbin"]),
                            Hsem=np.asarray(r["Hbin_sem"]), qsem=np.asarray(r["qbin_sem"]),
                            N=r["rpm"], d=d, tag=tag)
        _cache["exp"] = out
    return _cache["exp"]


def turbine_design():
    if "trb" not in _cache:
        t = Turbine(P=TRB["P_W"], RPM=TRB["RPM_DES"], d_mean_mm=TRB["D_MEAN_MM"],
                    mdot=TRB["MDOT_DES"], beta_deg=TRB["BETA_DEG"], doa=TRB["DOA"],
                    p_e=TRB["P_E"])
        t.from_inert_gas_real(R=TRB["R_GAS"], gam=TRB["GAM"], T01=TRB["T01_DES"],
                              nozzles=TRB["N_NOZ"])
        _cache["trb"] = t
    return _cache["trb"]


def pooled_psi_phi():
    """Pooled (phi, psi) over all runs (nominal values, for extrapolation)."""
    PHI, PSI = [], []
    for e in exp_runs().values():
        u2 = np.pi * D2 * e["N"] / 60
        PHI += list(e["q"] / 1000 / (np.pi * D2 * B2 * u2))
        PSI += list(2 * G * e["H"] / u2 ** 2)
    return np.array(PHI), np.array(PSI)


# --------------------------------------------------------------------------- #
# video-annotation helpers (transparent-pump cavitation)
# --------------------------------------------------------------------------- #
def _annot(tag):
    p = os.path.join(ANNOT_DIR, f"cav_{tag}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def hq_throat_clear():
    """High-flow throat-cavitation CLEARING point on the H-Q ramp, from video.

    During the H-Q ramp (outlet closing, flow falling from runout) the diffuser
    throat starts out cavitating (Lock high-flow cutoff regime) and clears once
    the flow drops below the cutoff. hq.throat_onset_t in the annotation = the
    frame the diffuser runs bubble-free. Maps video -> (Q_clear, NPSHa) and
    compares with the Lock-predicted cutoff flow at that speed.
    Returns list of dicts (one per annotated run).
    """
    out = []
    for lbl, e in exp_runs().items():
        doc = _annot(e["tag"])
        if not doc:
            continue
        t_clear = (doc.get("hq") or {}).get("throat_onset_t")
        if t_clear is None or t_clear < 0:
            continue
        d = e["d"]
        i = int(np.argmin(np.abs(d["t"] - (t_clear + doc.get("clock_offset_s", 0.0)))))
        s = e["N"] / d["rpm"][i]                       # affinity to run Nref
        q_clear = float(d["q"][i] * s)                 # l/s, normalised
        npsha = float(ep.head_m(d["pin"][i] + ep.PATM - ep.PV))
        # Lock prediction: largest Q with NPSHr_throat(Q) > NPSHa  (cutoff regime)
        qg = np.linspace(1e-5, max(q_clear / 1000 * 2.0, 3e-4), 400)
        npshr = np.asarray(lock(qg, e["N"], K_FIT, ETAL_FIT)["NPSHr_throat"], float)
        above = np.where(npshr > npsha)[0]
        q_pred = float(qg[above[0]] * 1000) if len(above) else np.nan
        out.append(dict(run=lbl, tag=e["tag"], t_clear=t_clear, q_clear_lps=q_clear,
                        npsha_m=npsha, q_pred_lps=q_pred))
    return out


def video_throat_onsets():
    """(q_ref, NPSHa_onset) of throat inception per cavitation step, from video."""
    pts = []
    for lbl in CAV_RUNS:
        e = exp_runs()[lbl]
        doc = _annot(e["tag"])
        if not doc:
            continue
        d, cav = e["d"], ep.analyse_cav(e["tag"], e["d"])
        qref = {int(s["level"]): s["q_ref"] for s in cav["steps"]}
        for k, blk in (doc.get("steps") or {}).items():
            t_on = blk.get("throat_onset_t")
            if t_on is None or t_on < 0 or int(k) not in qref:
                continue
            i = int(np.argmin(np.abs(d["t"] - (t_on + doc.get("clock_offset_s", 0.0)))))
            pts.append((qref[int(k)], float(ep.head_m(d["pin"][i] + ep.PATM - ep.PV)), lbl))
    return pts


# =========================================================================== #
# FIGURES
# =========================================================================== #
def fig_theory_hq():
    """3.1 — H-Q + psi-phi, data vs Lock default (dashed) vs fitted (solid),
    with error bars and the video throat-clear markers on the H-Q panel."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.8))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(exp_runs())))
    clears = {c["run"]: c for c in hq_throat_clear()}
    for (lbl, e), c in zip(exp_runs().items(), cols):
        H = u_head(e["H"], e["Hsem"])
        plot_data(a1, e["q"], unp.nominal_values(H), yerr=_cover(unp.std_devs(H)),
                  color=c, ms=3, label=f"{lbl} ~{e['N']:.0f} rpm")
        qth = np.linspace(1e-5, e["q"].max() / 1000 * 1.15, 80)
        plot_theory(a1, qth * 1000, lock(qth, e["N"], K_DEFAULT, ETAL_DEFAULT)["H_static"],
                    color=c, alpha=.6)
        plot_tuned(a1, qth * 1000, lock(qth, e["N"], K_FIT, ETAL_FIT)["H_static"], color=c)
        phi, psi = u_psi_phi(e["q"], e["H"], e["N"], e["Hsem"], e["qsem"])
        plot_data(a2, unp.nominal_values(phi), unp.nominal_values(psi),
                  yerr=_cover(unp.std_devs(psi)), color=c, ms=3)
        u2 = np.pi * D2 * e["N"] / 60
        plot_tuned(a2, qth / (np.pi * D2 * B2 * u2),
                   2 * G * np.asarray(lock(qth, e["N"], K_FIT, ETAL_FIT)["H_static"]) / u2 ** 2,
                   color=c)
        if lbl in clears:
            a1.axvline(clears[lbl]["q_clear_lps"], color=c, ls=":", lw=1.0)
    if clears:
        a1.plot([], [], "k:", label="video: throat clears")
    a1.plot([], [], "k--", label="Lock (default)")
    a1.plot([], [], "k-", label="Lock (fitted)")
    a1.set(xlabel="Q [l/s]", ylabel="static head H [m]", ylim=(-2, None))
    a1.axhline(0, color="k", lw=.5)
    a1.legend(fontsize=6, ncol=2, loc="upper right", framealpha=0.85,
              columnspacing=0.8, handletextpad=0.4)
    a2.set(xlabel=r"$\phi_2$", ylabel=r"$\psi$", ylim=(-0.1, None))
    a2.axhline(0, color="k", lw=.5)
    return save(fig, "theory_HQ_tuned")


def fig_psi_phi():
    """7.1 — standalone psi-phi collapse, 5 speeds + Lock default/fitted."""
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(exp_runs())))
    for (lbl, e), c in zip(exp_runs().items(), cols):
        phi, psi = u_psi_phi(e["q"], e["H"], e["N"], e["Hsem"], e["qsem"])
        plot_data(ax, unp.nominal_values(phi), unp.nominal_values(psi),
                  yerr=_cover(unp.std_devs(psi)), xerr=_cover(unp.std_devs(phi)),
                  color=c, ms=3, label=f"{lbl} ~{e['N']:.0f} rpm")
    Nref = max(e["N"] for e in exp_runs().values())
    u2 = np.pi * D2 * Nref / 60
    qth = np.linspace(1e-6, 0.022 * np.pi * D2 * B2 * u2, 80)
    plot_theory(ax, qth / (np.pi * D2 * B2 * u2),
                2 * G * np.asarray(lock(qth, Nref, K_DEFAULT, ETAL_DEFAULT)["H_static"]) / u2 ** 2,
                color="k", label="Lock (default)")
    plot_tuned(ax, qth / (np.pi * D2 * B2 * u2),
               2 * G * np.asarray(lock(qth, Nref, K_FIT, ETAL_FIT)["H_static"]) / u2 ** 2,
               color="k", label="Lock (fitted)")
    ax.set(xlabel=r"flow coefficient $\phi_2$", ylabel=r"head coefficient $\psi$",
           ylim=(-0.1, None))
    ax.axhline(0, color="k", lw=.5); ax.legend()
    return save(fig, "results_psi_phi")


def fig_eta_phi():
    """7.2 — overall efficiency vs flow: measured (binned, error bars) vs
    Lock model. Untuned churning x1 (model default, dashed) and fitted
    churning x2.5 (solid) per run. Cavitation-onset annotation: TODO(Martin)."""
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(exp_runs())))
    for (lbl, e), c in zip(exp_runs().items(), cols):
        d = e["d"]; t = d["t"]; m = (t >= 0.5) & (t <= 20)
        Q = d["q"][m] / 1000
        H = ep.head_m(d["pout"][m] - d["pin"][m])
        rpm = d["rpm"][m]; w = rpm * 2 * np.pi / 60; tq = d["tq"][m]
        Phyd = RHO * G * Q * H; Psh = tq * w
        good = (rpm > 2000) & (Psh > 0) & (H > 0) & (Q > 0)
        Q, H, rpm, tq, eo = Q[good], H[good], rpm[good], tq[good], (Phyd / Psh)[good]
        qb = np.linspace(0, np.nanpercentile(Q * 1000, 98), 16)
        idx = np.digitize(Q * 1000, qb)
        for i in range(1, len(qb)):
            s = idx == i
            if s.sum() < 5:
                continue
            qc = np.median(Q[s] * 1000)
            ec = np.median(eo[s])
            sem = 1.253 * np.std(eo[s]) / np.sqrt(s.sum())          # random P
            rel_sys = np.sqrt(                                       # systematic B
                (UNC["q_rel"]) ** 2 +
                (H_SYS_M / max(np.median(H[s]), 1e-3)) ** 2 +
                (UNC["tq_nm"] / max(np.median(tq[s]), 1e-3)) ** 2 +
                (UNC["rpm"] / np.median(rpm[s])) ** 2)
            sig = np.hypot(sem, ec * rel_sys)
            plot_data(ax, [qc], [ec * 100], yerr=[_cover(sig) * 100], color=c, ms=3)
        qth = np.linspace(1e-5, (np.nanpercentile(Q, 98)) * 1.1, 60)
        et0 = pump().efficiency_lock(qth, RPM=e["N"], D_3=D3, D_inlet=DINLET,
                                     K_factor=K_FIT, eta_losses=ETAL_FIT,
                                     disk_mult=1.0)
        plot_theory(ax, qth * 1000, np.asarray(et0["eta_ovr"]) * 100, color=c, alpha=.6)
        et = pump().efficiency_lock(qth, RPM=e["N"], D_3=D3, D_inlet=DINLET,
                                    K_factor=K_FIT, eta_losses=ETAL_FIT,
                                    disk_mult=DISK_MULT)
        plot_tuned(ax, qth * 1000, np.asarray(et["eta_ovr"]) * 100, color=c)
        ax.plot([], [], "o", color=c, ms=3, label=f"{lbl} ~{e['N']:.0f} rpm")
    ax.plot([], [], "k--", label=r"Lock, untuned churning $\times$1")
    ax.plot([], [], "k-", label=rf"Lock + fitted churning $\times${DISK_MULT}")
    ax.axhline(ETA_DESIGN_PCT, color="gray", ls=":", label=f"design ~{ETA_DESIGN_PCT:.0f}%")
    # TODO(Martin): annotate observed cavitation onset (arrow) once phi_onset chosen
    ax.set(xlabel="Q [l/s]", ylabel="overall efficiency [%]", ylim=(0, None))
    ax.legend(fontsize=6)
    return save(fig, "results_eta_flow")


def fig_eta_extrap():
    """7.2b — efficiency Reynolds step-up to 20k rpm."""
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    eta_T, N_meas = 0.19, 7483
    Ns = np.linspace(3500, 20000, 100)
    for a in (0.1, 0.15, 0.2):
        plot_tuned(ax, Ns, (1 - (1 - eta_T) * (N_meas / Ns) ** a) * 100,
                   alpha=0.8, label=rf"Re step-up $a={a}$")
    ax.axhline(ETA_DESIGN_PCT, color="gray", ls=":", label=f"design ~{ETA_DESIGN_PCT:.0f}%")
    ax.axvline(N_DES, color="k", lw=.5)
    sig = _cover(0.01)  # TODO(Martin): real eta uncertainty from fig_eta_phi point
    plot_data(ax, [N_meas], [eta_T * 100], yerr=[sig * 100], color="C0",
              label=f"measured @{N_meas} rpm")
    ax.set(xlabel="shaft speed [rpm]", ylabel="overall efficiency [%]")
    ax.legend()
    return save(fig, "results_eta_extrap_20k")


def fig_hq_extrap():
    """(extra) — H-Q extrapolated to 20k (affinity on the pooled psi-phi master
    curve) vs Lock, with the design point marked."""
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    PHI, PSI = pooled_psi_phi()
    u2d = np.pi * D2 * N_DES / 60
    phi_grid = np.linspace(0.0005, 0.02, 40)
    order = np.argsort(PHI)
    ps = np.interp(phi_grid, PHI[order], PSI[order])
    H20, Q20 = ps * u2d ** 2 / (2 * G), phi_grid * np.pi * D2 * B2 * u2d * 1000
    plot_data(ax, Q20, H20, yerr=_cover(np.full_like(H20, H_SYS_M * (N_DES / 7483) ** 2)),
              ms=3, color="C0", label="experiment, affinity to 20k")
    qth = np.linspace(1e-5, 0.0012, 80)
    plot_tuned(ax, qth * 1000, lock(qth, N_DES, K_FIT, ETAL_FIT)["H_static"],
               color="C3", label="Lock (fitted) @20k")
    plot_theory(ax, qth * 1000, lock(qth, N_DES, K_DEFAULT, ETAL_DEFAULT)["H_static"],
                color="C3", label="Lock (default) @20k")
    ax.plot([DESIGN_Q_LPS], [DESIGN_H_M], "k*", ms=12,
            label=f"design point ({DESIGN_Q_LPS} l/s, {DESIGN_H_M:.0f} m)")
    ax.set(xlabel="Q [l/s]", ylabel="head [m]")
    ax.legend()
    return save(fig, "results_HQ_extrap_20k")


def fig_npshr_vs_q():
    """7.3 — NPSHr vs Q: 3% breakdown (data) + video throat onsets + Lock
    throat curve (dashed) + the high-flow clearing comparison."""
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for lbl in CAV_RUNS:
        e = exp_runs()[lbl]
        cav = ep.analyse_cav(e["tag"], e["d"])
        pts = sorted((s["q_ref"], s["npshr"]) for s in cav["steps"]
                     if s.get("broke") and np.isfinite(s["npshr"]))
        if pts:
            q, n = map(np.array, zip(*pts))
            # TODO(Martin): refine NPSHr sigma via local-slope method; placeholder
            # = PT systematic + half the NPSHa bin width.
            sig = np.hypot(NPSHA_SYS_M, 0.25)
            plot_data(ax, q, n, yerr=_cover(np.full_like(n, sig)), ms=4,
                      label=f"{lbl} 3% breakdown (data)")
    vid = video_throat_onsets()
    if vid:
        q, n, _ = zip(*vid)
        ax.plot(q, n, "x", ms=6, color="k", label="video: throat inception")
    qth = np.linspace(1e-5, 0.45e-3, 80)
    lk = lock(qth, exp_runs()["50%"]["N"], K_FIT, ETAL_FIT)
    plot_theory(ax, qth * 1000, lk["NPSHr_throat"], color="k",
                label="Lock throat NPSHr (high-flow cutoff)")
    # at the clearing frame NPSHa = NPSHr(Q_clear): a point ON the curve,
    # same footing as the inception crosses
    clr = hq_throat_clear()
    if clr:
        ax.plot([c["q_clear_lps"] for c in clr], [c["npsha_m"] for c in clr],
                "+", ms=7, mew=1.2, color="0.35",
                label="video: H-Q throat clears")
    ax.axhline(0, color="k", lw=.5)
    ax.set(xlabel="Q [l/s]", ylabel="NPSH [m]", ylim=(-5, None))
    ax.legend()
    return save(fig, "results_npshr_vs_q")


def _coupled_window():
    f = h5py.File(COUPLED_H5, "r")
    t = np.asarray(f["channels"]["adc_rpm_mv"]["time"][:])
    m = (t >= COUPLED_T[0]) & (t <= COUPLED_T[1])

    def ch(n):
        return np.asarray(f["channels"][n]["data"][:])[m]
    return dict(t=t[m], rpm=ch("adc_rpm_mv"), tq=ch("adc_torque_mv"),
                ptt=ch("adc_pt_turbine_mv"), q=ch("fms_fm0_flowrate"),
                H=ep.head_m(ch("adc_pt_out_mv") - ch("adc_pt_in_mv")))


def fig_coupled_hq():
    """(extra) — turbine-driven H-Q scatter vs motor-driven curves."""
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    w = _coupled_window()
    sel = (w["rpm"] > 2000) & (w["H"] > -2)
    sc = ax.scatter(w["q"][sel], w["H"][sel], s=6, c=w["rpm"][sel], cmap="plasma",
                    label="turbine-driven")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("rpm")
    for lbl in ("40%", "45%"):
        e = exp_runs()[lbl]
        plot_tuned(ax, e["q"], e["H"], label=f"motor {lbl} ~{e['N']:.0f} rpm")
    ax.set(xlabel="Q [l/s]", ylabel="developed head [m]")
    ax.legend()
    return save(fig, "results_coupled_HQ")


def fig_coupled_torque():
    """7.5 — coupled turbine torque vs speed: measured (markers) vs naive
    STARTED model (dashed) vs unstarted model (solid + band)."""
    t = turbine_design()
    gam, Rg = TRB["GAM"], TRB["R_GAS"]
    beta = np.deg2rad(TRB["BETA_DEG"]); r_mean = TRB["D_MEAN_MM"] / 2000.0
    A_th, M3, phi_r = t.A_throat, t.M3, t.phi_r
    T3s = T01_TEST / (1 + 0.5 * (gam - 1) * M3 ** 2)
    a3s = np.sqrt(gam * Rg * T3s); c3s = M3 * a3s; c3us = c3s * np.cos(beta)

    def u_of(rpm):
        return rpm * 2 * np.pi / 60 * r_mean

    def choked_mdot(p01):
        return (A_th * p01 / np.sqrt(T01_TEST) * np.sqrt(gam / Rg)
                * (2 / (gam + 1)) ** ((gam + 1) / (2 * (gam - 1))))

    def torque_started(rpm, p01):
        u = u_of(rpm)
        return choked_mdot(p01) * (1 + phi_r) * u * (c3us - u) / (u / r_mean)

    def torque_unstarted(rpm, p01, jet_frac):
        u = u_of(rpm); c3e = jet_frac * c3s
        return choked_mdot(p01) * (1 + phi_r) * u * (c3e * np.cos(beta) - u) / (u / r_mean)

    w = _coupled_window()
    sel = w["rpm"] > 2000
    rpm_w, tq_w, ptt_w = w["rpm"][sel], w["tq"][sel], w["ptt"][sel]
    p01_w = (ptt_w + 1.01325) * 1e5
    md_w = choked_mdot(p01_w); omega = rpm_w * 2 * np.pi / 60
    c3u_eff = (tq_w * omega / md_w) / ((1 + phi_r) * u_of(rpm_w)) + u_of(rpm_w)
    frac = float(np.nanmedian((c3u_eff / np.cos(beta)) / c3s))

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    plot_data(ax, rpm_w[::20], tq_w[::20], yerr=_cover(UNC["tq_nm"]),
              ms=2, alpha=0.6, color="C0", label="measured (every 20th pt)")
    rr = np.linspace(rpm_w.min(), rpm_w.max(), 40)
    p01_med = float(np.median(p01_w))
    plot_theory(ax, rr, torque_started(rr, p01_med), color="C3",
                label=f"started model (design jet {c3s:.0f} m/s)")
    ax.fill_between(rr, torque_unstarted(rr, p01_med, 0.30),
                    torque_unstarted(rr, p01_med, 0.40),
                    color="C2", alpha=0.2, label="unstarted, 30–40% jet")
    plot_tuned(ax, rr, torque_unstarted(rr, p01_med, frac), color="C2",
               label=f"unstarted, fitted {frac * 100:.0f}% jet")
    ax.set(xlabel="shaft speed [rpm]", ylabel="shaft torque [Nm]", ylim=(0, None))
    ax.legend()
    return save(fig, "results_coupled_torque")


def fig_turbine_starting():
    """8.1 — Goldman rotor-starting map: Mw3 vs speed, limit, test range, design."""
    t = turbine_design()
    gam, Rg = TRB["GAM"], TRB["R_GAS"]
    beta = np.deg2rad(TRB["BETA_DEG"]); r_mean = TRB["D_MEAN_MM"] / 2000.0
    M3 = t.M3
    T3s = T01_TEST / (1 + 0.5 * (gam - 1) * M3 ** 2)
    a3s = np.sqrt(gam * Rg * T3s); c3s = M3 * a3s
    c3us, c3ms = c3s * np.cos(beta), c3s * np.sin(beta)

    g = SupersonicStartingGoldman(gam)
    Msl = g.mstar_from_mach(MACH_LOWER, gam)
    Msu = g.mstar_from_mach(MACH_UPPER, gam)
    Mw3_max = g.max_inlet_mach(Msl, Msu)

    def Mw3_super(rpm):
        u = rpm * 2 * np.pi / 60 * r_mean
        return np.hypot(c3us - u, c3ms) / a3s

    from scipy.optimize import brentq
    rpm_thresh = brentq(lambda n: Mw3_super(n) - Mw3_max, 1000, 30000)

    w = _coupled_window()
    rpm_w = w["rpm"][w["rpm"] > 2000]

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ns = np.linspace(1000, 21000, 200)
    plot_tuned(ax, ns, [Mw3_super(n) for n in ns], color="C0",
               label=r"rotor inlet $M_{w3}$ (started branch)")
    ax.axhline(Mw3_max, color="k", ls="--",
               label=rf"Goldman limit $M_{{w3,max}}={Mw3_max:.3f}$")
    ax.axvspan(rpm_w.min(), rpm_w.max(), color="C1", alpha=.2, label="test speed range")
    ax.axvline(rpm_thresh, color="C3", ls=":", label=f"starting threshold {rpm_thresh:.0f} rpm")
    ax.plot([TRB["RPM_DES"]], [t.Mw3], "k*", ms=12,
            label=rf"design 20k ($M_{{w3}}$={t.Mw3:.2f}, started)")
    ax.annotate("UNSTARTED\n(bow shock, subsonic passage)",
                (4000, Mw3_super(4000)), xytext=(7000, 1.75), fontsize=8, color="C3",
                arrowprops=dict(arrowstyle="->", color="C3"))
    ax.set(xlabel="shaft speed [rpm]", ylabel=r"rotor relative inlet Mach $M_{w3}$")
    ax.legend(loc="lower left")
    return save(fig, "turbine_starting")


def fig_turbine_eta_uc0():
    """4.4 — impulse stage efficiency vs blade-jet ratio: ideal (dashed) vs
    Weiss-coefficient real (solid), design point marked."""
    t = turbine_design()
    beta = np.deg2rad(TRB["BETA_DEG"])
    nu = np.linspace(0.01, np.cos(beta) * 0.999, 200)
    eta_ideal = 4 * nu * (np.cos(beta) - nu)                 # phi_n = phi_r = 1
    eta_real = 2 * t.phi_n ** 2 * nu * (np.cos(beta) - nu) * (1 + t.phi_r)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    plot_theory(ax, nu, eta_ideal, color="k", label="ideal impulse")
    plot_tuned(ax, nu, eta_real, color="C0",
               label=r"with Weiss $\varphi_n,\varphi_r$ (design pt values)")
    nu_des = float(t.u / t.c3)
    ax.plot([nu_des], [t.eff_real], "k*", ms=12,
            label=rf"design ($\nu$={nu_des:.2f}, $\eta$={t.eff_real:.2f} incl. windage)")
    ax.set(xlabel=r"blade-jet speed ratio $\nu = u/c_3$", ylabel=r"stage efficiency $\eta$",
           ylim=(0, 1))
    ax.legend()
    return save(fig, "turbine_theory")


# =========================================================================== #
FIGURES = {
    "theory_hq": fig_theory_hq,            # 3.1
    "psi_phi": fig_psi_phi,                # 7.1
    "eta_phi": fig_eta_phi,                # 7.2
    "eta_extrap": fig_eta_extrap,          # 7.2b
    "hq_extrap": fig_hq_extrap,            # extra (objective 3)
    "npshr_vs_q": fig_npshr_vs_q,          # 7.3
    "coupled_hq": fig_coupled_hq,          # extra
    "coupled_torque": fig_coupled_torque,  # 7.5
    "turbine_starting": fig_turbine_starting,  # 8.1
    "turbine_eta_uc0": fig_turbine_eta_uc0,    # 4.4
}


def main(argv):
    if "--list" in argv:
        for k, fn in FIGURES.items():
            print(f"{k:18s} {fn.__doc__.splitlines()[0]}")
        return
    names = [a for a in argv if not a.startswith("-")] or list(FIGURES)
    use_style()
    print(f"UNC placeholders in effect (coverage={UNC['coverage']}x sigma):")
    for k, v in UNC.items():
        if k != "coverage":
            print(f"  {k:6s} = {v}   <- TODO(Martin): replace from datasheet/calibration")
    for name in names:
        if name not in FIGURES:
            print(f"!! unknown figure '{name}' (use --list)")
            continue
        path = FIGURES[name]()
        plt.close("all")
        print(f"[ok] {name:18s} -> {path}")
    # throat-clear comparison table (the high-flow Lock cutoff check)
    clears = hq_throat_clear()
    if clears:
        print("\nH-Q throat-cavitation clearing (video vs Lock fitted):")
        for c in clears:
            print(f"  {c['run']:4s} t={c['t_clear']:6.2f}s  Q_clear={c['q_clear_lps']:.3f} l/s "
                  f"(NPSHa={c['npsha_m']:.1f} m)   Lock predicted Q={c['q_pred_lps']:.3f} l/s")


if __name__ == "__main__":
    main(sys.argv[1:])
