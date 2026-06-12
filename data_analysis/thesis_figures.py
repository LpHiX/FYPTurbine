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

Figure register: AI SK-FIGURES.md (vault) is the single source of truth for
figure -> chapter placement and status. This file only owns generation.
Removed 2026-06-12 (Martin's call): psi_phi (duplicates theory_hq right panel),
eta_extrap (superseded by the fitted eta-Re trend, plots_round2 P2),
turbine_starting (cut for now).
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
from prop_components.turbine import Turbine
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


# --------------------------------------------------------------------------- #
# shutoff churning loss (disk_mult corroboration)
# --------------------------------------------------------------------------- #
# At shutoff (Q ~ 0) hydraulic power is zero, so shaft power = disk/blade
# churning + seal/bearing parasitics. tau*omega at Q~0 from the five H-Q ramps
# therefore measures the churning loss DIRECTLY at five speeds — the
# corroboration path for the fitted disk_mult ~ 2.5 (SK-08 discipline box): if
# measured shutoff churning sits ~2.5x over Barske's disk correlation across
# speeds, the multiplier graduates from "tuned" to "measured".
CHURN_Q_SHUTOFF_LPS = 0.02   # "at shutoff" flow threshold [l/s]; BEP ~ 0.22 l/s
CHURN_T_RAMP_HI = 22.0       # only the H-Q ramp window (healthy suction; the cav
                             # matrix sweeps NPSH down, which could alter churning)
CHURN_RPM_MIN = 2000


def barske_disk_power(rpm):
    """Barske empirical disk/churning correlation, pump as-built geometry."""
    p = pump()
    return (1956 * RHO * p.visc ** 0.2 * (np.asarray(rpm, float) / 1000) ** 2.8
            * (p.d_2 ** 4.6 + 4.6 * p.d_1 ** 3.6 * p.b_1))


def churning_shutoff():
    """Per-run shutoff churning power: median tau*omega at Q~0 minus the
    measured shaft-only parasitic. Returns list of dicts sorted by rpm."""
    rows = []
    for tag, path in ep.find_runs(LOGDIR).items():
        d = ep.load(path)
        t, q, rpm, tq = d["t"], d["q"], d["rpm"], d["tq"]
        spun = rpm[rpm > CHURN_RPM_MIN]
        if not spun.size:
            continue
        Nmed = np.nanmedian(spun)
        m = ((t <= CHURN_T_RAMP_HI) & (rpm > 0.85 * Nmed) & (rpm < 1.15 * Nmed)
             & (q < CHURN_Q_SHUTOFF_LPS) & np.isfinite(tq))
        if m.sum() < 30:   # relax once if the ramp barely touches shutoff
            m = ((t <= CHURN_T_RAMP_HI) & (rpm > 0.85 * Nmed) & (rpm < 1.15 * Nmed)
                 & (q < 0.03) & np.isfinite(tq))
        if m.sum() < 15:
            print(f"  churning: {tag} only {m.sum()} shutoff samples - skipped")
            continue
        w = rpm[m] * np.pi / 30
        P = tq[m] * w
        P_med = float(np.nanmedian(P))
        P_sem = 1.253 * float(np.nanstd(P)) / np.sqrt(m.sum())
        N = float(np.nanmedian(rpm[m]))
        P_par = float(np.interp(N, MR, MP))   # measured seal+bearing
        rows.append(dict(tag=tag, rpm=N, n=int(m.sum()),
                         P_shaft=P_med, P_sem=P_sem, P_par=P_par,
                         P_churn=P_med - P_par,
                         P_barske=float(barske_disk_power(N))))
    return sorted(rows, key=lambda r: r["rpm"])


# =========================================================================== #
# FIGURES
# =========================================================================== #
def fig_theory_hq():
    """H-Q + psi-phi, data vs Lock default (dashed) vs fitted (solid),
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
    _h, _l = a1.get_legend_handles_labels()
    a2.legend(_h, _l, fontsize=6, ncol=2, loc="upper right", framealpha=0.85,
              columnspacing=0.8, handletextpad=0.4)
    a2.set(xlabel=r"$\phi_2$", ylabel=r"$\psi$", ylim=(-0.1, 1.6))
    a2.axhline(0, color="k", lw=.5)
    fig.tight_layout()
    return save(fig, "theory_HQ_tuned")


def fig_eta_phi():
    """Overall efficiency vs flow: measured (binned, error bars) vs
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


def fig_churning():
    """Shutoff churning power vs speed: measured (markers) vs Barske
    disk model (dashed) vs the disk_mult-scaled Barske fitted from efficiency.
    Corroborates whether the fitted disk_mult is recovered DIRECTLY from the
    shutoff shaft power. Also prints the corroboration table."""
    rows = churning_shutoff()
    if len(rows) < 3:
        print("  churning: not enough shutoff points - figure skipped")
        return None

    N = np.array([r["rpm"] for r in rows])
    Pc = np.array([r["P_churn"] for r in rows])
    Pb = np.array([r["P_barske"] for r in rows])
    sem = np.array([r["P_sem"] for r in rows])
    a, loga0 = np.polyfit(np.log(N), np.log(Pc), 1)
    ratio = Pc / Pb

    print(f"\nShutoff churning corroboration (fitted disk_mult was {DISK_MULT}):")
    print(f"{'tag':>16} {'rpm':>6} {'n':>5} {'P_shaft':>8} {'P_par':>6} "
          f"{'P_churn':>8} {'P_barske':>8} {'ratio':>6}")
    for r in rows:
        print(f"{r['tag']:>16} {r['rpm']:6.0f} {r['n']:5d} {r['P_shaft']:8.1f} "
              f"{r['P_par']:6.1f} {r['P_churn']:8.1f} {r['P_barske']:8.1f} "
              f"{r['P_churn']/r['P_barske']:6.2f}")
    print(f"fit: P_churn = {np.exp(loga0):.3e} * n^{a:.2f}   (Barske exponent 2.8)")
    print(f"ratio to Barske: mean {ratio.mean():.2f}, "
          f"range {ratio.min():.2f}-{ratio.max():.2f}")
    print(f"at 20k rpm: power-law fit -> {np.exp(loga0) * 20000 ** a:.0f} W, "
          f"mean-ratio x Barske -> {ratio.mean() * barske_disk_power(20000):.0f} W")

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    nn = np.linspace(N.min() * 0.9, N.max() * 1.15, 100)
    plot_data(ax, N, Pc, yerr=_cover(sem), color="C0",
              label="measured shutoff churning")
    plot_tuned(ax, nn, np.exp(loga0) * nn ** a, color="C0",
               label=rf"fit $P \propto n^{{{a:.2f}}}$")
    plot_theory(ax, nn, barske_disk_power(nn), color="k", label="Barske disk model")
    ax.plot(nn, DISK_MULT * barske_disk_power(nn), ls=":", color="C3",
            label=rf"${DISK_MULT}\times$ Barske (fitted from $\eta$)")
    ax.plot(nn, np.interp(nn, MR, MP), ls="-.", color="gray", lw=1,
            label="shaft-only parasitic (meas.)")
    ax.set(xlabel="shaft speed [rpm]", ylabel="power [W]",
           xscale="log", yscale="log")
    ax.legend(fontsize=7)
    return save(fig, "results_churning_power")


def fig_hq_extrap():
    """H-Q extrapolated to 20k (affinity on the pooled psi-phi master
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
    """NPSHr vs Q: 3% breakdown (data) + video throat onsets + Lock
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
    """Turbine-driven H-Q scatter vs motor-driven curves."""
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
    """Coupled turbine torque vs speed: measured (markers) vs naive
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


def fig_turbine_eta_uc0():
    """Impulse stage efficiency vs blade-jet ratio: ideal (dashed) vs
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
# diagram / design / validation figures (MoC blade, Barske geometry, Goldman)
# =========================================================================== #
MACH_IN_MOC, BETA_MOC_DEG, DV_MOC = 1.3, 75.0, 0.004  # as-designed (turbinemoc.ipynb)
MOC_SPREAD_NARROW = (1.25, 1.35)   # illustration only: near-degenerate passage
GOLDMAN = dict(M_in=2.5, beta_deg=70.0, Re=35000, nu_l_deg=22.0, nu_u_deg=49.0)


def moc_design():
    if "moc" not in _cache:
        from prop_components.turbinemoc import SupersonicTurbineMOC
        m = SupersonicTurbineMOC(gamma=TRB["GAM"], mach_inlet=MACH_IN_MOC,
                                 mach_lower=MACH_LOWER, mach_upper=MACH_UPPER,
                                 beta_inlet_deg=BETA_MOC_DEG, dv=DV_MOC)
        m.generate()
        _cache["moc"] = m
    return _cache["moc"]


def _draw_passage(ax, moc, color="k", lw=1.2):
    """Blade passage in the unrotated frame: inlet transitions + vortex arcs +
    mirrored outlet transitions + TE closure. Geometry logic mirrors
    SupersonicTurbineMOC.plot_blade(unrotated=True), restyled for the thesis."""
    res = moc.results
    xl, yl = moc.coords["lower"]["x"], moc.coords["lower"]["y"]
    xu, yu = moc.coords["upper"]["x"], moc.coords["upper"]["y"]
    Rl, Ru = res["Rl"], res["Ru"]
    a_l, a_u = res["alpha_lower_inlet"], res["alpha_upper_inlet"]
    for x, y in ((xl, yl), (xu, yu)):
        ax.plot(x, y, color=color, lw=lw)
        ax.plot(-x, y, color=color, lw=lw)
    th = np.linspace(np.pi / 2 - a_l, np.pi / 2 + a_l, 100)
    ax.plot(Rl * np.cos(th), Rl * np.sin(th), color=color, lw=lw)
    th = np.linspace(np.pi / 2 - a_u, np.pi / 2 + a_u, 100)
    ax.plot(Ru * np.cos(th), Ru * np.sin(th), color=color, lw=lw)
    y_te = yu[-1] + (xl[-1] - xu[-1]) * np.tan(moc.beta_inlet - a_u)
    ax.plot([xu[-1], xl[-1]], [yu[-1], y_te], color=color, lw=lw)
    ax.plot([-xu[-1], -xl[-1]], [yu[-1], y_te], color=color, lw=lw)
    ax.set_aspect("equal")
    return Rl, Ru


def fig_moc_contour():
    """MoC blade passage, as-designed blade (M_in 1.3, surfaces 1.05/1.5),
    upper/lower surfaces labelled. Pairs with moc_separation_design."""
    moc = moc_design()
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    Rl, Ru = _draw_passage(ax, moc)
    ax.annotate(rf"lower (concave) surface, $M_l={moc.mach_lower}$",
                xy=(0, Rl), xytext=(0, Rl * 1.12), ha="center", fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.annotate(rf"upper (convex) surface, $M_u={moc.mach_upper}$",
                xy=(0, Ru), xytext=(0, Ru * 0.72), ha="center", fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set(xlabel=r"$x/r^{*}_{\!s}$", ylabel=r"$y/r^{*}_{\!s}$")
    return save(fig, "moc_contour_design")


def fig_moc_spread():
    """TWO PDFs (thesis subfigure pair): same inlet (M 1.3, beta 75) with
    (a) nearly-uniform surface Machs -> sliver passage, (b) the design spread
    1.05/1.5 -> wide passage. Shared axis limits so widths compare directly."""
    from prop_components.turbinemoc import SupersonicTurbineMOC
    built = []
    for name, (ml, mu) in (("moc_spread_narrow", MOC_SPREAD_NARROW),
                           ("moc_spread_wide", (MACH_LOWER, MACH_UPPER))):
        m = SupersonicTurbineMOC(gamma=TRB["GAM"], mach_inlet=MACH_IN_MOC,
                                 mach_lower=ml, mach_upper=mu,
                                 beta_inlet_deg=BETA_MOC_DEG, dv=DV_MOC)
        m.generate()
        built.append((name, m))
    xm, ylo, yhi = 0.0, np.inf, -np.inf
    for _, m in built:
        r = m.results
        for k in ("lower", "upper"):
            xm = max(xm, float(np.max(np.abs(m.coords[k]["x"]))))
            ylo = min(ylo, float(np.min(m.coords[k]["y"])))
        # the vortex arcs extend beyond the transition-line endpoints
        xm = max(xm, r["Rl"] * np.sin(r["alpha_lower_inlet"]),
                 r["Ru"] * np.sin(r["alpha_upper_inlet"]))
        ylo = min(ylo, r["Rl"] * np.cos(r["alpha_lower_inlet"]),
                  r["Ru"] * np.cos(r["alpha_upper_inlet"]))
        yhi = max(yhi, r["Rl"])
    path = None
    for name, m in built:
        fig, ax = plt.subplots(figsize=(3.1, 3.1))
        _draw_passage(ax, m)
        ax.text(0.03, 0.03, rf"$M_l={m.mach_lower}$,  $M_u={m.mach_upper}$",
                transform=ax.transAxes, va="bottom", fontsize=8)
        ax.set(xlim=(-1.1 * xm, 1.1 * xm), ylim=(0.95 * ylo, 1.05 * yhi),
               xlabel=r"$x/r^{*}_{\!s}$", ylabel=r"$y/r^{*}_{\!s}$")
        path = save(fig, name)
    return path


def fig_moc_separation():
    """Hi along both surfaces of the AS-DESIGNED blade at design-point inlet
    conditions (Sasman-Cresci BL on the MoC Mach distributions) vs the
    Schlichting 1.8-2.4 separation band. Pairs with moc_contour_design."""
    from prop_components.blade_profiler import DisplacedBladeProfiler
    prof = DisplacedBladeProfiler(turbine_design(), moc_design(),
                                  bl_method="sasman_cresci")
    prof.evaluate_boundary_layers()
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for side, c in (("lower", "C0"), ("upper", "C3")):
        r = prof.bl_results[side]
        plot_tuned(ax, r["s"], r["Hi"], color=c, label=f"{side} surface")
    ax.axhspan(1.8, 2.4, color="C3", alpha=0.10)
    ax.axhline(1.8, color="gray", lw=0.6, ls="--")
    ax.axhline(2.4, color="gray", lw=0.6, ls="--")
    ax.text(0.02, 0.97, "separation range $H_i$ = 1.8–2.4",
            transform=ax.transAxes, va="top", fontsize=7, color="0.3")
    ax.set(xlabel="fraction of surface arc length $s/c$",
           ylabel="incompressible form factor $H_i$")
    ax.legend(loc="lower right")
    return save(fig, "moc_separation_design")


def _dim(ax, p0, p1, text, tpos=None, fs=6.5, **kw):
    """Double-headed dimension arrow between p0 and p1 with a label."""
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="<->", lw=0.6, shrinkA=0, shrinkB=0, **kw))
    if tpos is None:
        tpos = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
    ax.text(*tpos, text, fontsize=fs, ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=0.4))


def fig_barske_geometry():
    """TWO PDFs (thesis subfigure pair): Barske pump geometry schematics driven
    by the sized geometry object — (a) meridional half-section with d0/d1/d2,
    b1/b2, s_ax, casing B x H; (b) end view with blades, annular casing and the
    tangential conical diffuser (throat d3 -> exit d4). Schematic: radial
    proportions true, wall thicknesses and diffuser arrangement illustrative."""
    p = pump()
    mm = 1000.0
    r0, r1, r2 = p.d_0 / 2 * mm, p.d_1 / 2 * mm, p.d_2 / 2 * mm
    b1, b2, sax = p.b_1 * mm, p.b_2 * mm, p.s_ax * mm
    Bc, Hc = p.B * mm, p.H * mm
    d3, d4 = p.d_3 * mm, p.d_4 * mm
    Rc = r2 + Hc          # casing inner radius
    rsh = 5.0             # shaft radius, illustrative

    # ---- (a) meridional half-section: x axial [mm], y radial [mm] ----
    # The section is physically very thin (B ~ 4.7 mm vs r2 ~ 29 mm), so the
    # axial direction is exaggerated for legibility (set_aspect below); state
    # "axial direction exaggerated" in the caption. Note the casing front is
    # CONICAL: it follows the tapered blade front (b1 at root > b2 at tip,
    # B = 2*s_ax + b2 applies at the tip radius only).
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    wall = dict(color="0.25", lw=1.4)
    imp = dict(color="C0", lw=1.2)
    xf1 = 2 * sax + b1            # front wall axial position at the root radius
    ax.axhline(0, color="k", lw=0.6, ls="-.")                     # shaft axis
    # casing: rear wall, outer wall, conical front wall, eye + inlet pipe
    ax.plot([0, 0], [rsh, Rc], **wall)
    ax.plot([0, Bc], [Rc, Rc], **wall)
    ax.plot([Bc, Bc], [Rc, r2], **wall)
    ax.plot([Bc, xf1], [r2, r1], **wall)                           # conical front
    ax.plot([xf1, xf1], [r1, r0], **wall)
    ax.plot([xf1, xf1 + 3.0], [r0, r0], **wall)                    # inlet pipe wall
    # impeller: hub + tapered blade (root b1 at r1 -> tip b2 at r2)
    ax.plot([sax, sax + b1, sax + b1], [0, 0, r1], **imp)          # hub
    ax.plot([sax, sax], [0, r2], **imp)                            # rear face
    ax.plot([sax, sax + b2], [r2, r2], **imp)                      # blade tip
    ax.plot([sax + b2, sax + b1], [r2, r1], **imp)                 # tapered front edge
    ax.plot([-1.5, sax], [rsh * 0.5, rsh * 0.5], color="0.5", lw=2)  # shaft stub
    # dimensions (diameters staggered left/right of the section)
    _dim(ax, (-1.2, 0), (-1.2, r2), r"$d_2/2$", tpos=(-2.1, r2 * 0.45))
    _dim(ax, (xf1 + 1.2, 0), (xf1 + 1.2, r1), r"$d_1/2$",
         tpos=(xf1 + 2.4, r1 * 0.45))
    _dim(ax, (xf1 + 2.4, 0), (xf1 + 2.4, r0), r"$d_0/2$",
         tpos=(xf1 + 3.6, r0 * 0.75))
    _dim(ax, (sax, r1 * 1.10), (sax + b1, r1 * 1.10), r"$b_1$",
         tpos=(sax + b1 / 2, r1 * 1.28))
    _dim(ax, (sax, r2 + 0.45 * Hc), (sax + b2, r2 + 0.45 * Hc), r"$b_2$",
         tpos=(sax + b2 / 2, r2 + 1.05 * Hc))
    _dim(ax, (0, r2 * 0.80), (sax, r2 * 0.80), r"$s_{ax}$",
         tpos=(-1.4, r2 * 0.80))
    _dim(ax, (0, Rc + 1.6), (Bc, Rc + 1.6), r"$B$", tpos=(Bc / 2, Rc + 3.2))
    _dim(ax, (sax + b2 / 2, r2), (sax + b2 / 2, Rc), r"$H$",
         tpos=(sax + b2 / 2 + 1.3, (r2 + Rc) / 2))
    ax.set_aspect(0.22)   # axial exaggeration ~4.5x — caption must say so
    ax.set(xlim=(-3.5, xf1 + 5.0), ylim=(-1.5, Rc + 5.0))
    ax.axis("off")
    save(fig, "barske_meridional")

    # ---- (b) end view: impeller, annular casing, tangential diffuser ----
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(Rc * np.cos(th), Rc * np.sin(th), **wall)              # casing bore
    ax.plot(r2 * np.cos(th), r2 * np.sin(th), color="C0", lw=1.0, ls=":")
    ax.plot(r1 * np.cos(th), r1 * np.sin(th), **imp)               # hub
    ax.plot(r0 * np.cos(th), r0 * np.sin(th), color="0.6", lw=0.8, ls="--")  # eye
    tb = p.blade_thickness * mm
    for k in range(p.blade_number):                                # radial blades
        a = 2 * np.pi * k / p.blade_number
        ca, sa = np.cos(a), np.sin(a)
        ax.plot([r1 * ca - tb / 2 * sa, r2 * ca - tb / 2 * sa],
                [r1 * sa + tb / 2 * ca, r2 * sa + tb / 2 * ca], **imp)
        ax.plot([r1 * ca + tb / 2 * sa, r2 * ca + tb / 2 * sa],
                [r1 * sa - tb / 2 * ca, r2 * sa - tb / 2 * ca], **imp)
        ax.plot([r2 * ca - tb / 2 * sa, r2 * ca + tb / 2 * sa],
                [r2 * sa + tb / 2 * ca, r2 * sa - tb / 2 * ca], **imp)
    # tangential conical diffuser, inner wall tangent at top of the casing bore
    Ld = 2.2 * d4
    ax.plot([0, Ld], [Rc, Rc], **wall)                             # inner wall
    ax.plot([0, Ld], [Rc + d3, Rc + d4], **wall)                   # diverging wall
    ax.plot([0, 0], [Rc, Rc + d3], color="0.25", lw=0.8)           # cutwater/throat
    _dim(ax, (0.06 * Ld, Rc), (0.06 * Ld, Rc + d3 + 0.06 * (d4 - d3)),
         r"$d_3$", tpos=(-0.35 * d4, Rc + d3 * 2.2))
    _dim(ax, (Ld, Rc), (Ld, Rc + d4), r"$d_4$", tpos=(Ld + 0.75 * d4, Rc + d4 / 2))
    ax.annotate("annular casing", xy=(-Rc * 0.72, Rc * 0.72),
                xytext=(-1.55 * Rc, 1.25 * Rc), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    # rotation arrow (counter-clockwise, towards the tangential diffuser)
    rr = 0.55 * r1
    tha = np.linspace(np.deg2rad(150), np.deg2rad(30), 40)
    ax.plot(rr * np.cos(tha), rr * np.sin(tha), color="0.4", lw=0.8)
    ax.annotate("", xy=(rr * np.cos(tha[-1] - 0.12), rr * np.sin(tha[-1] - 0.12)),
                xytext=(rr * np.cos(tha[-1]), rr * np.sin(tha[-1])),
                arrowprops=dict(arrowstyle="<-", lw=0.8, color="0.4"))
    ax.text(0, rr * 0.45, r"$\omega$", fontsize=8, ha="center", color="0.4")
    ax.set_aspect("equal")
    ax.set(xlim=(-1.6 * Rc, 1.7 * Rc), ylim=(-1.25 * Rc, 1.45 * Rc))
    ax.axis("off")
    return save(fig, "barske_plan")


def fig_goldman_validation():
    """TWO PDFs (thesis subfigure pair): replication of Goldman TM X-2095's
    'typical' design case (M_in 2.5, beta 70 deg, Re 35e3, nu_l 22 / nu_u 49)
    with the MoC + Sasman-Cresci toolchain — (a) surface Mach distributions
    (cf. Goldman Fig. 5), (b) Hi with the separation band (cf. Goldman Fig. 4).
    Compare side-by-side with the paper's scanned figures."""
    from quicktests.goldman_exact import (M_from_nu, build_blade_surface,
                                          run_single_case_sasman_cresci)
    gd = GOLDMAN
    Ml = M_from_nu(np.deg2rad(gd["nu_l_deg"]))
    Mu = M_from_nu(np.deg2rad(gd["nu_u_deg"]))
    s_lo_f, Me_lo_f, _ = build_blade_surface(gd["M_in"], Ml, gd["beta_deg"],
                                             side="lower", M_other=Mu)
    s_up_f, Me_up_f, _ = build_blade_surface(gd["M_in"], Mu, gd["beta_deg"],
                                             side="upper", M_other=Ml)
    (s_lo, Hi_lo, _, _, s_up, Hi_up, _, _) = run_single_case_sasman_cresci(
        gd["M_in"], Ml, Mu, gd["beta_deg"], gd["Re"])

    fig, ax = plt.subplots(figsize=(3.1, 2.8))
    plot_tuned(ax, s_lo_f, Me_lo_f, color="C0", label="lower surface")
    plot_tuned(ax, s_up_f, Me_up_f, color="C3", label="upper surface")
    ax.set(xlabel="fraction of chord", ylabel="surface Mach number $M_e$",
           xlim=(0, 1), ylim=(1.5, 3.5))
    ax.legend(loc="lower right")
    save(fig, "goldman_validation_mach")

    fig, ax = plt.subplots(figsize=(3.1, 2.8))
    plot_tuned(ax, s_lo, Hi_lo, color="C0", label="lower surface")
    plot_tuned(ax, s_up, Hi_up, color="C3", label="upper surface")
    ax.axhspan(1.8, 2.4, color="C3", alpha=0.10)
    ax.axhline(1.8, color="gray", lw=0.6, ls="--")
    ax.axhline(2.4, color="gray", lw=0.6, ls="--")
    ax.set(xlabel="fraction of chord", ylabel="incompressible form factor $H_i$",
           xlim=(0, 1), ylim=(1.3, 2.6))
    ax.legend(loc="upper left")
    return save(fig, "goldman_validation_hi")


# =========================================================================== #
FIGURES = {
    "theory_hq": fig_theory_hq,            # H-Q + psi-phi pair (keep combined)
    "eta_phi": fig_eta_phi,
    "churning": fig_churning,              # disk_mult corroboration
    "hq_extrap": fig_hq_extrap,            # objective-3 figure
    "npshr_vs_q": fig_npshr_vs_q,
    "coupled_hq": fig_coupled_hq,
    "coupled_torque": fig_coupled_torque,
    "turbine_eta_uc0": fig_turbine_eta_uc0,
    # diagram / design / validation
    "moc_contour": fig_moc_contour,        # as-designed passage, labelled
    "moc_spread": fig_moc_spread,          # 2 PDFs: narrow vs wide Mach spread
    "moc_separation": fig_moc_separation,  # Hi on as-designed blade vs 1.8-2.4
    "barske_geometry": fig_barske_geometry,  # 2 PDFs: meridional + end view
    "goldman_validation": fig_goldman_validation,  # 2 PDFs: Mach + Hi replication
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
