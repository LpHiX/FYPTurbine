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
    p_bar=0.05,     # TODO(Martin): PT systematic [bar]. Placeholder 0.5% FS of 12 bar.
    q_rel=0.03,     # TODO(Martin): flowmeter, fraction of reading. Placeholder 1%.
    tq_nm=0.02,     # TODO(Martin): torque systematic [Nm] incl. tare drift band.
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


# design-space sweep bounds (turbinesizer.ipynb driver cell)
DS_D_RANGE = (80, 120)       # mm
DS_MD_RANGE = (0.035, 0.06)  # kg/s
DS_N = 60                    # grid points per axis
DS_H_MIN = 4e-3              # m   minimum printable blade height
DS_THROAT_MIN = 0.5e-3       # m   minimum printable nozzle throat width
DS_M3_MAX = 1.8              # nozzle exit Mach ceiling (loss-correlation range)
DS_P0_MAX = 10e5             # Pa  guard (matches turbinesizer GUARD)


def fig_turbine_design_space():
    """(d_m, mdot) design-space contours: M3, blade height, real efficiency,
    with manufacturing/rig constraint lines and the selected design point."""
    ds = np.linspace(*DS_D_RANGE, DS_N)
    ms = np.linspace(*DS_MD_RANGE, DS_N)
    M3 = np.full((DS_N, DS_N), np.nan)
    H = np.full_like(M3, np.nan)
    TH = np.full_like(M3, np.nan)
    ETA = np.full_like(M3, np.nan)
    for i, d in enumerate(ds):
        for j, md in enumerate(ms):
            try:
                s = Turbine(P=TRB["P_W"], RPM=TRB["RPM_DES"], d_mean_mm=d,
                            mdot=md, beta_deg=TRB["BETA_DEG"], doa=TRB["DOA"],
                            p_e=TRB["P_E"])
                s.from_inert_gas_real(R=TRB["R_GAS"], gam=TRB["GAM"],
                                      T01=TRB["T01_DES"], nozzles=TRB["N_NOZ"])
                if not (s.p01 <= DS_P0_MAX and s.T3 > 0):
                    continue
                M3[i, j], H[i, j] = s.M3, s.Height
                TH[i, j], ETA[i, j] = s.nozzle_throat_length, s.eff_real
            except Exception:
                pass
    feas = ((H >= DS_H_MIN) & (TH >= DS_THROAT_MIN) & (M3 <= DS_M3_MAX)
            & ~np.isnan(M3))
    panels = [(M3, 1.0, r"nozzle exit Mach $M_3$"),
              (H, 1e3, r"blade height $b$ (mm)"),
              (ETA, 1.0, r"predicted $\eta_{ts}$")]
    fig, axs = plt.subplots(1, 3, figsize=(9.8, 3.1), sharey=True)
    from matplotlib.lines import Line2D
    for ax, (Z, sc, ttl) in zip(axs, panels):
        cs = ax.contourf(ms, ds, np.ma.masked_invalid(Z * sc), levels=30,
                         cmap="viridis")
        fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.03)
        ax.contour(ms, ds, np.ma.masked_invalid(H * 1e3),
                   levels=[DS_H_MIN * 1e3], colors="r", linewidths=1.4)
        ax.contour(ms, ds, np.ma.masked_invalid(TH * 1e3),
                   levels=[DS_THROAT_MIN * 1e3], colors="orange", linewidths=1.4)
        ax.contour(ms, ds, np.ma.masked_invalid(M3),
                   levels=[DS_M3_MAX], colors="w", linewidths=1.4)
        ax.contourf(ms, ds, feas.astype(float), levels=[-0.5, 0.5],
                    colors=["k"], alpha=0.30)
        ax.plot(TRB["MDOT_DES"], TRB["D_MEAN_MM"], "r*", ms=11)
        ax.set(title=ttl, xlabel=r"$\dot{m}$ (kg/s)")
    axs[0].set_ylabel(r"$d_m$ (mm)")
    axs[0].legend(handles=[
        Line2D([0], [0], color="r", lw=1.4, label=rf"$b \geq$ {DS_H_MIN*1e3:.0f} mm"),
        Line2D([0], [0], color="orange", lw=1.4,
               label=rf"throat $\geq$ {DS_THROAT_MIN*1e3:.1f} mm"),
        Line2D([0], [0], color="grey", lw=1.4, label=rf"$M_3 \leq$ {DS_M3_MAX} (white)"),
        Line2D([0], [0], marker="*", color="r", ls="", ms=10, label="design point"),
    ], loc="upper left", fontsize=6)
    fig.tight_layout()
    return save(fig, "turbine_design_space")


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
    """Blade passage in the ROTATED (physical) frame: rotated transition lines
    + vortex arcs + TE closure + the adjacent blade's lower surface (pitched
    down by one passage) so the figure reads as a real cascade. Geometry logic
    mirrors SupersonicTurbineMOC.plot_blade(unrotated=False, full_passage=True)
    — the view turbinemoc.ipynb uses — restyled for the thesis."""
    res = moc.results
    xl, yl = moc.coords["lower_rot"]["x"], moc.coords["lower_rot"]["y"]
    xu, yu = moc.coords["upper_rot"]["x"], moc.coords["upper_rot"]["y"]
    Rl, Ru = res["Rl"], res["Ru"]
    a_l, a_u = res["alpha_lower_inlet"], res["alpha_upper_inlet"]
    for x, y in ((xl, yl), (xu, yu)):
        ax.plot(x, y, color=color, lw=lw)
        ax.plot(-x, y, color=color, lw=lw)
    for R, a in ((Rl, a_l), (Ru, a_u)):
        th = np.linspace(np.pi / 2 - a, np.pi / 2 + a, 100)
        ax.plot(R * np.cos(th), R * np.sin(th), color=color, lw=lw)
    y_te = yu[-1] + (xl[-1] - xu[-1]) * np.tan(moc.beta_inlet)
    ax.plot([xu[-1], xl[-1]], [yu[-1], y_te], color=color, lw=lw)
    ax.plot([-xu[-1], -xl[-1]], [yu[-1], y_te], color=color, lw=lw)
    # adjacent blade: the next passage's lower surface shifted down one pitch,
    # closing the solid blade silhouette (plot_blade rotated branch)
    th_l = np.linspace(np.pi / 2 + a_l, np.pi / 2 - a_l, 100)
    xb = np.concatenate([np.flip(xl), Rl * np.cos(th_l), -xl])
    yb = np.concatenate([np.flip(yl), Rl * np.sin(th_l), yl]) - yl[-1] + y_te
    ax.plot(xb, yb, color=color, lw=lw)
    # FILL the solid blade between the upper surface (+ TE closures) and the
    # adjacent blade's lower surface, so the figure reads as a cascade
    th_u = np.linspace(np.pi / 2 + a_u, np.pi / 2 - a_u, 100)
    x_up = np.concatenate([[xl[-1]], np.flip(xu), Ru * np.cos(th_u),
                           -xu, [-xl[-1]]])
    y_up = np.concatenate([[y_te], np.flip(yu), Ru * np.sin(th_u),
                           yu, [y_te]])
    ax.fill(np.concatenate([x_up, np.flip(xb)]),
            np.concatenate([y_up, np.flip(yb)]),
            color="0.85", zorder=0)
    ax.set_aspect("equal")
    return Rl, Ru


def fig_moc_contour():
    """MoC blade passage, as-designed blade (M_in 1.3, surfaces 1.05/1.5),
    upper/lower surfaces labelled. Pairs with moc_separation_design."""
    moc = moc_design()
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    Rl, Ru = _draw_passage(ax, moc)
    # ax.annotate(rf"lower (concave) surface, $M_l={moc.mach_lower}$",
    #             xy=(0, Rl), xytext=(0, Rl * 1.14), ha="center", fontsize=7,
    #             arrowprops=dict(arrowstyle="->", lw=0.7))
    # ax.annotate(rf"upper (convex) surface, $M_u={moc.mach_upper}$",
    #             xy=(0, Ru), xytext=(0, (Rl + Ru) / 2), ha="center", fontsize=7,
    #             arrowprops=dict(arrowstyle="->", lw=0.7))
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
    # draw first, then take the union of the drawn-line bounds for shared limits
    # (the adjacent-blade outline extends well below the transition endpoints)
    figs = []
    for name, m in built:
        fig, ax = plt.subplots(figsize=(3.1, 3.1))
        _draw_passage(ax, m)
        ax.text(0.03, 0.03, rf"$M_l={m.mach_lower}$,  $M_u={m.mach_upper}$",
                transform=ax.transAxes, va="bottom", fontsize=8)
        figs.append((name, fig, ax))
    xlo = ylo = np.inf
    xhi = yhi = -np.inf
    for _, _, ax in figs:
        for ln in ax.get_lines():
            xd, yd = np.asarray(ln.get_xdata()), np.asarray(ln.get_ydata())
            xlo, xhi = min(xlo, xd.min()), max(xhi, xd.max())
            ylo, yhi = min(ylo, yd.min()), max(yhi, yd.max())
    px, py = 0.05 * (xhi - xlo), 0.05 * (yhi - ylo)
    path = None
    for name, fig, ax in figs:
        ax.set(xlim=(xlo - px, xhi + px), ylim=(ylo - py, yhi + py),
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


def fig_moc_displaced():
    """Ideal MoC contour (dashed) vs the boundary-layer-displaced 'metal'
    contour actually manufactured (solid, delta* offset, Sasman-Cresci at
    design-point inlet conditions) — DisplacedBladeProfiler, as run in
    turbinemoc.ipynb. The displacement is what makes the printed passage
    deliver the inviscid design flow."""
    from prop_components.blade_profiler import DisplacedBladeProfiler
    prof = DisplacedBladeProfiler(turbine_design(), moc_design(),
                                  bl_method="sasman_cresci")
    prof.evaluate_boundary_layers()
    prof.displace_contour()
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for side, c in (("lower", "C0"), ("upper", "C3")):
        d = prof.displaced_coords[side]
        plot_theory(ax, d["x_ideal"], d["y_ideal"], color=c,
                    label=f"{side} surface, ideal MoC")
        plot_tuned(ax, d["x_disp"], d["y_disp"], color=c,
                   label=f"{side} surface, $\\delta^*$-displaced (as-built)")
    ax.set_aspect("equal")
    ax.set(xlabel=r"$x/r^{*}_{\!s}$", ylabel=r"$y/r^{*}_{\!s}$")
    ax.legend(fontsize=6, loc="lower center")
    return save(fig, "moc_displaced_design")


def _dim(ax, p0, p1, text, tpos=None, fs=6.5, **kw):
    """Double-headed dimension arrow between p0 and p1 with a label."""
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="<->", lw=0.6, shrinkA=0, shrinkB=0, **kw))
    if tpos is None:
        tpos = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
    ax.text(*tpos, text, fontsize=fs, ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=0.4))


def fig_barske_geometry():
    """ONE PDF (Martin 06-13): the two Barske schematics combined into a single
    figure, side by side and sharing the radial (y) axis so the meridional
    section and the end view map onto each other. (a) meridional half-section
    with d0/d1/d2, b1/b2, s_ax, casing B x H; (b) end view with blades, annular
    casing and the tangential conical diffuser (throat d3 -> exit d4, symmetric
    about a centreline at r2, running to the LEFT). Dimension symbols enlarged.
    Schematic: radial proportions true, wall thicknesses illustrative."""
    p = pump()
    mm = 1000.0
    r0, r1, r2 = p.d_0 / 2 * mm, p.d_1 / 2 * mm, p.d_2 / 2 * mm
    b1, b2, sax = p.b_1 * mm, p.b_2 * mm, p.s_ax * mm
    Bc, Hc = p.B * mm, p.H * mm
    d3 = p.d_3 * mm           # throat from sizing
    d4 = 6.0                  # exit diameter [mm], hard-coded (Martin 06-13)
    Rc = r2 + Hc             # casing inner radius
    rsh = 5.0                # shaft radius, illustrative
    LBL = 12                 # dimension-symbol font size (was ~6.5, invisible)

    from matplotlib.patches import Polygon, Rectangle, Circle
    rib = 3.0                     # rib height [mm] — visualize() value
    x_back = rib + b1             # blade/rib back face axial position
    Ld = 30.0                     # diffuser length [mm], hard-coded (used below)
    # shared radial axis so radial features line up across the two views;
    # width_ratios set to each panel's x-data span so that, with equal aspect,
    # both axes boxes render at the SAME height (height = k * y_range for both).
    ylim = (-(Rc + 8), Rc + 8)
    xlim_m = (-11, x_back + 11)
    xlim_p = (-Ld - 12, 1.55 * Rc)
    wr = [xlim_m[1] - xlim_m[0], xlim_p[1] - xlim_p[0]]
    fig, (axm, axp) = plt.subplots(
        1, 2, figsize=(7.4, 5.2), sharey=True,
        gridspec_kw={"width_ratios": wr})

    # ---- (a) meridional view: LITERALLY BarskePump.visualize()'s ax1
    # construction (keep the plain matplotlib look, label onto it) — same
    # polygons, same colours, axes in mm, dims drawn on top.
    axm.plot([-10, x_back + 10], [0, 0], color="black", ls="--", lw=0.8)
    axm.plot([-10, -sax], [-r0, -r0], color="green", lw=1.2)
    axm.plot([-10, -sax], [r0, r0], color="green", lw=1.2)
    for s in (+1, -1):
        # blade (visualize polygon: root spans 0..rib+b1, tip width b2)
        axm.add_patch(Polygon([[0, s * r1], [rib + b1 - b2, s * r2],
                               [x_back, s * r2], [x_back, s * r1]],
                              closed=True, fill=False, edgecolor="C0", lw=1.2))
        # casing polyline (visualize: eye wall -> conical front -> outer)
        axm.add_patch(Polygon([[-sax, s * r0], [-sax, s * r1],
                               [-sax + rib + b1 - b2, s * r2],
                               [-sax + rib + b1 - b2, s * (r2 + Hc)],
                               [x_back + sax, s * (r2 + Hc)],
                               [x_back + sax, s * r1]],
                              closed=False, fill=False, edgecolor="black",
                              lw=1.2))
    axm.add_patch(Rectangle((b1, -r1), rib, 2 * r1, fill=False,
                            edgecolor="purple", lw=1.0))
    # ---- dimensions (symbols only; values live in tab_pump_design) ----
    _dim(axm, (-8.0, -r0), (-8.0, r0), r"$d_0$", tpos=(-8.0, r0 * 0.55), fs=LBL)
    for xd, rr, lab in ((x_back + 3.5, r1, r"$d_1$"),
                        (x_back + 7.0, r2, r"$d_2$")):
        axm.plot([x_back, xd], [rr, rr], color="0.7", lw=0.4)
        axm.plot([x_back, xd], [-rr, -rr], color="0.7", lw=0.4)
        _dim(axm, (xd, -rr), (xd, rr), lab, tpos=(xd + 2.4, rr * 0.35), fs=LBL)
    rmid = (r1 + r2) / 2
    xb_mid = (rmid - r1) / (r2 - r1) * (rib + b1 - b2)
    axm.annotate(r"$s_{ax}$", xy=(xb_mid - sax / 2, rmid),
                 xytext=(xb_mid - sax - 7.0, rmid + 4.0), fontsize=LBL,
                 arrowprops=dict(arrowstyle="->", lw=0.6))
    _dim(axm, (rib + b1 - b2, r2 + 0.5 * Hc), (x_back, r2 + 0.5 * Hc),
         r"$b_2$", tpos=(rib + b1 - b2 / 2 - 2.7, r2 + 0.5 * Hc), fs=LBL)
    _dim(axm, (0, -r1 * 0.55), (b1, -r1 * 0.55), r"$b_1$",
         tpos=(b1 / 2, -r1 * 0.55 - 2.6), fs=LBL)
    _dim(axm, (-sax + rib + b1 - b2, r2 + Hc + 2.6), (x_back + sax, r2 + Hc + 2.6),
         r"$b_c$", tpos=((rib + b1 - b2 + x_back) / 2, r2 + Hc + 5.4), fs=LBL)
    _dim(axm, (x_back + 1.0, -r2), (x_back + 1.0, -(r2 + Hc)), r"$h_c$",
         tpos=(x_back + 4.0, -(r2 + Hc / 2)), fs=LBL)
    axm.set_aspect("equal")
    axm.set(xlim=xlim_m, ylim=ylim, xlabel="axial [mm]", ylabel="radial [mm]")

    # ---- (b) end view: visualize()'s ax2 (circles + legend look), with
    # blade THICKNESS added and the tangential diffuser for d3/d4 ----
    axp.add_patch(Circle((0, 0), r0, fill=False, color="green",
                         label=r"inlet ($d_0$)"))
    axp.add_patch(Circle((0, 0), r1, fill=False, color="blue",
                         label=r"blade root ($d_1$)"))
    axp.add_patch(Circle((0, 0), r2, fill=False, color="red",
                         label=r"blade tip ($d_2$)"))
    axp.add_patch(Circle((0, 0), Rc, fill=False, color="black", ls="--",
                         label="annular casing"))
    tb = p.blade_thickness * mm
    for k in range(p.blade_number):                  # radial blades, thick
        a = 2 * np.pi * k / p.blade_number
        ca, sa = np.cos(a), np.sin(a)
        axp.plot([r1 * ca - tb / 2 * sa, r2 * ca - tb / 2 * sa],
                 [r1 * sa + tb / 2 * ca, r2 * sa + tb / 2 * ca], color="blue", lw=1.2)
        axp.plot([r1 * ca + tb / 2 * sa, r2 * ca + tb / 2 * sa],
                 [r1 * sa - tb / 2 * ca, r2 * sa - tb / 2 * ca], color="blue", lw=1.2)
        axp.plot([r2 * ca - tb / 2 * sa, r2 * ca + tb / 2 * sa],
                 [r2 * sa + tb / 2 * ca, r2 * sa - tb / 2 * ca], color="blue", lw=1.2)
    # tangential conical diffuser: symmetric about a centreline at r2, throat
    # d3 -> exit d4, running to the LEFT from x=0 to x=-Ld (hard-coded above).
    yc = r2
    axp.plot([0, -Ld], [yc + d3 / 2, yc + d4 / 2], color="black", lw=1.2)  # upper wall
    axp.plot([0, -Ld], [yc - d3 / 2, yc - d4 / 2], color="black", lw=1.2)  # lower wall
    axp.plot([0, 0], [yc - d3 / 2, yc + d3 / 2], color="black", lw=0.8)    # throat cap
    axp.plot([-Ld, -Ld], [yc - d4 / 2, yc + d4 / 2], color="black", lw=0.8)  # exit cap
    _dim(axp, (0.10 * Ld, yc - d3 / 2), (0.10 * Ld, yc + d3 / 2),
         r"$d_3$", tpos=(0.40 * Ld, yc), fs=LBL)
    _dim(axp, (-Ld, yc - d4 / 2), (-Ld, yc + d4 / 2), r"$d_4$",
         tpos=(-Ld - 0.22 * Ld, yc), fs=LBL)
    # rotation arrow
    rr = 0.55 * r1
    tha = np.linspace(np.deg2rad(150), np.deg2rad(30), 40)
    axp.plot(rr * np.cos(tha), rr * np.sin(tha), color="0.4", lw=0.8)
    axp.annotate("", xy=(rr * np.cos(tha[-1] - 0.12), rr * np.sin(tha[-1] - 0.12)),
                 xytext=(rr * np.cos(tha[-1]), rr * np.sin(tha[-1])),
                 arrowprops=dict(arrowstyle="<-", lw=0.8, color="0.4"))
    axp.text(0, rr * 0.45, r"$\omega$", fontsize=LBL, ha="center", color="0.4")
    axp.set_aspect("equal")
    axp.set(xlim=xlim_p, xlabel="tangential [mm]")
    axp.legend(fontsize=8, loc="lower right")
    fig.subplots_adjust(wspace=0.05)
    return save(fig, "barske_geometry")


def fig_goldman_validation():
    """TWO PDFs (thesis subfigure pair): replication of Goldman TM X-2095's
    'typical' design case (M_in 2.5, beta 70 deg, Re 35e3, nu_l 22 / nu_u 49)
    with the MoC + Sasman-Cresci toolchain — (a) surface Mach distributions
    (cf. Goldman Fig. 5), (b) Hi with the separation band (cf. Goldman Fig. 4).
    Compare side-by-side with the paper's scanned figures."""
    from quicktests.goldman_exact import (M_from_nu, build_blade_surface,
                                          run_single_case_sasman_cresci)
    gd = GOLDMAN
    Ml = 1.68
    Mu = 2.9
    s_lo_f, Me_lo_f, _ = build_blade_surface(gd["M_in"], Ml, gd["beta_deg"],
                                             side="lower", M_other=Mu)
    s_up_f, Me_up_f, _ = build_blade_surface(gd["M_in"], Mu, gd["beta_deg"],
                                             side="upper", M_other=Ml)
    (s_lo, Hi_lo, _, _, s_up, Hi_up, _, _) = run_single_case_sasman_cresci(
        gd["M_in"], Ml, Mu, gd["beta_deg"], Re_chord=150000, Hi_0=1.67, theta_0_test=10)

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


def fig_turbine_triangles():
    """Rotor inlet (3) and exit (4) velocity triangles drawn from the
    design-point turbine object — quantitatively true, not schematic.
    Convention: u tangential (horizontal, rotor moves right), meridional
    down the page; angles measured from tangential (Sudhof-style)."""
    t = turbine_design()
    u, c3u, c3m = float(t.u), float(t.c3u), float(t.c3m)
    w3u, w3m = c3u - u, c3m
    # exit triangle exactly as the model carries it (turbine.py): symmetric
    # ideal impulse blade — w4 mirrors w3 (c4u = u - w3u), meridional kept
    w4u, w4m = -w3u, w3m
    c4u, c4m = float(t.c4u), c3m

    def tri(ax, ox, vecs, labels, colors):
        for (vx, vy), lab, c in zip(vecs, labels, colors):
            ax.annotate("", xy=(ox[0] + vx, ox[1] + vy), xytext=ox,
                        arrowprops=dict(arrowstyle="-|>", lw=1.1, color=c,
                                        shrinkA=0, shrinkB=0))
            ax.text(ox[0] + vx * 0.55, ox[1] + vy * 0.55, lab, fontsize=8,
                    color=c, ha="center", va="bottom",
                    bbox=dict(fc="white", ec="none", pad=0.3))

    fig, ax = plt.subplots(figsize=(6.2, 2.2))
    # ---- station 3 (rotor inlet): c3 = u + w3, all from a common origin ----
    o3 = (0.0, 0.0)
    tri(ax, o3, [(c3u, -c3m), (u, 0.0)], [r"$c_3$", r"$u$"], ["C0", "0.3"])
    ax.annotate("", xy=(c3u, -c3m), xytext=(u, 0.0),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="C3",
                                shrinkA=0, shrinkB=0))
    ax.text(u + w3u * 0.55, -w3m * 0.5, r"$w_3$", fontsize=8, color="C3",
            ha="left", va="center")
    ax.text(c3u * 0.5, 14, "rotor inlet (3)", fontsize=8, ha="center")
    # ---- station 4 (rotor exit), shifted right clear of station 3 ----
    # (c4u is strongly negative at this u/c3, so the exit triangle leans left)
    ox4 = c3u + max(0.0, -c4u) + 0.18 * c3u
    o4 = (ox4, 0.0)
    tri(ax, o4, [(c4u, -c4m), (u, 0.0)], [r"$c_4$", r"$u$"], ["C0", "0.3"])
    ax.annotate("", xy=(ox4 + c4u, -c4m), xytext=(ox4 + u, 0.0),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="C3",
                                shrinkA=0, shrinkB=0))
    ax.text(ox4 + u + w4u * 0.55, -w4m * 0.5, r"$w_4$", fontsize=8, color="C3",
            ha="right", va="center")
    ax.text(ox4 + u * 0.5, 14, "rotor exit (4)", fontsize=8, ha="center")
    # angle labels (from tangential)
    ax.text(c3u * 0.22, -c3m * 0.10, r"$\alpha_3$", fontsize=7)
    ax.text(u + w3u * 0.22, -w3m * 0.12, r"$\beta_3$", fontsize=7, color="C3")
    ax.text(ox4 + u + w4u * 0.30, -w4m * 0.18, r"$\beta_4$", fontsize=7,
            color="C3")
    # annotate() arrows do not register in autoscale -> set limits explicitly
    xs = [0, c3u, u, ox4, ox4 + u, ox4 + c4u, ox4 + u + w4u]
    ax.set(xlim=(min(xs) - 30, max(xs) + 30), ylim=(-c3m - 25, 30))
    ax.set_aspect("equal")
    ax.axis("off")
    # sanity check against the model's own resultant
    c4_chk = np.hypot(c4u, c4m)
    print(f"  triangles: c4 reconstructed {c4_chk:.1f} vs model {t.c4:.1f} m/s")
    return save(fig, "turbine_triangles")


def fig_forced_vortex_triangles():
    """Velocity-triangle nomenclature on the forced-vortex line (pump Euler head).
    Schematic, not to scale: Barske radial blades -> w purely meridional (radial),
    u purely tangential, c the hypotenuse; u = omega r is read straight off the
    u-axis, so each blade-speed arrow is the height of the line at that radius.
    Two arcs centred on the rotation axis pass through the r1, r2 ticks."""
    from matplotlib.patches import Arc
    cU, cC, cW, cK = "0.2", "C0", "C3", "0.55"
    r1, r2, Lw = 1.0, 2.0, 0.5            # u = omega r with omega = 1 (schematic)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))

    # impeller rim arcs, centred on the rotation axis, through each radius tick
    for r in (r1, r2):
        ax.add_patch(Arc((0.0, 0.0), 2 * r, 2 * r, theta1=-20, theta2=120,
                         color=cK, lw=1.1))
    # rotation sense
    ax.annotate("", xy=(-0.34, -0.10), xytext=(-0.12, -0.34),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=cK,
                                connectionstyle="arc3,rad=0.4",
                                shrinkA=0, shrinkB=0))
    ax.text(-0.52, -0.30, r"$\omega$", color=cK, fontsize=11,
            ha="center", va="center")

    # drawn axes (mpl spines are off)
    ax.annotate("", xy=(2.78, 0.0), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="0.1",
                                shrinkA=0, shrinkB=0))
    ax.annotate("", xy=(0.0, 2.58), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="0.1",
                                shrinkA=0, shrinkB=0))
    ax.text(2.80, -0.04, r"$r$", fontsize=11, ha="left", va="top")
    ax.text(-0.07, 2.58, r"$u$", fontsize=11, ha="right", va="top")

    # forced-vortex line u = omega r (each u-vector reaches it)
    ax.plot([0.0, 2.3], [0.0, 2.3], color=cU, lw=1.4)
    ax.text(1.5, 1.72, r"$u = \omega r$", color=cU, fontsize=10,
            ha="left", va="bottom", rotation=45, rotation_mode="anchor")

    def triangle(r, suf):
        O = np.array([r, 0.0]); A = np.array([r, r]); C = np.array([r + Lw, r])
        _cant_arrow(ax, O, A, cU)         # u : tangential, up to the line
        _cant_arrow(ax, A, C, cW)         # w : radial (meridional)
        _cant_arrow(ax, O, C, cC)         # c : absolute (hypotenuse)
        s = 0.08                          # right-angle mark at the u-tip
        ax.plot([A[0] + s, A[0] + s, A[0]], [A[1], A[1] - s, A[1] - s],
                color=cK, lw=0.8)
        ax.text(r - 0.06, r / 2, r"$u_%s$" % suf, color=cU, fontsize=11,
                ha="right", va="center")
        ax.text(r + Lw / 2, r + 0.05, r"$w_%s$" % suf, color=cW, fontsize=11,
                ha="center", va="bottom")
        ax.text(r + Lw / 2 + 0.07, r / 2, r"$c_%s$" % suf, color=cC,
                fontsize=11, ha="left", va="center")
        ax.plot([r, r], [-0.04, 0.04], color="0.1", lw=1.0)   # r tick
        ax.plot([r], [0.0], marker="o", ms=3, color="0.1")
        ax.text(r, -0.13, r"$r_%s$" % suf, fontsize=10, ha="center", va="top")

    triangle(r1, "1")
    triangle(r2, "2")

    ax.set(xlim=(-1.25, 2.98), ylim=(-0.95, 2.72))
    ax.set_aspect("equal")
    ax.axis("off")
    return save(fig, "forced_vortex_triangles")


def fig_campbell():
    """Campbell diagram of the turbine shaft (ross) — natural frequencies vs
    speed, 1x synchronous excitation line, operating range to 20k rpm shaded.
    Model per rotordynamics.ipynb but CORRECTED to the as-built shaft:
    10 mm solid ALUMINIUM (the notebook still had the retired 20 mm printed
    GreyV4 shaft), L = 95 mm, 6 Timoshenko elements, ball bearings at nodes
    1 and 5, turbine disk overhung at node 6 (measured mass/inertia).
    The coupler-side disk is omitted (free end) — state in caption."""
    import ross as rs
    al = rs.Material(name="Al6061", rho=2700, E=69e9, G_s=26e9)
    L, n_el = 0.095, 6
    shaft = [rs.ShaftElement(L=L / n_el, idl=0.0, odl=0.010, material=al,
                             shear_effects=True, rotary_inertia=True,
                             gyroscopic=True) for _ in range(n_el)]
    turb = rs.DiskElement(n=6, m=15.232e-3, Ip=7800e-9, Id=4000e-9)
    brgs = [rs.BallBearingElement(n=n, n_balls=9, d_balls=0.003, fs=20,
                                  alpha=0) for n in (1, 5)]
    rotor = rs.Rotor(shaft_elements=shaft, disk_elements=[turb],
                     bearing_elements=brgs)
    speeds = np.linspace(0, 16000, 36)            # rad/s sweep
    cam = rotor.run_campbell(speed_range=speeds)
    rpm = speeds * 60 / (2 * np.pi)
    wd_rpm = np.asarray(cam.wd) * 60 / (2 * np.pi)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for j in range(min(6, wd_rpm.shape[1])):
        plot_tuned(ax, rpm, wd_rpm[:, j], color="C0",
                   label="natural frequencies" if j == 0 else None)
    plot_theory(ax, rpm, rpm, color="C3", label="1x synchronous")
    ax.axvspan(0, 20000, color="C2", alpha=0.08)
    ax.axvline(20000, color="C2", lw=0.8, ls=":")
    ax.text(10000, ax.get_ylim()[1] * 0.04, "operating range", fontsize=7,
            color="C2", ha="center")
    # first forward critical: lowest crossing of wd with the 1x line
    crit = None
    for j in range(wd_rpm.shape[1]):
        d = wd_rpm[:, j] - rpm
        sgn = np.where(np.diff(np.sign(d)))[0]
        if len(sgn):
            i = sgn[0]
            x0 = rpm[i] + (rpm[i + 1] - rpm[i]) * d[i] / (d[i] - d[i + 1])
            crit = x0 if crit is None else min(crit, x0)
    if crit:
        ax.plot([crit], [crit], "k*", ms=10,
                label=f"first critical ~{crit / 1000:.0f}k rpm")
        print(f"  campbell: first forward critical ~ {crit:.0f} rpm "
              f"(registry turb_crit_speed = 99000)")
    ax.set(xlabel="shaft speed [rpm]", ylabel="natural frequency [rpm]",
           xlim=(0, rpm.max()), ylim=(0, None))
    ax.legend(loc="upper left", fontsize=7)
    return save(fig, "turbine_campbell")


# 2026-04-24 valve Kv characterisation runs (valvekv.ipynb)
VALVE_H5 = {
    "inlet (1/2\")":  (r"D:\Projects\propbackend_logs\2026-04-24\hotfirelog\test_20260424_171514_HotfireLog.h5",
                       "servos_pumpinlet_angle", 20.0, 3.0),   # kv full-open, design dp [bar]
    "outlet (1/4\")": (r"D:\Projects\propbackend_logs\2026-04-24\hotfirelog\test_20260424_175759_HotfireLog.h5",
                       "servos_pumpoutlet_angle", 7.0, 20.0),
}
# online ball-valve opening characteristic (valvekv.ipynb sources)
_VK_ANGLE = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
_VK_RATIO = [0.0, 0.0336, 0.0554, 0.0978, 0.1620, 0.2487, 0.3837, 0.6398,
             0.8598, 1.0]


def fig_valve_kv():
    """Valve flow capacity vs opening angle: online-data extrapolation
    (dashed) vs the 2026-04-24 measured Kv characterisation (solid), with
    the pump's throat-limited max flow line. Promoted from valvekv.ipynb
    ('Online data extrapolation vs experimental data'); y-axis trimmed to
    2 kg/s (Martin 06-13)."""
    import pandas as pd
    from scipy.interpolate import interp1d
    a2r = interp1d(_VK_ANGLE, _VK_RATIO, kind="cubic")
    ang = np.linspace(0, 90, 200)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for (name, c) in zip(VALVE_H5, ("C0", "C3")):
        path, ch, kv_full, head = VALVE_H5[name]
        with h5py.File(path, "r") as f:
            g = f["channels"]
            t = np.asarray(g["adc_pt_in_mv"]["time"][:], float)
            m = (t >= 0) & (t <= 180)
            pin = np.asarray(g["adc_pt_in_mv"]["data"][:], float)[m]
            pout = np.asarray(g["adc_pt_out_mv"]["data"][:], float)[m]
            q = np.asarray(g["fms_fm0_flowrate"]["data"][:], float)[m]
            angle = 90 - np.asarray(g[ch]["data"][:], float)[m] / 2
        roll = lambda x: pd.Series(x).rolling(1000, center=True).mean().to_numpy()
        dp = roll(pin) - roll(pout)
        kv = roll(q) / 1000 * 3600 / np.sqrt(np.clip(dp, 1e-6, None))
        plot_tuned(ax, angle, kv * np.sqrt(head) / 3600 * 1000, color=c,
                   label=f"{name} measured")
        plot_theory(ax, ang, kv_full * a2r(ang) * np.sqrt(head) / 3600 * 1000,
                    color=c, label=f"{name} extrapolated online data")
    mdotmax = np.pi / 4 * D3 ** 2 * np.sqrt(2) * (np.pi * D2 * N_DES / 60) * 1000
    ax.axhline(mdotmax, color="k", lw=0.8, ls="--")
    ax.text(2, mdotmax + 0.04, "pump max flow (throat-limited)", fontsize=6.5)
    ax.set(xlabel="valve opening angle [deg]", ylabel="flow rate [kg/s]",
           xlim=(0, 90), ylim=(0, 2.0))
    ax.legend(fontsize=6.5, loc="upper left")
    return save(fig, "setup_valve_kv")


def fig_eta_re():
    """Peak overall efficiency per run vs Reynolds number: loss ratio
    (1-eta) ~ Re^-a FITTED from the five measured speeds, extrapolated to
    20k rpm with the fit-uncertainty band. Promoted from plots_round2.py P2."""
    NU_W = 1.002e-6
    def re_u(rpm):
        return np.pi * D2 * np.asarray(rpm, float) / 60 * (D2 / 2) / NU_W
    rows = []
    for lbl, e in exp_runs().items():
        d = e["d"]; t = d["t"]; m = (t >= 0.5) & (t <= 20)
        Q = d["q"][m] / 1000
        H = ep.head_m(d["pout"][m] - d["pin"][m])
        rpm = d["rpm"][m]; w = rpm * 2 * np.pi / 60; tq = d["tq"][m]
        Phyd = RHO * G * Q * H; Psh = tq * w
        good = (rpm > 2000) & (Psh > 0) & (H > 0) & (Q > 0)
        Q, eo = Q[good], (Phyd / Psh)[good]
        qb = np.linspace(0, np.nanpercentile(Q * 1000, 98), 16)
        idx = np.digitize(Q * 1000, qb)
        med = [np.median(eo[idx == i]) for i in range(1, len(qb))
               if (idx == i).sum() >= 5]
        if med:
            rows.append((e["N"], max(med)))
    rows.sort()
    N = np.array([r[0] for r in rows]); eta = np.array([r[1] for r in rows])
    Re = re_u(N)
    aa, cc = np.polyfit(np.log(Re), np.log(1 - eta), 1)
    a = -aa
    resid = np.log(1 - eta) - (cc + aa * np.log(Re))
    a_sig = float(np.sqrt(np.sum(resid ** 2) / max(len(N) - 2, 1)
                          / np.sum((np.log(Re) - np.log(Re).mean()) ** 2)))
    Re0, e0 = Re[-1], eta[-1]      # anchor the extrapolation at the top point
    def extrap(rpm, ai):
        return 1 - (1 - e0) * (Re0 / re_u(rpm)) ** ai
    nn = np.linspace(3000, 21000, 200)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    plot_data(ax, N, eta * 100, ms=5, color="C0",
              label=r"measured peak $\eta$ per run")
    plot_tuned(ax, nn, extrap(nn, a) * 100, color="C0",
               label=rf"fit $(1-\eta)\propto Re^{{-a}}$, $a={a:.2f}\pm{a_sig:.2f}$")
    ax.fill_between(nn, extrap(nn, max(a - a_sig, 0)) * 100,
                    extrap(nn, a + a_sig) * 100, color="C0", alpha=0.15)
    ax.axvline(N_DES, color="k", lw=0.5)
    ax.axhline(ETA_DESIGN_PCT, color="gray", ls=":",
               label=f"design estimate {ETA_DESIGN_PCT:.0f}%")
    ax.set(xlabel="shaft speed [rpm]", ylabel="overall efficiency [%]",
           ylim=(0, 40))
    ax.legend(fontsize=7, loc="upper left")
    print(f"  eta_re: a = {a:.3f} +/- {a_sig:.3f}, "
          f"eta(20k) = {extrap(20000, a) * 100:.1f}% "
          f"({extrap(20000, max(a - a_sig, 0)) * 100:.1f}"
          f"-{extrap(20000, a + a_sig) * 100:.1f}%)")
    return save(fig, "results_eta_re_fit")


def fig_sankey():
    """Shaft-power budget at the measured BEP of the 50% run as a Sankey
    diagram: useful hydraulic power from data; churning from the Barske disk
    correlation x the measured 2.5 multiplier at the run speed; mechanical
    (seal + bearings) from the measured shaft-only parasitic; remainder =
    internal hydraulic losses (nozzle/diffuser/incidence)."""
    from matplotlib.sankey import Sankey
    e = exp_runs()["50%"]
    d = e["d"]; t = d["t"]; m = (t >= 0.5) & (t <= 20)
    Q = d["q"][m] / 1000
    H = ep.head_m(d["pout"][m] - d["pin"][m])
    rpm = d["rpm"][m]; w = rpm * 2 * np.pi / 60; tq = d["tq"][m]
    Phyd = RHO * G * Q * H; Psh = tq * w
    good = (rpm > 2000) & (Psh > 0) & (H > 0) & (Q > 0)
    Q, Ph, Ps = Q[good], Phyd[good], Psh[good]
    qb = np.linspace(0, np.nanpercentile(Q * 1000, 98), 16)
    idx = np.digitize(Q * 1000, qb)
    best, ph_b, ps_b = -1.0, 0.0, 0.0
    for i in range(1, len(qb)):
        mm = idx == i
        if mm.sum() >= 5:
            eta_i = float(np.median(Ph[mm] / Ps[mm]))
            if eta_i > best:
                best = eta_i
                ph_b, ps_b = float(np.median(Ph[mm])), float(np.median(Ps[mm]))
    N = e["N"]
    P_churn = float(barske_disk_power(N) * DISK_MULT)
    P_mech = float(np.interp(N, MR, MP))
    P_int = ps_b - ph_b - P_churn - P_mech
    fr = np.array([ph_b, P_churn, P_mech, P_int]) / ps_b * 100
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    sk = Sankey(ax=ax, scale=0.012, head_angle=130, shoulder=0.02,
                offset=0.3, unit="%", format="%.0f")
    sk.add(flows=[100, -fr[1], -fr[3], -fr[2], -fr[0]],
           labels=["shaft power", "churning", "internal hydraulic",
                   "mechanical", "useful"],
           orientations=[0, 1, 1, -1, 0],
           pathlengths=[0.5, 0.4, 0.3, 0.4, 0.6],
           facecolor="C0", alpha=0.75, lw=0.5)
    sk.finish()
    ax.axis("off")
    print(f"  sankey @ {N:.0f} rpm BEP: shaft {ps_b:.0f} W -> useful {ph_b:.0f} W"
          f" ({fr[0]:.0f}%), churning {P_churn:.0f} W ({fr[1]:.0f}%), "
          f"mech {P_mech:.0f} W ({fr[2]:.0f}%), internal {P_int:.0f} W ({fr[3]:.0f}%)")
    return save(fig, "results_power_sankey")


def fig_cav_suction():
    """Gulich Fig 6.9-style suction test figure: head normalised by the
    fully wetted reference vs NPSHa, one curve per held flow step
    (Q = const), for the two video-validated runs. Successive inlet-pressure
    reduction at constant speed; the 3% head-drop criterion is the line."""
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    curves = []
    for lbl, mk in zip(CAV_RUNS, ("o", "s")):
        e = exp_runs()[lbl]
        cav = ep.analyse_cav(e["tag"], e["d"])
        for s in cav["steps"]:
            if s.get("lowhead") or len(s["cen"]) < 5:
                continue
            curves.append((float(s["q_ref"]), lbl, mk, s))
    curves.sort(key=lambda c: c[0])
    cmap = plt.get_cmap("viridis")
    for i, (qr, lbl, mk, s) in enumerate(curves):
        c = cmap(0.05 + 0.85 * i / max(len(curves) - 1, 1))
        ax.plot(s["cen"], s["Hmed"] / s["H_ref"], marker=mk, ms=2.5, lw=1.0,
                color=c, label=f"{qr:.2f} l/s ({lbl})")
    ax.axhline(0.97, color="C3", lw=0.8, ls="--")
    ax.text(0.98, 0.971, "3% head-drop criterion", fontsize=6.5, color="C3",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom")
    ax.set(xlabel=r"NPSH$_\mathrm{a}$ [m]", ylabel=r"$H/H_\mathrm{ref}$")
    ax.legend(fontsize=6, title="held flow (run)", title_fontsize=6,
              loc="lower right")
    return save(fig, "results_cav_suction")


# =========================================================================== #
# radial-inflow cantilever turbine terminology figure (promoted 2026-06-13 from
# proto_cantilever_fig.py). MoC blade contour with the rotor inlet/exit velocity
# triangles overlaid and aligned to the blade feet (true scale), beta_3/beta_4
# against u. MoC frame: local x = meridional, local y = tangential; LE/TE are
# the two feet of the symmetric impulse bucket.
# (The annular-cascade ring panel is kept below but commented out, 2026-06-13.)
# =========================================================================== #
CANT_BLADE_FILL = "0.85"


def _cant_blade_polygon(moc):
    """Single solid MoC blade silhouette (the _draw_passage crescent), native
    frame x = meridional, y = tangential. LE/TE are the two feet."""
    res = moc.results
    xl, yl = moc.coords["lower_rot"]["x"], moc.coords["lower_rot"]["y"]
    xu, yu = moc.coords["upper_rot"]["x"], moc.coords["upper_rot"]["y"]
    Rl, Ru = res["Rl"], res["Ru"]
    a_l, a_u = res["alpha_lower_inlet"], res["alpha_upper_inlet"]
    y_te = yu[-1] + (xl[-1] - xu[-1]) * np.tan(moc.beta_inlet)

    th_l = np.linspace(np.pi / 2 + a_l, np.pi / 2 - a_l, 100)
    xb = np.concatenate([np.flip(xl), Rl * np.cos(th_l), -xl])
    yb = np.concatenate([np.flip(yl), Rl * np.sin(th_l), yl]) - yl[-1] + y_te

    th_u = np.linspace(np.pi / 2 + a_u, np.pi / 2 - a_u, 100)
    x_up = np.concatenate([[xl[-1]], np.flip(xu), Ru * np.cos(th_u), -xu, [-xl[-1]]])
    y_up = np.concatenate([[y_te], np.flip(yu), Ru * np.sin(th_u), yu, [y_te]])

    bx = np.concatenate([x_up, np.flip(xb)])
    by = np.concatenate([y_up, np.flip(yb)])
    return bx, by


def _cant_oriented_blade(bx, by, chord_units):
    """Blade for the L panel: u (local y) -> horizontal, meridional (local x)
    -> vertical (LE up, TE down), lower-edge-left / upper-edge-right. Mapping
    (x,y)->(y,-x), centred, scaled so the LE-TE span = chord_units."""
    X, Y = by.copy(), -bx.copy()
    X -= X.mean()
    Y -= Y.mean()
    s = chord_units / (Y.max() - Y.min())
    return X * s, Y * s


def _cant_arrow(ax, p0, p1, color, lw=1.6):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                shrinkA=0, shrinkB=0))


def _cant_comp(ax, p0, p1, color):
    """A component leg: dotted shaft ending in a triangle head (shows the
    direction of the tangential / meridional component)."""
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=color,
                                linestyle=":", shrinkA=0, shrinkB=0))


def _cant_triangle_at(ax, A, U, W, suf, cu_lab, wu_lab, c_below):
    """One velocity triangle anchored at the u-tip A (the w_u/w corner).
    u : O->A (O = A - U),  w : A->C (C = A + W),  c : O->C. Each of c and w is
    decomposed into its own dotted right-triangle (tangential + meridional),
    drawn with directed legs (dotted shaft, triangle head). c_below puts the
    c-triangle below its arrow and the w-triangle above; not-c_below mirrors
    it. No gray: c and its parts blue (C0), w and its parts orange (C3), u
    black."""
    cU, cC, cW = "k", "C0", "C3"
    O = A - U
    C = A + W
    right = C[0] >= A[0]
    side = 1 if right else -1
    _cant_arrow(ax, O, A, cU)                     # u
    _cant_arrow(ax, A, C, cW)                     # w
    _cant_arrow(ax, O, C, cC)                     # c
    # main vector labels: c on the lower-left of its arrow, w on the upper-right
    cm, wm = (O + C) / 2, (A + C) / 2
    ax.text(cm[0] - 7 * side, cm[1] - 7, r"$c_%s$" % suf, color=cC, fontsize=11,
            ha="right" if right else "left", va="top")
    ax.text(wm[0] + 7 * side, wm[1] + 5, r"$w_%s$" % suf, color=cW, fontsize=11,
            ha="left" if right else "right", va="bottom")
    ax.text((O[0] + A[0]) / 2, O[1] + 5, r"$u$", color=cU, fontsize=10,
            ha="center", va="bottom")

    # ---- c decomposition right-triangle (blue, directed dotted legs) ----
    if c_below:                                   # legs below the c arrow
        Kc = np.array([O[0], C[1]])               # c_m (down the left), then c_u
        _cant_comp(ax, O, Kc, cC)
        _cant_comp(ax, Kc, C, cC)
        ax.text((Kc[0] + C[0]) / 2, C[1] - 6, cu_lab, color=cC, fontsize=9,
                ha="center", va="top")
        ax.text(O[0] - 4 * side, (O[1] + C[1]) / 2, r"$c_{%s m}$" % suf,
                color=cC, fontsize=9, ha="right" if right else "left",
                va="center")
    else:                                         # legs along the top / C side
        Kc = np.array([C[0], O[1]])               # c_u (along the top), then c_m
        _cant_comp(ax, O, Kc, cC)
        _cant_comp(ax, Kc, C, cC)
        ax.text((O[0] + Kc[0]) / 2, O[1] + 19, cu_lab, color=cC, fontsize=9,
                ha="center", va="bottom")
        ax.text(Kc[0] + 4 * side, (O[1] + C[1]) / 2, r"$c_{%s m}$" % suf,
                color=cC, fontsize=9, ha="left" if right else "right",
                va="center")

    # ---- w decomposition right-triangle (orange, directed dotted legs) ----
    if c_below:                                   # w-triangle above its arrow
        Kw = np.array([C[0], A[1]])               # w_u (along the top), then w_m
        _cant_comp(ax, A, Kw, cW)
        _cant_comp(ax, Kw, C, cW)
        ax.text((A[0] + Kw[0]) / 2, A[1] + 6, wu_lab, color=cW, fontsize=9,
                ha="center", va="bottom")
    else:                                         # w-triangle below its arrow
        Kw = np.array([A[0], C[1]])               # w_m (down), then w_u (bottom)
        _cant_comp(ax, A, Kw, cW)
        _cant_comp(ax, Kw, C, cW)
        ax.text((Kw[0] + C[0]) / 2, C[1] - 6, wu_lab, color=cW, fontsize=9,
                ha="center", va="top")


def _cant_beta(ax, foot, tangent_deg, lab, ext=62, r=30):
    """Extend the LE/TE straight segment to the LEFT of `foot`, draw a leftward
    horizontal reference, and arc the metal angle between them."""
    d = np.deg2rad(tangent_deg)
    dx, dy = np.cos(d), np.sin(d)
    if dx > 0:                                  # force the line to point LEFT
        dx, dy = -dx, -dy
    p1 = foot + ext * np.array([dx, dy])
    ax.plot([foot[0], p1[0]], [foot[1], p1[1]], color="k", lw=1.2)
    ax.plot([foot[0], foot[0] - ext * 0.95], [foot[1], foot[1]],
            color="k", lw=0.8, ls="--")
    da = np.arctan2(dy, dx) - np.pi
    da = (da + np.pi) % (2 * np.pi) - np.pi      # wrap to (-pi, pi]
    th = np.linspace(np.pi, np.pi + da, 40)
    ax.plot(foot[0] + r * np.cos(th), foot[1] + r * np.sin(th),
            color="C3", lw=1.1)
    am = np.pi + da / 2
    ax.text(foot[0] + 1.7 * r * np.cos(am), foot[1] + 1.7 * r * np.sin(am),
            lab, color="C3", fontsize=12, ha="center", va="center")


def _cant_panel_triangles(ax, t, X, Y):
    """MoC blade contour with the rotor inlet (3) and exit (4) velocity
    triangles overlaid and aligned to it. Inlet anchored (u-tip) on the LE,
    exit on the TE; w3 runs along the LE tangent, w4 along the TE tangent, and
    beta_3 / beta_4 are the angles those make with the horizontal (u). True
    scale, displaced left of the blade with dashed leaders back to the feet."""
    vs = 0.20                                    # m/s -> figure units (true scale)
    u, c3u, c3m = float(t.u), float(t.c3u), float(t.c3m)
    c4u = float(t.c4u)
    ax.fill(X, Y, color=CANT_BLADE_FILL, ec="k", lw=1.3, zorder=1)
    le = np.array([X[np.argmax(Y)], Y.max()])    # LE foot (top)
    te = np.array([X[np.argmin(Y)], Y.min()])    # TE foot (bottom)
    U = np.array([u, 0.0]) * vs
    W3 = np.array([c3u - u, -c3m]) * vs           # along LE tangent
    W4 = np.array([c4u - u, -c3m]) * vs           # along TE tangent
    dx = 155.0                                    # shift triangles well left of
    A3 = le - np.array([dx, 0.0])                 # the blade so they clear the
    A4 = te - np.array([dx, 0.0])                 # beta_3 / beta_4 arcs
    ax.plot([le[0], A3[0]], [le[1], A3[1]], color="k", ls="--", lw=0.6, zorder=0)
    ax.plot([te[0], A4[0]], [te[1], A4[1]], color="k", ls="--", lw=0.6, zorder=0)
    _cant_triangle_at(ax, A3, U, W3, "3", r"$c_{3u}$", r"$w_{3u}$", c_below=True)
    _cant_triangle_at(ax, A4, U, W4, "4", r"$-c_{4u}$", r"$-w_{4u}$", c_below=False)
    n = len(X)
    i_le, i_te = int(np.argmax(Y)), int(np.argmin(Y))

    def tang(i):
        a, b = X[(i - 3) % n], X[(i + 3) % n]
        c, d = Y[(i - 3) % n], Y[(i + 3) % n]
        return np.rad2deg(np.arctan2(d - c, b - a))

    _cant_beta(ax, le, tang(i_le), r"$\beta_3$")
    _cant_beta(ax, te, tang(i_te), r"$\beta_4$")
    ax.text(A3[0], A3[1] + 38, "rotor inlet (3)", fontsize=8, ha="center")
    ax.text(A4[0] + 30, A4[1] - 48, "rotor exit (4)", fontsize=8, ha="center")
    # the triangle arrows are FancyArrowPatch annotations, which do NOT drive
    # autoscale and get clipped to the view, so set the limits from all the
    # arrow/blade extents (else the exit triangle, sitting furthest left, is cut).
    pts = [[X.min(), Y.min()], [X.max(), Y.max()],
           [le[0] - 62, le[1]], [te[0] - 62, te[1]]]
    for A, W in ((A3, W3), (A4, W4)):
        O, C = A - U, A + W
        pts += [list(O), list(A), list(C),
                [C[0], O[1]], [O[0], C[1]], [A[0], C[1]]]
    pts = np.array(pts, float)
    ax.set_xlim(pts[:, 0].min() - 8, pts[:, 0].max() + 8)
    ax.set_ylim(pts[:, 1].min() - 10, pts[:, 1].max() + 10)
    ax.set_aspect("equal")
    ax.axis("off")


# --- annular cascade ring panel: commented out 2026-06-13 (Martin), kept for
# possible later use. Re-enable the constants, this function, and the second
# axis in fig_cantilever() to bring it back.
# CANT_N_BLADES = 50
# CANT_D_IN_MM, CANT_D_MEAN_MM, CANT_D_OUT_MM = 85.0, 95.0, 105.0
#
# def _cant_panel_ring(ax, Xo, Yo):
#     """Annular cascade. Xo,Yo = oriented blade (LE at +Y, TE at -Y, meridional
#     vertical), the same silhouette the left panel shows. Each blade is placed
#     with its OWN radial axis: LE (+Y) -> outer diameter, TE (-Y) -> inner
#     diameter. Uniform scale (shape preserved); per-blade rotation is the
#     radial heading."""
#     r_in, r_mean, r_out = CANT_D_IN_MM / 2, CANT_D_MEAN_MM / 2, CANT_D_OUT_MM / 2
#     band = r_out - r_in
#     s = 0.98 * band / (Yo.max() - Yo.min())       # meridional span -> radial band
#     bx_r, by_r = Xo * s, Yo * s                    # by_r -> radial, bx_r -> tangential
#
#     for k in range(CANT_N_BLADES):
#         th = 2 * np.pi * k / CANT_N_BLADES
#         radial = r_mean + by_r                     # LE outer, TE inner
#         tang = bx_r
#         gx = radial * np.cos(th) - tang * np.sin(th)
#         gy = radial * np.sin(th) + tang * np.cos(th)
#         ax.fill(gx, gy, color=CANT_BLADE_FILL, ec="k", lw=0.5, zorder=2)
#
#     th = np.linspace(0, 2 * np.pi, 400)
#     for r, ls in ((r_in, "-"), (r_mean, (0, (5, 4))), (r_out, "-")):
#         ax.plot(r * np.cos(th), r * np.sin(th), color="0.25", ls=ls, lw=0.9,
#                 zorder=1)
#     for r, lab, ang in ((r_in, r"$d_i$", -90), (r_mean, r"$d_m$", -66),
#                         (r_out, r"$d_o$", -50)):
#         a = np.deg2rad(ang)
#         ax.annotate(lab, xy=(r * np.cos(a), r * np.sin(a)),
#                     xytext=(1.34 * r_out * np.cos(a), 1.34 * r_out * np.sin(a)),
#                     fontsize=11, ha="center", va="center", color="0.2",
#                     arrowprops=dict(arrowstyle="->", lw=0.7, color="0.45"))
#     ax.text(0, 1.18 * r_out, f"{CANT_N_BLADES} blades", fontsize=8, ha="center")
#     ax.set_aspect("equal")
#     ax.axis("off")


def fig_cantilever():
    """Cantilever turbine terminology: rotor inlet/exit velocity triangles on
    the MoC blade contour (promoted from proto_cantilever_fig)."""
    t = turbine_design()
    moc = moc_design()
    bx, by = _cant_blade_polygon(moc)
    X_tri, Y_tri = _cant_oriented_blade(bx, by, chord_units=100)    # contour + triangles
    print(f"  cantilever: u={t.u:.1f}  c3={t.c3:.1f}  c3u={t.c3u:.1f}  "
          f"c3m={t.c3m:.1f}  c4u={t.c4u:.1f}  c4={t.c4:.1f}")
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    _cant_panel_triangles(ax, t, X_tri, Y_tri)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return save(fig, "turbine_cantilever")
    # ring panel (commented out 2026-06-13): to restore the two-panel version,
    # use plt.subplots(1, 2, figsize=(12, 5.0),
    #                  gridspec_kw=dict(width_ratios=[1.5, 1.4]))
    # and call _cant_panel_ring(axes[1], -X_ring, Y_ring) with
    # X_ring, Y_ring = _cant_oriented_blade(bx, by, chord_units=1.0)


# =========================================================================== #
# SP-8107 / Logan (Handbook of Turbomachinery) efficiency-vs-specific-speed pump
# map, re-annotated with a second metric n_q axis (Martin 06-14). The scanned
# chart's x-axis is stage specific speed N_s in US units (rpm, gpm, ft); the
# thesis works in metric n_q (rpm, m^3/s, m, intro eq.1), so a converted log axis
# is drawn below the original.
#   x_px = NS_X0 + NS_PX_DEC*log10(N_s)   fitted from the five power-of-ten N_s
#   labels in the bitmap (decade spacing 296 px, residuals a few px).
#   N_s(US) = NS_PER_NQ * n_q,  NS_PER_NQ = 51.64 (gpm,ft -> m^3/s,m).
# Sanity on the unit basis: the chart's centrifugal optimum sits at N_s ~2000-3000,
# i.e. n_q ~40-60, the known metric centrifugal band. If a future source uses a
# different N_s definition, change NS_PER_NQ only.
# =========================================================================== #
PUMP_MAP_SRC = os.path.join(_ROOT, "report", "figs", "pump_map_source.png")
NS_X0, NS_PX_DEC = 103.4, 296.05        # N_s=1 x-pixel; pixels per N_s decade
NS_PER_NQ = 51.64                       # N_s(US gpm,ft) = NS_PER_NQ * n_q(metric)
PMAP_CROP_Y = 818                       # keep plot + N_s numbers; drop old subtitle/caption
PMAP_EXTRA = 180                        # white rows added below for the n_q axis
PMAP_XSPAN = (100, 1494)                # plot-box left/right x-pixels for the n_q line
PMAP_NQ_TICKS = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
PMAP_FONT_NUM, PMAP_FONT_LAB = 9, 10    # n_q tick-number / axis-label font sizes


def fig_pump_map_nq():
    """SP-8107 / Logan efficiency-vs-specific-speed pump map with a second log
    axis added below, converting the chart's US N_s (rpm, gpm, ft) to metric n_q
    (rpm, m^3/s, m). Scanned bitmap re-annotated; the original subtitle and
    'Figure 18' caption are cropped (the LaTeX \\caption replaces them).
    Calibration and the N_s<->n_q factor live in the PMAP_*/NS_* constants above;
    tweak NS_X0/NS_PX_DEC if the n_q ticks drift off the N_s ticks."""
    from PIL import Image
    img = np.asarray(Image.open(PUMP_MAP_SRC).convert("RGB"))[:PMAP_CROP_Y]
    H, W, _ = img.shape
    canvas = np.full((H + PMAP_EXTRA, W, 3), 255, np.uint8)
    canvas[:H] = img

    def nq_to_x(nq):
        return NS_X0 + NS_PX_DEC * np.log10(NS_PER_NQ * np.asarray(nq, float))

    xL, xR = PMAP_XSPAN
    y_axis, y_num, y_lab = H + 52, H + 78, H + 114   # n_q line / numbers / label rows
    fig, ax = plt.subplots(figsize=(6.5, 6.5 * (H + PMAP_EXTRA) / W))
    ax.imshow(canvas)
    ax.set(xlim=(0, W), ylim=(H + PMAP_EXTRA, 0))
    ax.axis("off")
    # N_s unit reminder where the cropped subtitle was
    ax.text((xL + xR) / 2, H + 10, r"Stage specific speed, $N_s$  (rpm, gpm, ft)",
            ha="center", va="center", fontsize=PMAP_FONT_LAB)
    ax.plot([xL, xR], [y_axis, y_axis], color="k", lw=1.2)
    for nq in PMAP_NQ_TICKS:
        x = float(nq_to_x(nq))
        if x < xL - 1 or x > xR + 1:
            continue
        dec = abs(np.log10(nq) - round(np.log10(nq))) < 1e-6   # power of ten
        ax.plot([x, x], [y_axis, y_axis + (13 if dec else 8)], color="k", lw=1.2)
        ax.text(x, y_num, "%g" % nq, ha="center", va="center",
                fontsize=PMAP_FONT_NUM, fontweight=("bold" if dec else "normal"))
    ax.text((xL + xR) / 2, y_lab, r"Metric specific speed, $n_q$  (rpm, m$^3$/s, m)",
            ha="center", va="center", fontsize=PMAP_FONT_LAB)
    print(f"  pump_map_nq: n_q=1 -> N_s={NS_PER_NQ:.0f} -> x={nq_to_x(1):.0f}px;  "
          f"n_q=10 -> N_s={NS_PER_NQ*10:.0f} -> x={nq_to_x(10):.0f}px")
    return save(fig, "pump_map_nq")


# =========================================================================== #
# Balje (Turbomachines, 1981) maximum-efficiency vs DIMENSIONLESS design specific
# speed N_D pump map (Martin 06-14). Cleaner than the SP-8107 chart: a single
# "partial emission pumps" curve, no Barske/PE split. N_D is Balje's universal
# specific speed (omega*sqrt(Q)/(g*H)^0.75, dimensionless; centrifugal optimum at
# N_D ~1). The thesis works in metric n_q, so a converted log axis is drawn below.
#   x_px = ND_X0 + ND_PX_DEC*log10(N_D)   least-squares fit over the plot borders
#   (0.01@77px, 40@633px) + the 0.1/1/10 labels; residuals < 5 px, decade 154 px.
#   n_q = NQ_PER_ND * N_D,  NQ_PER_ND = (60/2*pi)*g^0.75 = 52.93 (g=9.81).
# >>> TODO(Martin): cross-check NQ_PER_ND and N_D's exact definition against
#     Balje's book tomorrow; change NQ_PER_ND only if his N_D differs. <<<
# =========================================================================== #
PUMP_MAP6_SRC = os.path.join(_ROOT, "report", "figs", "pump_map_balje_source.png")
ND_X0, ND_PX_DEC = 383.57, 154.115      # N_D=1 x-pixel; pixels per N_D decade
NQ_PER_ND = 52.93                       # n_q = NQ_PER_ND * N_D (dimensionless -> metric)
PMAP6_CROP_Y = 422                      # keep plot + N_D numbers; drop axis title
PMAP6_EXTRA = 150                       # white rows added below for the n_q axis
PMAP6_XSPAN = (77, 633)                 # plot-box left/right x-pixels (N_D 0.01..40)
PMAP6_NQ_TICKS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
PMAP6_FONT_NUM, PMAP6_FONT_LAB = 9, 10  # n_q tick-number / axis-label font sizes


def fig_pump_map_balje():
    """Balje dimensionless-N_D pump efficiency map with a second log axis added
    below converting N_D to metric n_q (n_q = 52.93*N_D). Cleaner alternative to
    pump_map_nq: one partial-emission curve, no Barske/PE ambiguity. Scanned
    bitmap re-annotated; the original axis title is cropped (LaTeX \\caption
    replaces it). Calibration/conversion in the PMAP6_*/ND_*/NQ_PER_ND constants;
    NQ_PER_ND pending a cross-check against Balje's book."""
    from PIL import Image
    img = np.asarray(Image.open(PUMP_MAP6_SRC).convert("RGB"))[:PMAP6_CROP_Y]
    H, W, _ = img.shape
    canvas = np.full((H + PMAP6_EXTRA, W, 3), 255, np.uint8)
    canvas[:H] = img

    def nq_to_x(nq):
        return ND_X0 + ND_PX_DEC * np.log10(np.asarray(nq, float) / NQ_PER_ND)

    xL, xR = PMAP6_XSPAN
    y_axis, y_num, y_lab = H + 44, H + 68, H + 104   # n_q line / numbers / label rows
    fig, ax = plt.subplots(figsize=(6.5, 6.5 * (H + PMAP6_EXTRA) / W))
    ax.imshow(canvas)
    ax.set(xlim=(0, W), ylim=(H + PMAP6_EXTRA, 0))
    ax.axis("off")
    # N_D reminder where the cropped axis title was
    ax.text((xL + xR) / 2, H + 8, r"Design specific speed, $N_D$  (dimensionless)",
            ha="center", va="center", fontsize=PMAP6_FONT_LAB)
    ax.plot([xL, xR], [y_axis, y_axis], color="k", lw=1.2)
    for nq in PMAP6_NQ_TICKS:
        x = float(nq_to_x(nq))
        if x < xL - 1 or x > xR + 1:
            continue
        dec = abs(np.log10(nq) - round(np.log10(nq))) < 1e-6   # power of ten
        ax.plot([x, x], [y_axis, y_axis + (12 if dec else 7)], color="k", lw=1.2)
        ax.text(x, y_num, "%g" % nq, ha="center", va="center",
                fontsize=PMAP6_FONT_NUM, fontweight=("bold" if dec else "normal"))
    ax.text((xL + xR) / 2, y_lab, r"Metric specific speed, $n_q$  (rpm, m$^3$/s, m)",
            ha="center", va="center", fontsize=PMAP6_FONT_LAB)
    print(f"  pump_map_balje: n_q=10 -> N_D={10/NQ_PER_ND:.3f} -> x={nq_to_x(10):.0f}px;  "
          f"n_q=50 -> N_D={50/NQ_PER_ND:.2f} -> x={nq_to_x(50):.0f}px")
    return save(fig, "pump_map_balje")


# =========================================================================== #
FIGURES = {
    "pump_map_balje": fig_pump_map_balje,  # intro: Balje N_D map + n_q axis (preferred)
    "pump_map_nq": fig_pump_map_nq,        # intro: SP-8107 map + n_q axis (backup)
    "theory_hq": fig_theory_hq,            # H-Q + psi-phi pair (keep combined)
    "eta_phi": fig_eta_phi,
    "churning": fig_churning,              # disk_mult corroboration
    "hq_extrap": fig_hq_extrap,            # objective-3 figure
    "npshr_vs_q": fig_npshr_vs_q,
    "coupled_hq": fig_coupled_hq,
    "coupled_torque": fig_coupled_torque,
    "turbine_eta_uc0": fig_turbine_eta_uc0,
    "turbine_design_space": fig_turbine_design_space,  # (d_m, mdot) sweep + constraints
    # diagram / design / validation
    "moc_contour": fig_moc_contour,        # as-designed passage, labelled
    "moc_spread": fig_moc_spread,          # 2 PDFs: narrow vs wide Mach spread
    "moc_separation": fig_moc_separation,  # Hi on as-designed blade vs 1.8-2.4
    "moc_displaced": fig_moc_displaced,    # ideal vs delta*-displaced contour
    "barske_geometry": fig_barske_geometry,  # 1 PDF: meridional + end view, shared y
    "goldman_validation": fig_goldman_validation,  # 2 PDFs: Mach + Hi replication
    "turbine_triangles": fig_turbine_triangles,  # design-point velocity triangles
    "forced_vortex_triangles": fig_forced_vortex_triangles,  # pump Euler-head nomenclature
    "cantilever": fig_cantilever,          # cantilever turbine terminology (triangles + ring)
    "campbell": fig_campbell,              # turbine shaft Campbell diagram (ross)
    "valve_kv": fig_valve_kv,              # valve Kv: online data vs measured
    "eta_re": fig_eta_re,                  # fitted (1-eta) ~ Re^-a + 20k extrap
    "sankey": fig_sankey,                  # BEP shaft-power budget (Sankey)
    "cav_suction": fig_cav_suction,        # Gulich 6.9-style suction curves
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
