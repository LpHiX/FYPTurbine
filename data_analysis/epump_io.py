"""
epump_io.py — reusable core for e-pump (Barske) test-data analysis.

This module owns everything that must be trustworthy and identical across the
interactive notebook (prelim_analysis.ipynb) and the batch figure regenerator
(make_figures.py):

    - H5 loading (BoardStateLogger runs)               -> load(), find_runs(), build_catalog()
    - unit/derived helpers                             -> head_m(), smooth()
    - H-Q ramp analysis (affinity-normalised)          -> analyse_hq()
    - cavitation NPSHr-at-3%-breakdown analysis        -> analyse_cav()

NOTHING here plots or writes files — drivers do that. Edit channel names,
constants, or analysis logic in ONE place: here.

Confirmed 2026-06-08: adc_pt_*_mv channels are already scaled to BAR (legacy
name), so head_m() takes bar directly. If that ever changes, fix it here only.
"""
from __future__ import annotations
import os
import re
import glob
import numpy as np
import pandas as pd
import h5py

# --------------------------------------------------------------------------- #
# Constants (water @ ~20 C)
# --------------------------------------------------------------------------- #
RHO = 998.0          # kg/m3
G = 9.81             # m/s2
PATM = 1.013         # bar, absolute atmospheric
PV = 0.0234          # bar, water vapour pressure @ 20 C
BAR2PA = 1.0e5

# raw h5 channel name -> short key used everywhere downstream.
# Single source of truth: rename a firmware channel -> edit one line.
CHANNELS = {
    "rpm":     "adc_rpm_mv",
    "pin":     "adc_pt_in_mv",
    "pout":    "adc_pt_out_mv",
    "ppump":   "adc_pt_pump_mv",
    "tq":      "adc_torque_mv",
    "cur":     "adc_motor_current_mv",
    "escv":    "adc_esc_v_mv",
    "out_dem": "servos_pumpoutlet_angle_demand",
    "in_dem":  "servos_pumpinlet_angle_demand",
    "mot_dem": "servos_motor_angle_demand",
}
_TIME_CHANNEL = "fms_fm0_flowrate"   # master clock + flow rate (l/s)

_TAG_RE = re.compile(r"test_(\d{8}_\d{6})_HotfireLog\.h5$")


# --------------------------------------------------------------------------- #
# Unit / smoothing helpers
# --------------------------------------------------------------------------- #
def head_m(dp_bar):
    """Pressure difference (bar) -> head (m of water)."""
    return dp_bar * BAR2PA / (RHO * G)


def smooth(x, w=15):
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


# --------------------------------------------------------------------------- #
# Loading / cataloguing
# --------------------------------------------------------------------------- #
def find_runs(logdir):
    """{tag -> full path} for every test_*_HotfireLog.h5 in logdir.
    tag is the full 'YYYYMMDD_HHMMSS' stamp (unique across campaigns)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(logdir, "test_*_HotfireLog.h5"))):
        m = _TAG_RE.search(os.path.basename(p))
        if m:
            out[m.group(1)] = p
    return out


def load(path):
    """One H5 run -> dict of named float arrays on the flow-rate time base.
    Channel set is fixed by CHANNELS. 'notes' carries config/notes."""
    with h5py.File(path, "r") as h:
        def ch(name):
            g = h["channels"][name]
            return np.asarray(g["time"][:], float), np.asarray(g["data"][:], float)

        t, q = ch(_TIME_CHANNEL)
        d = {"t": t, "q": q}
        for key, raw in CHANNELS.items():
            d[key] = ch(raw)[1]
        try:
            d["notes"] = h["config/notes"][()].decode()
        except Exception:
            d["notes"] = ""
    m = _TAG_RE.search(os.path.basename(path))
    d["tag"] = m.group(1) if m else os.path.basename(path)
    return d


def load_runs(logdir, tags=None):
    """Eager-load every (or selected) run in logdir into {tag -> data dict}."""
    runs = find_runs(logdir)
    if tags is not None:
        runs = {t: p for t, p in runs.items() if t in set(tags)}
    return {tag: load(path) for tag, path in runs.items()}


def build_catalog(logdir):
    """Cheap metadata table over a campaign dir — your coverage map.
    Reads only the flow time vector, rpm, and notes (no full-run loads here
    beyond rpm, which is small). Returns a DataFrame sorted by tag."""
    rows = []
    for tag, p in find_runs(logdir).items():
        with h5py.File(p, "r") as h:
            t = np.asarray(h["channels"][_TIME_CHANNEL]["time"][:], float)
            rpm = np.asarray(h["channels"]["adc_rpm_mv"]["data"][:], float)
            try:
                notes = h["config/notes"][()].decode()
            except Exception:
                notes = ""
        spun = rpm[rpm > 2000]
        rows.append(dict(
            tag=tag,
            duration_s=float(t[-1] - t[0]) if len(t) else np.nan,
            n=len(t),
            rpm_med=float(np.nanmedian(spun)) if spun.size else np.nan,
            notes=notes,
            path=p,
        ))
    return pd.DataFrame(rows).sort_values("tag").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# H-Q ramp analysis
# --------------------------------------------------------------------------- #
def analyse_hq(tag, d, t_lo=0.5, t_hi=21.5, spin_frac=0.75, n_bins=28):
    """Discharge-valve closing ramp at ~fixed ESC throttle. pt_out is the true
    pump discharge, so H = head_m(pout - pin) is the real developed head.
    The ESC has no speed loop -> RPM drifts ~8% over the ramp, so normalise to
    a reference speed via the affinity laws (H ~ N^2, Q ~ N) for a clean curve.
    The curve legitimately crosses H=0 at the pump's runout flow (pressurised
    feed drives flow past it when the valve is wide open).

    t_lo/t_hi bound the ramp window — CAMPAIGN-SPECIFIC, retune for new data.
    """
    t = d["t"]
    rpm_all = d["rpm"]
    Nref = float(np.nanmedian(rpm_all[rpm_all > 2000]))
    m = (t >= t_lo) & (t <= t_hi) & (rpm_all > spin_frac * Nref)
    if m.sum() < 50:
        return None
    rpm = rpm_all[m]
    s = Nref / rpm
    q = d["q"][m] * s                                   # Q ~ N
    H = head_m(d["pout"][m] - d["pin"][m]) * s ** 2     # H ~ N^2
    qraw = d["q"][m]
    Hraw = head_m(d["pout"][m] - d["pin"][m])
    qb = np.linspace(0, np.nanpercentile(q, 99.5), n_bins)
    Hb, qc, idx = [], [], np.digitize(q, qb)
    for i in range(1, len(qb)):
        sel = idx == i
        if sel.sum() >= 3:
            qc.append(np.nanmedian(q[sel]))
            Hb.append(np.nanmedian(H[sel]))
    qc, Hb = np.array(qc), np.array(Hb)
    q_runout = np.nan
    for i in range(len(Hb) - 1):
        if Hb[i] >= 0 >= Hb[i + 1]:
            q_runout = float(np.interp(0, [Hb[i + 1], Hb[i]], [qc[i + 1], qc[i]]))
            break
    return dict(q=qraw, H=Hraw, qn=q, Hn=H, rpm=Nref,
                qbin=qc, Hbin=Hb,
                H_shutoff=float(np.nanmax(Hb)) if len(Hb) else np.nan,
                q_runout=q_runout, q_max=float(np.nanmax(qraw)))


# --------------------------------------------------------------------------- #
# Cavitation analysis
# --------------------------------------------------------------------------- #
def plateau_segments(out_dem, t, min_hold=2.0):
    """Time windows where outlet demand is held constant (the matrix steps).
    Returns list of (level, i0, i1)."""
    od = np.round(out_dem)
    segs = []
    i = 0
    n = len(od)
    while i < n:
        if np.isnan(od[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and (np.isnan(od[j + 1]) or od[j + 1] == od[i]):
            j += 1
        if t[j] - t[i] > min_hold:
            segs.append((float(od[i]), i, j))
        i = j + 1
    return segs


def analyse_cav(tag, d, cav_start=23.0, settle=1.2, drop_pct=0.97):
    """For each held outlet step (>0), the inlet valve sweeps and drops NPSH.
    Build head vs NPSHa on a binned (median) curve, take the fully-wetted
    reference head at high NPSHa, and find the 3% breakdown -> NPSHr on the
    descending branch. Steps near shutoff (Q~0) are skipped.

    cav_start = time the cavitation matrix begins — CAMPAIGN-SPECIFIC.
    """
    t = d["t"]
    cav_mask = t > cav_start
    pin = d["pin"]; pout = d["pout"]; q = d["q"]
    H = head_m(pout - pin)
    npsha = head_m((pin + PATM - PV))     # velocity head neglected (v^2/2g << 1 m)
    segs = [s for s in plateau_segments(d["out_dem"], t)
            if s[0] >= 5 and t[s[1]] > cav_start]
    results = []
    for level, i0, i1 in segs:
        t0 = t[i0]
        sl = (t > t0 + settle) & (t <= t[i1])
        Hs, ns, qs = H[sl], npsha[sl], q[sl]
        good = np.isfinite(Hs) & np.isfinite(ns) & np.isfinite(qs)
        Hs, ns, qs = Hs[good], ns[good], qs[good]
        if len(Hs) < 40:
            continue
        q_ref = float(np.nanmedian(qs))
        if q_ref < 0.03:               # shutoff: no meaningful suction sweep
            continue
        nb = np.linspace(np.nanmin(ns), np.nanmax(ns), 20)
        cen, Hmed = [], []
        di = np.digitize(ns, nb)
        for b in range(1, len(nb)):
            m = di == b
            if m.sum() >= 4:
                cen.append(float(np.nanmedian(ns[m])))
                Hmed.append(float(np.nanmedian(Hs[m])))
        if len(cen) < 5:
            continue
        cen = np.array(cen); Hmed = np.array(Hmed)
        hi = cen >= np.nanpercentile(cen, 70)
        H_ref = float(np.nanmedian(Hmed[hi]))
        if H_ref < 2.0:
            results.append(dict(level=level, H_ref=H_ref, q_ref=q_ref,
                                npshr=np.nan, npsha_min=float(np.nanmin(ns)),
                                broke=False, lowhead=True,
                                cen=cen, Hmed=Hmed, n=ns, H=Hs))
            continue
        thr = drop_pct * H_ref
        # Breakdown anchored at the LOW-NPSHa end: walk upward from the lowest
        # NPSHa bin; NPSHr = highest NPSHa still suppressed below thr before
        # head recovers. If the lowest bin is already above thr, the pump never
        # broke down within the achieved suction range.
        asc = np.argsort(cen)
        npshr = np.nan
        if Hmed[asc[0]] < thr:
            for k in asc:
                if Hmed[k] < thr:
                    npshr = float(cen[k])
                else:
                    break
        broke = bool(np.isfinite(npshr))
        nss = np.nan
        if broke and npshr > 0:
            nss = float(np.nanmedian(d["rpm"][cav_mask]) *
                        np.sqrt(max(q_ref, 0) / 1000.0) / npshr ** 0.75)
        results.append(dict(level=level, H_ref=H_ref, q_ref=q_ref, lowhead=False,
                            npshr=npshr, npsha_min=float(np.nanmin(ns)),
                            broke=broke, nss=nss,
                            cen=cen, Hmed=Hmed, n=ns, H=Hs))
    return dict(rpm=float(np.nanmedian(d["rpm"][cav_mask])), steps=results)
