"""
cav_video.py — transparent-window cavitation video annotation pipeline.

Workflow:
    1. make_templates(...)         -> emits a blank JSON per run (pre-keyed to the
                                      hard-coded test sequence). You fill it while
                                      watching the video.
    2. (you watch, type onset times + a few cavity radii, read off the on-screen
        backend clock — which IS the data time base)
    3. onset_table(doc, d) / radius_table(doc, d)  -> tidy DataFrames wired to the
                                      synced H5 data (NPSHa at inception, r_cav etc.)

The test sequence is IDENTICAL every run (only esc throttle changes, already
captured in RUNS). It is hard-coded below — do NOT parse hotfiresequence.json.

Two cavitation sites, logged separately:
    eye    = impeller inlet, forced-vortex centre  -> r_cav^2 = 2(p_vap - p0)/(rho w^2)
    throat = diffuser throat                        -> Lock cutoff
"""
from __future__ import annotations
import os
import re
import json
import numpy as np
import pandas as pd

import epump_io as ep

# --------------------------------------------------------------------------- #
# Hard-coded test sequence (identical every run; only esc throttle varies)
# --------------------------------------------------------------------------- #
HQ_WINDOW = (0.0, 20.0)          # PumpOutlet ramps 0->180 : H-Q flow sweep

# outlet demand (fixed-Q step) -> inlet down-sweep window [s] (NPSH dropping).
# outlet 20 = most open = highest flow ; 160 = most closed = lowest flow.
CAV_STEPS = {
    20:  (25.0, 45.0),
    40:  (50.0, 70.0),
    60:  (75.0, 95.0),
    80:  (100.0, 120.0),
    100: (125.0, 145.0),
    120: (150.0, 170.0),
    140: (175.0, 195.0),
    160: (200.0, 220.0),
}


# --------------------------------------------------------------------------- #
# 1. Template generation
# --------------------------------------------------------------------------- #
def _blank_block(window=None, n_radii=5):
    # Fill the numbers; leave unused fields null and they're ignored.
    # onset times are frame-perfect single values (backend seconds).
    # radii: at chosen times, the three eye boundaries (px DIAMETERS, outer->inner):
    #   d_outer_px = outer bubble edge  (bubbles | water)        -> p = p_vap isobar
    #   d_mid_px   = bubble-layer edge  (dense bubbles | sparse) -> intermediate isobar
    #   d_core_px  = full-air core edge (air | bubbles)          -> full-vaporisation isobar
    blk = {}
    if window is not None:
        blk["window_s"] = f"scrub {window[0]:.0f}-{window[1]:.0f} s"   # backend time, read-only ref
    blk["eye_onset_t"] = None        # first bubble at impeller centre
    blk["throat_onset_t"] = None     # first bubble at diffuser throat
    blk["radii"] = [{"t": None, "d_outer_px": None, "d_mid_px": None, "d_core_px": None}
                    for _ in range(n_radii)]
    blk["note"] = ""
    return blk


def make_template(run_tag, path, hex_px=None, hex_mm=95.6, ref=""):
    """Write a blank annotation JSON pre-keyed to the sequence for one run.
    Scale set by a reference hexagon: enter its length in px (hex_px); known
    CAD length hex_mm (default 95.6 mm) -> px_per_mm = hex_px / hex_mm."""
    doc = {
        "run_tag": run_tag,
        "clock_offset_s": 0.0,                       # video clock - backend clock (s)
        "calibration": {"hex_px": hex_px, "hex_mm": hex_mm, "ref": ref},
        "hq": _blank_block(HQ_WINDOW),               # throat cutoff visible on the H-Q ramp
        "steps": {str(o): _blank_block(w) for o, w in CAV_STEPS.items()},
    }
    _write_json(doc, path)
    return path


_RADIUS_RE = re.compile(
    r'\{\s*"t":\s*(.+?),\s*"d_outer_px":\s*(.+?),\s*"d_mid_px":\s*(.+?),'
    r'\s*"d_core_px":\s*(.+?)\s*\}', re.DOTALL)


def _write_json(doc, path):
    """Pretty JSON, but each radius row collapsed onto one line."""
    s = json.dumps(doc, indent=2)
    s = _RADIUS_RE.sub(
        r'{"t": \1, "d_outer_px": \2, "d_mid_px": \3, "d_core_px": \4}', s)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


def make_templates(run_tags, out_dir, calibration=None):
    """One template per run. calibration = {run_tag: {'hex_px':.., 'ref':..}}."""
    os.makedirs(out_dir, exist_ok=True)
    calibration = calibration or {}
    paths = []
    for tag in run_tags:
        cal = calibration.get(tag, {})
        p = os.path.join(out_dir, f"cav_{tag}.json")
        if os.path.exists(p):
            print(f"skip (exists): {p}")            # never clobber filled-in work
        else:
            make_template(tag, p, cal.get("hex_px"), cal.get("hex_mm", 95.6), cal.get("ref", ""))
            print(f"wrote {p}")
        paths.append(p)
    return paths


def load_annotations(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# 2. Map annotations -> synced physical quantities
# --------------------------------------------------------------------------- #
def _idx(d, t):
    return int(np.argmin(np.abs(d["t"] - t)))


def _npsha_arr(pin_bar_gauge):
    """NPSHa [m] from gauge inlet pressure [bar] (matches analyse_cav)."""
    return ep.head_m(np.asarray(pin_bar_gauge, float) + ep.PATM - ep.PV)


def onset_table(doc, d):
    """Frame-perfect visual inception per site -> NPSHa_i at that instant.
    Returns one row per (step, site)."""
    off = doc.get("clock_offset_s", 0.0)
    rows = []

    def add(step_label, site, t):
        if t is None:
            return
        tt = t + off
        i = _idx(d, tt)
        pin = float(d["pin"][i]); rpm = float(d["rpm"][i])
        rows.append(dict(step=step_label, site=site, t=tt,
                         p_in_bar=pin, rpm=rpm, npsha_i=float(_npsha_arr(pin))))

    blocks = [("hq", doc.get("hq"))] + [(int(k), v) for k, v in doc.get("steps", {}).items()]
    for label, blk in blocks:
        if not blk:
            continue
        add(label, "eye", blk.get("eye_onset_t"))
        add(label, "throat", blk.get("throat_onset_t"))
    return pd.DataFrame(rows)


def radius_table(doc, d):
    """Each cavity-radius measurement -> r_cav [m] + forced-vortex fit columns.

    Forced vortex (solid-body rotation):  p(r) = p0 + 0.5 rho w^2 r^2, so the
    cavity (p < p_vap) has radius  r_cav^2 = 2(p_vap - p0)/(rho w^2).
    p0 (eye-centre pressure) sits below the pt_in tap by an unknown core
    depression Delta. Rearranged into a form robust to that offset:

        fv_y = r_cav^2 * w^2   vs   fv_x = (p_vap - p_in_abs)
        => linear: slope = 2/rho (~2e-3, validates forced vortex)
                   intercept * rho/2 = core depression Delta [Pa] below pt_in
    """
    cal = doc.get("calibration") or {}
    hex_px = cal.get("hex_px")
    hex_mm = cal.get("hex_mm", 95.6)
    ppm = (hex_px / hex_mm) if hex_px else None     # px per mm from the reference hexagon
    off = doc.get("clock_offset_s", 0.0)
    pv_pa = ep.PV * 1e5
    rows = []

    def add(step_label, radii):
        for r in radii or []:
            if r.get("t") is None:
                continue
            t = r["t"] + off
            i = _idx(d, t)
            pin = float(d["pin"][i]); rpm = float(d["rpm"][i])
            omega = 2 * np.pi * rpm / 60.0
            pin_abs_pa = (pin + ep.PATM) * 1e5
            # one row per visible boundary -> long format, fit each separately
            for boundary, key in (("outer", "d_outer_px"), ("mid", "d_mid_px"),
                                  ("core", "d_core_px")):
                dpx = r.get(key)
                if dpx is None:
                    continue
                d_mm = (dpx / ppm) if ppm else np.nan     # measured DIAMETER
                r_m = (d_mm / 2.0) / 1000.0               # -> radius [m]
                rows.append(dict(step=step_label, boundary=boundary, t=t,
                                 p_in_bar=pin, rpm=rpm, omega=omega,
                                 d_px=dpx, r_mm=d_mm / 2.0, r_m=r_m, r_cav2=r_m ** 2,
                                 fv_x=(pv_pa - pin_abs_pa),          # Pa
                                 fv_y=(r_m ** 2 * omega ** 2)))      # m^2/s^2
    if doc.get("hq"):
        add("hq", doc["hq"].get("radii"))
    for k, v in doc.get("steps", {}).items():
        add(int(k), v.get("radii"))
    return pd.DataFrame(rows)

if __name__ == "__main__":

    import sys, os
    sys.path.insert(0, r'C:\Users\Martin\Active\FYPTurbine\data_analysis')
    import cav_video as cv, epump_io as ep
# the full-test runs that have a cavitation matrix (esc throttle hard-coded in RUNS / your notebook)
    TAGS=['20260602_143646','20260602_142543','20260602_150821','20260602_152153','20260602_153853','20260602_162513','20260602_163135']
    outdir=r'C:\Users\Martin\Active\FYPTurbine\data_analysis\cav_annotations'
    paths=cv.make_templates(TAGS, outdir)
    print('templates in', outdir)
    import json
    print('--- example skeleton (one step shown) ---')
    d=json.load(open(paths[0]))
    d['steps']={'20':d['steps']['20'],'...':'(40,60,80,100,120,140,160 same)'}
    print(json.dumps(d, indent=2))