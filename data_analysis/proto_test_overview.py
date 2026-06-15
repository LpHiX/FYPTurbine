"""
proto_test_overview.py  --  PROTOTYPE procedure-overview plot.

One giant grid: rows = measurement channels, columns = the five full pump tests
(30/35/40/45/50% ESC throttle). Lets the experimental procedure (the slow valve
ramp at fixed drive speed) be discussed straight off the time series.

Reuses epump_io for loading so channel names / units stay in one place.
Confirmed units: pin/pout = bar, q = l/s, rpm = RPM.
NOTE: torque (adc_torque_mv) is loaded RAW with no scaling in epump_io. If the
Nm numbers look wrong, apply the DSP6001 scaling (1 V per 0.25 Nm) here. Flagged.

Run:  python proto_test_overview.py
Out:  proto_test_overview.png
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import epump_io as ep

LOGDIR = r"D:\Projects\propbackend_logs\2026-06-02\hotfirelog"

# tag -> column label (the five canonical full tests, findings doc)
TESTS = {
    "20260602_150821": "30% ESC",
    "20260602_152153": "35% ESC",
    "20260602_153853": "40% ESC",
    "20260602_162513": "45% ESC",
    "20260602_163135": "50% ESC",
}

# each row: (y-label, [(series_key, legend_label, colour), ...])
# series_key indexes into the loaded data dict from epump_io.load().
ROWS = [
    ("Pressure [bar]", [("pin", "inlet", "C0"), ("pout", "outlet", "C3")]),
    ("Flow [l/s]",     [("q",   None, "C2")]),
    ("Torque [Nm]*",   [("tq",  None, "C4")]),   # * raw channel, verify scaling
    ("Speed [RPM]",    [("rpm", None, "C1")]),
    ("Inlet valve [deg]",  [("in_dem",  None, "C0")]),
    ("Outlet valve [deg]", [("out_dem", None, "C3")]),
]

SMOOTH_WIN = 15      # boxcar window for the bold overlay; raw drawn faint under it
OVERLAY = True       # True -> single column, all 5 tests overlaid per channel
XLIM = None          # e.g. (0, 25) to crop to the ramp window; None = full run


def _plot_series(ax, t, y, colour, label):
    ax.plot(t, y, color=colour, lw=0.4, alpha=0.35)
    ax.plot(t, ep.smooth(y, SMOOTH_WIN), color=colour, lw=1.0, label=label)


def main():
    runs = ep.load_runs(LOGDIR, tags=list(TESTS))
    missing = [tg for tg in TESTS if tg not in runs]
    if missing:
        raise SystemExit(f"missing runs in {LOGDIR}: {missing}")

    nrows = len(ROWS)
    tags = list(TESTS)

    if OVERLAY:
        fig, axes = plt.subplots(nrows, 1, figsize=(7, 2.0 * nrows),
                                 sharex=True, squeeze=False)
        for r, (ylabel, series) in enumerate(ROWS):
            ax = axes[r][0]
            # one channel per row: last series (pressure -> outlet pout)
            key = series[-1][0]
            for i, tg in enumerate(tags):
                d = runs[tg]
                ax.plot(d["t"], ep.smooth(d[key], SMOOTH_WIN), lw=1.0,
                        color=f"C{i}", label=TESTS[tg])
            ax.set_ylabel(ylabel, fontsize=9)
            if XLIM:
                ax.set_xlim(*XLIM)
        axes[0][0].legend(fontsize=8, ncol=5, loc="upper center",
                          bbox_to_anchor=(0.5, 1.45))
        axes[-1][0].set_xlabel("Time [s]")
    else:
        ncols = len(tags)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 1.8 * nrows),
                                 sharex="col", sharey="row", squeeze=False)
        for c, tg in enumerate(tags):
            d = runs[tg]
            axes[0][c].set_title(TESTS[tg], fontsize=10)
            for r, (ylabel, series) in enumerate(ROWS):
                ax = axes[r][c]
                for key, lab, colour in series:
                    _plot_series(ax, d["t"], d[key], colour, lab)
                if c == 0:
                    ax.set_ylabel(ylabel, fontsize=9)
                if r == 0 and any(lab for _, lab, _ in series):
                    ax.legend(fontsize=6, loc="upper left")
                if XLIM:
                    ax.set_xlim(*XLIM)
            axes[-1][c].set_xlabel("Time [s]", fontsize=9)

    fig.suptitle("Pump test campaign: channel time series per ESC throttle",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = "proto_test_overview.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}  ({nrows} rows x {1 if OVERLAY else len(tags)} cols)")


if __name__ == "__main__":
    main()
