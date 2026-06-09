"""Plot-convention helpers enforcing the thesis house style.

Convention (confirmed with Knoll — consistency is what matters):
    solid line   = post-experiment tuned curve / spline fit
    dashed line  = original (untuned) theory
    markers only = experimental data (with error bars)

Usage:
    import matplotlib.pyplot as plt
    from figstyle import use_style, plot_data, plot_theory, plot_tuned
    use_style()
    fig, ax = plt.subplots()
    plot_theory(ax, phi, psi_theory, label="Lock (default)")
    plot_tuned(ax, phi, psi_tuned,  label="Lock (tuned)")
    plot_data(ax, phi, psi, yerr=psi_err, label="experiment")
    ax.legend()
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

STYLE = Path(__file__).resolve().parent.parent / "thesis.mplstyle"


def use_style():
    plt.style.use(str(STYLE))


def plot_data(ax, x, y, yerr=None, xerr=None, label=None, **kw):
    """Experimental data: markers only, with error bars."""
    return ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt="o", linestyle="none",
                       label=label, **kw)


def plot_theory(ax, x, y, label=None, **kw):
    """Original / untuned theory: dashed line."""
    return ax.plot(x, y, linestyle="--", label=label, **kw)


def plot_tuned(ax, x, y, label=None, **kw):
    """Post-experiment tuned curve or spline trend: solid line."""
    return ax.plot(x, y, linestyle="-", label=label, **kw)


def spline_trend(ax, x, y, label=None, n=300, **kw):
    """Qualitative spline trend through binned points (solid), for the
    'never connect medians with straight segments' rule. Caption it as a
    trend, not a fit. Falls back to a plain line if scipy is unavailable."""
    import numpy as np
    x = np.asarray(x, float); y = np.asarray(y, float)
    order = np.argsort(x); x, y = x[order], y[order]
    try:
        from scipy.interpolate import make_interp_spline
        xs = np.linspace(x.min(), x.max(), n)
        ys = make_interp_spline(x, y, k=min(3, len(x) - 1))(xs)
    except Exception:
        xs, ys = x, y
    return ax.plot(xs, ys, linestyle="-", label=label, **kw)
