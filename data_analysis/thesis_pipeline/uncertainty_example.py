"""Worked template: end-to-end measurement uncertainty.

Install once:   pip install uncertainties

The idea (this is the whole answer to "how do I plot uncertainty everywhere"):
  1. Assign a 1-sigma to each RAW channel ONCE — systematic (datasheet/calibration,
     fixed) combined in quadrature with random (the per-point sample scatter).
  2. Wrap the arrays with `uncertainties.unumpy`. Every derived quantity you
     compute from them (head, psi, eta, NPSHr) then carries its uncertainty
     AUTOMATICALLY via linear error propagation — no hand-derived partials.
  3. To draw error bars:  unp.nominal_values(Y) and unp.std_devs(Y).
  4. To report a scalar: feed the ufloat straight into the registry with
     reg.add_ufloat(key, Y, ...).

This file is illustrative (synthetic arrays). Wire the real channels from
epump_io into steps 1-2 and delete the fake data.
"""
from __future__ import annotations

import numpy as np
from uncertainties import unumpy as unp

RHO, G = 998.0, 9.81          # water density, gravity
D2 = 0.0585                   # impeller tip diameter [m]

# --- 1. raw channel 1-sigmas (assign ONCE) ----------------------------------
# Systematic: e.g. a 1.2 MPa transducer at 0.25% full-scale -> 0.003 bar.
P_SYS_PA = 0.0025 * 1.2e6     # Pa, fixed bias from the datasheet FS spec
N_SYS_RPM = 1.0              # tacho systematic [rpm]


def channel_unc(values_pa, sample_std_pa):
    """Combine fixed systematic bias with per-point random scatter (quadrature)."""
    sigma = np.hypot(sample_std_pa, P_SYS_PA)
    return unp.uarray(values_pa, sigma)


# --- 2-3. propagate to developed head and head coefficient ------------------
def head_and_psi(pin_pa, pin_std, pout_pa, pout_std, n_rpm, n_std):
    pin = channel_unc(pin_pa, pin_std)
    pout = channel_unc(pout_pa, pout_std)
    H = (pout - pin) / (RHO * G)                       # carries uncertainty
    n = unp.uarray(n_rpm, np.hypot(n_std, N_SYS_RPM))
    u2 = np.pi * D2 * n / 60.0
    psi = 2 * G * H / u2 ** 2                          # carries uncertainty
    return H, psi


def demo():
    import matplotlib.pyplot as plt
    from data_analysis.figstyle import use_style, plot_data
    use_style()

    # ---- synthetic stand-in for real binned data; replace with epump_io output
    q = np.linspace(0.05, 0.25, 8)                     # l/s
    pin_pa = np.full_like(q, 0.2e5); pin_std = np.full_like(q, 0.01e5)
    pout_pa = (2.0 - 6 * q) * 1e5;   pout_std = np.full_like(q, 0.02e5)
    n_rpm = np.full_like(q, 7000.0); n_std = np.full_like(q, 50.0)

    H, psi = head_and_psi(pin_pa, pin_std, pout_pa, pout_std, n_rpm, n_std)
    phi = q  # placeholder; use the real flow coefficient

    fig, ax = plt.subplots()
    plot_data(ax, phi, unp.nominal_values(psi), yerr=unp.std_devs(psi), label="experiment")
    ax.set_xlabel(r"flow coefficient $\phi$"); ax.set_ylabel(r"head coefficient $\psi$")
    ax.legend(); fig.savefig("uncertainty_demo.pdf")
    print("peak psi =", psi[np.argmax(unp.nominal_values(psi))])  # a ufloat: 1.10+/-0.05


if __name__ == "__main__":
    demo()
