"""Bounded re-fit of the Lock constants with the TRUE throat D3=3.0mm fixed.

Fit (K1, zeta) by least squares against the binned H-Q data, all 5 speeds.
- D3 = 3.0 mm: geometry (CAD-confirmed 2026-06-11), NOT a fit parameter.
- K1 in [0, 1]: prerotation factor (Lock's study 0.15-0.20; excess absorbs
  LE-rounding + effective-radius h0 optimism, both in Lock's paper).
- zeta in [0.194, 0.6]: nozzle loss coefficient; lower bound = clean-nozzle
  literature value, upper bound = sharp-edge/offset-orifice entry separation
  (Lock p.733 shear-water description).
Residuals only where the binned head is on the plateau/droop (H > 30% of the
run's max) — the collapse tail is the cutoff's job, not the fit's.
"""
import sys
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, r"C:\Users\Martin\active\FYPTurbine\data_analysis")
sys.path.insert(0, r"C:\Users\Martin\active\FYPTurbine")
import thesis_figures as tf
from uncertainties import unumpy as unp

D3_TRUE = 0.0030
runs = tf.exp_runs()

data = []
for lbl, e in runs.items():
    H = unp.nominal_values(tf.u_head(e["H"], e["Hsem"]))
    q = np.asarray(e["q"], float)          # l/s
    m = H > 0.3 * np.nanmax(H)
    data.append((lbl, e["N"], q[m] / 1000.0, H[m]))
    print(f"{lbl}: {m.sum()}/{len(H)} points in fit window")

def resid(p):
    K1, zeta = p
    r = []
    for lbl, N, qm, Hm in data:
        res = tf.pump().analyse_lock(qm, RPM=N, D_3=D3_TRUE, D_inlet=tf.DINLET,
                                     K_factor=K1, eta_losses=zeta,
                                     p_inlet=tf.P_INLET)
        r.append(np.asarray(res["H_static"], float) - Hm)
    return np.concatenate(r)

fit = least_squares(resid, x0=[0.4, 0.3], bounds=([0.0, 0.194], [1.0, 0.6]))
K1, zeta = fit.x
rms = np.sqrt(np.mean(fit.fun ** 2))
print(f"\nFIT: K1 = {K1:.3f}   zeta = {zeta:.3f}   rms = {rms:.3f} m   "
      f"(zeta at bound: {'YES' if abs(zeta-0.6)<1e-3 or abs(zeta-0.194)<1e-3 else 'no'})")

# old constants for comparison
r_old = []
for lbl, N, qm, Hm in data:
    res = tf.pump().analyse_lock(qm, RPM=N, D_3=tf.D3, D_inlet=tf.DINLET,
                                 K_factor=tf.K_FIT, eta_losses=tf.ETAL_FIT,
                                 p_inlet=tf.P_INLET)
    r_old.append(np.asarray(res["H_static"], float) - Hm)
print(f"old (K=0.55, z=1.1, D3=3.8) rms = {np.sqrt(np.mean(np.concatenate(r_old)**2)):.3f} m")

# predicted cutoff flows vs video clears
print("\ncutoffs with new fit (D3=3.0):")
clears = {"40%": 0.211, "45%": 0.227, "50%": 0.255}
for lbl, e in runs.items():
    qg = np.linspace(1e-6, 0.7e-3, 2000)
    res = tf.pump().analyse_lock(qg, RPM=e["N"], D_3=D3_TRUE, D_inlet=tf.DINLET,
                                 K_factor=K1, eta_losses=zeta, p_inlet=tf.P_INLET)
    broke = np.asarray(res["H_3"], float) <= (3171.0 - tf.P_INLET) / (tf.RHO * tf.G)
    qcut = qg[np.argmax(broke)] * 1000 if broke.any() else np.nan
    vid = f"  video clear {clears[lbl]}" if lbl in clears else ""
    print(f"  {lbl} ({e['N']:.0f} rpm): model cutoff {qcut:.3f} l/s{vid}")
