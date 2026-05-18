import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from mech_components.bearing import Bearing
from mech_components.mechanicalseal import MechanicalSeal
from prop_components.engine import Engine
from prop_components.turbine import Turbine
from prop_components.barskepump import BarskePump
from prop_components.gasgenerator import GasGenerator

# ── Design inputs (edit here, changes propagate to sweep and readout) ──
RPM      = 20_000
P_W      = 2_600       # W — shaft power requirement
BETA     = 15          # deg — nozzle exit angle
DOA      = 0.1         # degree of admission (10%)
P_E      = 1.1e5       # Pa — turbine exit pressure

# Gas (cold gas N2 testing)
R_GAS     = 296.8      # J/kg/K
GAM       = 1.4
T01       = 300        # K
N_NOZZLES = 3

# Sweep range
D_RANGE   = (80, 120)      # mm
MD_RANGE  = (0.02, 0.05)   # kg/s
N_SWEEP   = 100            # points per axis (keep at 10 for quick cold runs)

# ── Constraint thresholds (shift when FEA fails) ──
H_MIN      = 4e-3          # m — minimum manufacturable blade height
THROAT_MIN = 0.5e-3        # m — minimum printable nozzle throat width
M3_MAX     = 1.8           # — nozzle exit Mach ceiling 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker
from prop_components.turbine import Turbine

diameters  = np.linspace(*D_RANGE,  N_SWEEP)
massflows  = np.linspace(*MD_RANGE, N_SWEEP)

efficiencies  = np.zeros((len(diameters), len(massflows)))
p01s          = np.zeros((len(diameters), len(massflows)))
t3s           = np.zeros((len(diameters), len(massflows)))
M3s           = np.zeros((len(diameters), len(massflows)))
heights       = np.zeros((len(diameters), len(massflows)))
throat_length = np.zeros((len(diameters), len(massflows)))

for i, diameter in enumerate(diameters):
    for j, mdot in enumerate(massflows):
        try:
            stage = Turbine(P=P_W, RPM=RPM, d_mean_mm=diameter, mdot=mdot,
                            beta_deg=BETA, doa=DOA, p_e=P_E)
            stage.from_inert_gas(R=R_GAS, gam=GAM, T01=T01, nozzles=N_NOZZLES)
            efficiencies[i, j]  = stage.eff
            p01s[i, j]          = stage.p01
            t3s[i, j]           = stage.T3
            M3s[i, j]           = stage.M3
            heights[i, j]       = stage.Height
            throat_length[i, j] = stage.nozzle_throat_length
            if stage.p01 > 10e5 or stage.T3 < 0:
                efficiencies[i, j] = p01s[i, j] = t3s[i, j] = M3s[i, j] = heights[i, j] = throat_length[i, j] = np.nan
        except Exception:
            efficiencies[i, j] = p01s[i, j] = t3s[i, j] = M3s[i, j] = heights[i, j] = throat_length[i, j] = np.nan

# ── Masks and feasibility region ──
eff_masked    = np.ma.array(efficiencies,  mask=np.isnan(efficiencies))
M3_masked     = np.ma.array(M3s,          mask=np.isnan(M3s))
h_masked      = np.ma.array(heights,      mask=np.isnan(heights))
throat_masked = np.ma.array(throat_length, mask=np.isnan(throat_length))
feasible = (heights >= H_MIN) & (throat_length >= THROAT_MIN) & (M3s <= M3_MAX)

fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# ── M3 plot + Mach ceiling ──
cs3 = axs[0].contourf(massflows, diameters, M3_masked, levels=50, cmap='viridis')
cb3 = fig.colorbar(cs3, ax=axs[0])
cb3.locator = ticker.MaxNLocator(nbins=3); cb3.update_ticks()
axs[0].set_xlabel('Mass Flow Rate (kg/s)')
axs[0].set_ylabel('Mean Diameter (mm)')
axs[0].set_title('Mach Number (M3)')
axs[0].plot(0.034, 100, 'ro', label='Design Point')
cs_m = axs[0].contour(massflows, diameters, M3_masked,
                      levels=[M3_MAX], colors='white', linewidths=2, linestyles='--')
axs[0].clabel(cs_m, fmt=f'M={M3_MAX}')
axs[0].contourf(massflows, diameters, feasible.astype(float),
                levels=[0, 0.5], colors=['black'], alpha=0.35)
axs[0].legend()

# ── Blade height + H_min and throat_min overlays ──
cs4 = axs[1].contourf(massflows, diameters, h_masked * 1000, levels=50, cmap='viridis')
cb4 = fig.colorbar(cs4, ax=axs[1])
cb4.locator = ticker.MaxNLocator(nbins=3); cb4.update_ticks()
axs[1].set_xlabel('Mass Flow Rate (kg/s)')
axs[1].set_ylabel('Mean Diameter (mm)')
axs[1].set_title('Required Blade Height (mm)')
axs[1].plot(0.034, 100, 'ro', label='Design Point')
cs_h = axs[1].contour(massflows, diameters, h_masked,
                      levels=[H_MIN], colors='red', linewidths=2)
axs[1].clabel(cs_h, fmt=f'H={H_MIN*1000:.0f}mm')
cs_t = axs[1].contour(massflows, diameters, throat_masked,
                      levels=[THROAT_MIN], colors='orange', linewidths=2)
axs[1].clabel(cs_t, fmt=f'throat={THROAT_MIN*1000:.1f}mm')
axs[1].contourf(massflows, diameters, feasible.astype(float),
                levels=[0, 0.5], colors=['black'], alpha=0.35)
proxy_h = Line2D([0], [0], color='red',    linewidth=2, label=f'H={H_MIN*1000:.0f}mm floor')
proxy_t = Line2D([0], [0], color='orange', linewidth=2, label=f'throat={THROAT_MIN*1000:.1f}mm floor')
axs[1].legend(handles=[axs[1].lines[0], proxy_h, proxy_t])

# ── Efficiency + infeasible shading ──
cs0 = axs[2].contourf(massflows, diameters, eff_masked, levels=50, cmap='viridis')
cb0 = fig.colorbar(cs0, ax=axs[2])
cb0.locator = ticker.MaxNLocator(nbins=3); cb0.update_ticks()
axs[2].set_xlabel('Mass Flow Rate (kg/s)')
axs[2].set_ylabel('Mean Diameter (mm)')
axs[2].set_title('Turbine Efficiency')
axs[2].plot(0.034, 100, 'ro', label='Design Point')
axs[2].contourf(massflows, diameters, feasible.astype(float),
                levels=[0, 0.5], colors=['black'], alpha=0.35)
axs[2].legend()

plt.tight_layout()
plt.show()

---CELL---

# ── Selected design point — change D_SEL / MD_SEL, re-run this cell ──
from prop_components.turbine import Turbine

D_SEL  = 100     # mm — from plot
MD_SEL = 0.034   # kg/s — from plot

stage = Turbine(P=P_W, RPM=RPM, d_mean_mm=D_SEL, mdot=MD_SEL,
                beta_deg=BETA, doa=DOA, p_e=P_E)
stage.from_inert_gas(R=R_GAS, gam=GAM, T01=T01, nozzles=N_NOZZLES)
stage.pretty_print()
stage.partload_rpm(17000)