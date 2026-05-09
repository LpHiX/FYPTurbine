# Turbine Sizer — Planned Improvements

From session 2026-05-09 with Claude. This is a handoff note for the next session to implement.

## Context

`turbinesizer.ipynb` + `turbine.py` in `C:\Users\Martin\active\FYPTurbine\`.

Current workflow: fix RPM, P, beta, doa, p_e. Sweep `diameters × massflows` (10×10 cold gas, 100×100 commented out). Plot contour maps of M3, efficiency, blade height, throat length. Pick design point by eye.

Problem: no hard constraint overlays → selection is trial-and-error. When FEA fails, need a quick way to shift the design point without re-running the whole sweep.

---

## Improvement 1 — Constraint Overlays on Contour Maps

Add these after the existing `contourf` calls. Define constraint thresholds at the top of the cell so they're easy to change:

```python
# ── Constraint thresholds (adjust these when FEA fails) ──
H_MIN    = 3e-3    # m — minimum manufacturable blade height
THROAT_MIN = 0.5e-3  # m — minimum printable nozzle throat width
M3_MAX   = 2.0     # — nozzle exit Mach ceiling

# ── Overlay on existing axes (ax is your current axes object) ──

# Blade height floor — anything below is unbuildable
cs_h = ax.contour(massflows, diameters, heights,
                  levels=[H_MIN], colors='red', linewidths=2)
ax.clabel(cs_h, fmt=f'H={H_MIN*1000:.0f}mm')

# Nozzle throat printability floor
cs_t = ax.contour(massflows, diameters, throat_length,
                  levels=[THROAT_MIN], colors='orange', linewidths=2)
ax.clabel(cs_t, fmt=f'throat={THROAT_MIN*1000:.1f}mm')

# Mach ceiling
cs_m = ax.contour(massflows, diameters, M3s,
                  levels=[M3_MAX], colors='white', linewidths=2, linestyles='--')
ax.clabel(cs_m, fmt=f'M={M3_MAX}')

# Shade infeasible region (outside all constraints)
feasible = (heights >= H_MIN) & (throat_length >= THROAT_MIN) & (M3s <= M3_MAX)
ax.contourf(massflows, diameters, feasible.astype(float),
            levels=[0, 0.5], colors=['black'], alpha=0.35)
```

Feasible design space = unshaded region. When FEA fails and you need more blade height, lower `H_MIN`. When nozzle is too narrow to print, raise `THROAT_MIN`. The intersection shifts immediately — no re-sweep needed.

---

## Improvement 2 — Separate Parameter Cell at Top of Notebook

Move all design inputs into a single cell so changes propagate everywhere:

```python
# ── Design inputs ──
RPM    = 20_000
P_W    = 500         # W
BETA   = 20          # deg — nozzle exit angle (confirm from current design)
DOA    = 1.0         # degree of admission (confirm from current design)
P_E    = 1.1e5       # Pa — turbine exit pressure

# Gas (cold gas N2 testing)
R_GAS  = 296.8       # J/kg/K
GAM    = 1.4
T01    = 300         # K
N_NOZZLES = 4

# Sweep range
D_RANGE  = (80, 120)   # mm
MD_RANGE = (0.02, 0.05) # kg/s
N_SWEEP  = 10          # points per axis (keep low for cold gas; CEA is slow)

# ── Constraint thresholds ──
H_MIN      = 3e-3
THROAT_MIN = 0.5e-3
M3_MAX     = 2.0
```

---

## Improvement 3 — Quick Design Point Readout

After selecting a point from the plot (click or manually set), add a cell that re-runs the single design point and prints full `pretty_print`:

```python
# ── Selected design point ──
D_SEL   = 100     # mm — from plot
MD_SEL  = 0.034   # kg/s — from plot

stage = Turbine(P=P_W, RPM=RPM, d_mean_mm=D_SEL, mdot=MD_SEL,
                beta_deg=BETA, doa=DOA, p_e=P_E)
stage.from_inert_gas(R=R_GAS, gam=GAM, T01=T01, nozzles=N_NOZZLES)
stage.pretty_print()
stage.partload_rpm(10_000)
```

Run this cell any time you change D_SEL / MD_SEL. No need to re-run the sweep.

---

## Stress Analysis Formulas (to add as a notebook section or separate script)

See `D:\TPU-Obsidian\08 AI\Claude\topics\turbine-design.md` for full derivation and context.

### Doubly-fixed blade (root + tip shroud constrained)

```python
import numpy as np

def blade_stress(rho_blade, b, t, H, omega, r_mean, E):
    """
    rho_blade : kg/m³
    b         : blade chord in radial direction (m)
    t         : blade thickness tangentially (m)  
    H         : blade height axially (m)
    omega     : rad/s
    r_mean    : mean radius (m)
    E         : Young's modulus (Pa)
    Returns   : sigma_max (Pa), delta_max (m), R_tip (N per blade)
    """
    q = rho_blade * b * t * omega**2 * r_mean   # N/m
    I = t * b**3 / 12                            # bending about radial axis
    M_max = q * H**2 / 12                        # at root and tip
    sigma_max = M_max * (b/2) / I                # = q*H²/(2*t*b²)
    delta_max = q * H**4 / (384 * E * I)
    R_tip = q * H / 2                            # radial force on shroud per blade
    return sigma_max, delta_max, R_tip
```

### Shroud ring hoop stress

```python
def shroud_stress(rho_shroud, omega, r_tip, t_shroud, N_blades, R_tip):
    """t_shroud : shroud wall thickness (m) — axial"""
    sigma_centrifugal = rho_shroud * omega**2 * r_tip**2
    sigma_blades = N_blades * R_tip / (2 * np.pi * r_tip * t_shroud)
    return sigma_centrifugal + sigma_blades
```

### Disc (tapered) — Stodola stepwise

For a linearly tapered disc (h varies with r), divide into N annuli of constant thickness and apply Lamé per annulus, matching σ_r and u_r at interfaces. Boundary conditions: σ_r = 0 at bore, σ_r = -p_root at disc OD (blade root pressure, usually small).

This is numerical — implement as a separate function if FEA isn't available. For first-pass, use the uniform disc formula:

```python
def disc_hoop_max(rho_disc, omega, r_inner, r_outer, nu):
    """Max hoop stress in a uniform rotating annular disc (at bore)."""
    return (rho_disc * omega**2 / 4) * (
        (3 + nu) * (r_inner**2 + r_outer**2)
        - (1 + 3*nu) * r_inner**2
    )
```

---

## Notes

- CEA path (`from_gasgen`) is slow — keep sweep on cold gas, run CEA only on selected design point for final validation.
- `partload_rpm()` already checks Kantrowitz starting at reduced speed — run this after selecting design point.
- Blade material unknown at time of writing — get E and σ_UTS before running stress check.
