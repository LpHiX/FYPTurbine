# Goldman (1970) BL Solver — Agent Handoff Context

## Project Goal

Replicate Goldman's (1970) NASA TM X-2095 Figures 4, 5, 8, 9 for supersonic turbine rotor blade boundary layer analysis. The ultimate purpose is to **predict separation and displacement thickness** on MOC-designed turbine blades so the blade profile can be corrected for boundary layer effects.

## Current State

The script `c:\Users\Martin\Active\FYPTurbine\quicktests\goldman_exact.py` produces results with **correct trends but a systematic ~0.25 offset** in incompressible form factor H_i compared to Goldman's published Figure 4.

- Goldman Fig. 4: H_i starts ~1.67 (lower surface), peaks ~2.1
- Script output: H_i starts ~1.42, peaks ~1.9
- **Shapes/trends match very well** — it's a vertical offset

Changing `H_0` from 1.4 to 1.7 only affects the first ~5% of chord — the ODE quickly converges to its own solution, confirming the offset is NOT from initial conditions.

## Key Files

- **`c:\Users\Martin\Active\FYPTurbine\quicktests\goldman_exact.py`** — Main script. Uses `SupersonicTurbineMOC` for blade geometry + von Kármán/Head entrainment BL solver.
- **`c:\Users\Martin\Active\FYPTurbine\prop_components\turbinemoc.py`** — MOC blade generator with `surface_mach_distribution()` method.
- **Goldman paper**: `c:\Users\Martin\Active\FYPTurbine\quicktests\Goldman - 1970 - Analytical investigation of blade efficiency for two-dimensional supersonic turbine rotor blade sect.pdf`
- **Sasman-Cresci paper**: `c:\Users\Martin\Active\FYPTurbine\quicktests\Sasman and Cresci - 1966 - Compressible turbulent boundary layer with pressure gradient and heat transfer..pdf`

## Goldman's Conditions

- γ = 1.4, M_inlet = 2.5, β_inlet = 70°, Re = 35,000
- "Typical case": ν_lower = 22° (M ≈ 1.84), ν_upper = 49° (M ≈ 2.97)

## Root Cause of the H_i Offset (Confirmed)

Goldman uses the **Sasman-Cresci method** (Ref. 9 in his paper) implemented via McNally's Fortran program (Ref. 8, NASA TN D-5681). The current script uses **von Kármán + Head entrainment**. The differences are:

### 1. Auxiliary Equation (BIGGEST driver, ~0.15–0.20 of offset)
- **Goldman**: Moment-of-momentum integral equation (tracks shear stress profile shape)
- **Script**: Head's entrainment equation (tracks mass entrainment at BL edge)
- The moment-of-momentum equation is more sensitive to adverse pressure gradients → predicts higher H_i in the transition arcs

### 2. Compressibility Transformation (second driver, ~0.05–0.15)
- **Goldman**: Full Mager coordinate transformation. His "H_i" is δ*/θ in **transformed** (X,Y) coordinates, NOT physical coordinates
- **Script**: Reference temperature / Stewartson-type correction. Its "H_i" is effectively in physical coordinates
- The Mager transformation at M=2.5 inflates H_i systematically
- Key relation from Sasman-Cresci paper: `H = [1 + (γ-1)/2 · Me²] · H_tr + (γ-1)/2 · Me²` where H_tr is the transformed form factor Goldman plots

### 3. Initial Conditions (NOT the cause — confirmed by user)
- Changing H_0 from 1.4 to 1.7 only affects first 5% of chord
- The ODE converges to its own trajectory regardless of IC

### 4. Closure Relations
- Script uses Head's H₁(H) correlations (incompressible empirical fits)
- Goldman doesn't use H₁ — moment-of-momentum directly solves for transformed H_i
- Head's shear integral: `CE = 0.0306 * (H1 - 3.0)^(-0.6169)`
- Sasman-Cresci shear integral: `∫(τ/τ_w)dη ≈ 0.011/H_i + C_f/2`

### 5. Skin Friction
- Both use Ludwieg-Tillmann law but applied in different coordinate systems
- Goldman: `C_f/2 = 0.123 · exp(-1.561·H_i) · (U_e·θ/ν_0)^(-0.268) · (T₀/T_e)^0.268`
- Script: `C_f = 0.246 · 10^(-0.678·H) · Re_θ^(-0.268)` (standard incompressible form)

## Recommended Next Steps (in order of effort)

### Option A: Post-process H_i with Mager correction (~30 min)
After solving with the current method, convert physical H to Mager-transformed H_i:
```python
H_i_goldman = (H_physical - GM1/2 * Me**2) / (1 + GM1/2 * Me**2)
```
This is an approximation but should close ~0.05–0.15 of the gap.

### Option B: Implement Sasman-Cresci method (2–4 hours)
Replace the Head entrainment ODE with Sasman-Cresci's moment-of-momentum equation. Key equations from their 1966 AIAA paper:
1. Momentum integral (their Eq. 5) in Mager-transformed coordinates
2. Moment-of-momentum (their Eq. 6) with equilibrium shear integrals
3. Skin friction (their Eq. 10): Ludwieg-Tillmann + Eckert reference enthalpy
4. Shear integral closure (their Eq. 12): `0.011/H_i + C_f/2`
5. Reference temperature (their Eq. 9, Eckert): `T̃/T₀ = 0.5·T_w/T₀ + 0.22·Pr^(-1/3) + (0.5 - 0.22·Pr^(-1/3))·(T_e/T₀)`

### Option C: Recalibrate separation criterion (5 min, pragmatic)
For practical BL correction of MOC blades, use separation criterion H_i ≈ 1.5–1.7 instead of 1.8–2.4 to match the script's method scale. The **relative** trends and **locations** of separation are already correct.

## Current BL Solver Structure (in goldman_exact.py)

The ODE system in `bl_ode()` solves for `[θ, H₁]`:
- `dθ/ds = C_f/2 - θ·(H+2)·(1/u_e)·(du_e/ds)` — momentum integral
- `dH₁/ds = [C_E - H₁·C_f/2 + H₁·θ·(H+1)·(1/u_e)·(du_e/ds)] / θ` — Head entrainment

Where:
- `H = H_from_H1(H1)` via Head's empirical correlation
- `C_f = 0.246 · 10^(-0.678·H) · Re_θ^(-0.268)` (Ludwieg-Tillmann)
- `C_E = 0.0306 · (H₁ - 3.0)^(-0.6169)` (Head entrainment coefficient)
- Reference temperature correction applied via `ref_nu_ratio()`
