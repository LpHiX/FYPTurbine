"""
Von Kármán + Head entrainment boundary layer solver for supersonic turbine blades.
Validates against Goldman (1970) NASA TM X-2059 — Figures 4, 5, 8, 9.

Goldman conditions: M_in=2.5, β_in=70°, Re=35000, γ=1.4
Vortex-flow passage with inlet/outlet transition arcs.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# ── Gas constants ──
GAMMA = 1.4
GM1 = GAMMA - 1
GP1 = GAMMA + 1

# ── Isentropic relations ──
def T0_over_T(M):
    return 1 + GM1 / 2 * M**2

def rho0_over_rho(M):
    return T0_over_T(M) ** (1 / GM1)

def prandtl_meyer(M):
    """Prandtl-Meyer angle in degrees."""
    t = M**2 - 1
    if t <= 0:
        return 0.0
    return (np.sqrt(GP1 / GM1) * np.arctan(np.sqrt(GM1 / GP1 * t))
            - np.arctan(np.sqrt(t))) * 180 / np.pi

def mach_from_prandtl_meyer(nu_deg):
    """Invert Prandtl-Meyer function."""
    def res(M):
        return prandtl_meyer(M) - nu_deg
    return brentq(res, 1.001, 8.0)


# ── Sutherland viscosity (ratio to reference temperature) ──
def mu_ratio_sutherland(T, T_ref):
    S = 110.4  # K, air
    return (T / T_ref)**1.5 * (T_ref + S) / (T + S)


# ── Head's correlations ──
def H1_from_H(H):
    if H > 1.6:
        return 3.3 + 0.8234 * (H - 1.1)**(-1.287)
    else:
        return 3.3 + 1.5501 * (H - 0.6778)**(-3.064)

def H_from_H1(H1):
    if H1 > 100:
        return 1.2
    def res(H):
        return H1_from_H(H) - H1
    try:
        return brentq(res, 1.2, 4.0)
    except ValueError:
        return 1.4 if H1 > H1_from_H(1.4) else 3.5

def Cf_ludwieg_tillmann(H, Re_theta):
    if Re_theta < 10:
        Re_theta = 10
    return 0.246 * 10**(-0.678 * H) * Re_theta**(-0.268)

def CE_head(H1):
    if H1 <= 3.03:
        H1 = 3.03
    return 0.0306 * (H1 - 3.0)**(-0.6169)


# ── Surface Mach number distribution (vortex passage model) ──
def make_mach_distribution(M_in, M_surface, s_norm):
    """
    Model the Mach distribution along a vortex-flow blade surface.
    Goldman's MoC passage: inlet transition arcs, circular arcs (vortex),
    outlet transition arcs. Transition arc length scales with |delta-nu|
    to mimic the MoC design — bigger Mach change = longer arc.

    For impulse design, outlet transition is shorter than inlet
    (Goldman: outlet circular arcs less than inlet, to give G_out < G_in
    for displacement thickness correction).
    """
    # Prandtl-Meyer angle change determines transition arc extent
    nu_in = prandtl_meyer(M_in)
    nu_s = prandtl_meyer(M_surface)
    delta_nu = abs(nu_s - nu_in)

    # Scale transition length: base 15% chord, + 0.5% per degree of turning
    inlet_len = min(0.15 + 0.005 * delta_nu, 0.30)
    # Outlet transition is shorter (Goldman's asymmetric design)
    outlet_len = min(0.10 + 0.004 * delta_nu, 0.25)

    s_in_start = 0.05
    s_in_end = s_in_start + inlet_len
    s_out_end = 0.95
    s_out_start = s_out_end - outlet_len

    # Use raised-cosine profile (smoother than tanh, realistic gradient)
    def cosine_blend(s, s0, s1):
        t = np.clip((s - s0) / (s1 - s0), 0, 1)
        return 0.5 * (1 - np.cos(np.pi * t))

    blend_in = cosine_blend(s_norm, s_in_start, s_in_end)
    blend_out = cosine_blend(s_norm, s_out_start, s_out_end)

    # Build piecewise: inlet → vortex → outlet
    M = M_in + (M_surface - M_in) * blend_in + (M_in - M_surface) * blend_out
    # Clamp to avoid overshoots
    if M_surface < M_in:
        M = np.clip(M, M_surface, M_in)
    else:
        M = np.clip(M, M_in, M_surface)

    cs = CubicSpline(s_norm, M)
    return cs


# ── Build edge conditions from Mach distribution ──
def build_edge_conditions(Me_spline, s_arr, Re_chord, M_in, T0=500.0):
    """
    Compute ue(s), Re_unit(s) = ρe*ue/μe along the surface.
    Normalise so that Re based on chord = Re_chord at inlet conditions.
    """
    Me = Me_spline(s_arr)
    Te = T0 / T0_over_T(Me)
    T_in = T0 / T0_over_T(M_in)

    # velocity ratio ue / u_in
    ae_over_ain = np.sqrt(Te / T_in)
    ue_ratio = Me / M_in * ae_over_ain  # ue/u_in

    # density ratio ρe/ρ_in
    rho_ratio = rho0_over_rho(M_in) / rho0_over_rho(Me)

    # viscosity ratio μe/μ_in
    mu_rat = np.array([mu_ratio_sutherland(t, T_in) for t in Te])

    # Re_unit / Re_unit_in
    Re_unit_in = Re_chord  # since chord = 1 (normalised)
    Re_unit = Re_unit_in * rho_ratio * ue_ratio / mu_rat

    return Me, ue_ratio, Re_unit


# ── Reference temperature method ──
def reference_temp_ratio(Me, r_rec=0.89):
    """T*/Te for adiabatic wall using Eckert's reference temperature."""
    alpha = r_rec * GM1 / 2 * Me**2
    # T* = Te * [1 + 0.72 * r * (γ-1)/2 * Me²]
    return 1 + 0.72 * alpha


def ref_nu_ratio(Me, T0, r_rec=0.89):
    """
    Ratio ν*/νe for reference temperature method.
    ν* = μ*/ρ*, with ρ*/ρe = Te/T*, μ*/μe from Sutherland.
    Returns ν*/νe.
    """
    Te = T0 / T0_over_T(Me)
    Tstar_Te = reference_temp_ratio(Me, r_rec)
    T_star = Te * Tstar_Te

    # μ*/μe via Sutherland
    mu_star_over_mu_e = mu_ratio_sutherland(T_star, Te)
    # ρ*/ρe = Te/T* (ideal gas at same pressure)
    rho_star_over_rho_e = 1.0 / Tstar_Te

    return mu_star_over_mu_e / rho_star_over_rho_e


# ── BL ODE system (incompressible equations + reference temperature) ──
def bl_ode(s, y, Me_fn, dMe_ds_fn, Re_unit_fn, nu_ratio_fn):
    """
    Incompressible von Kármán + Head equations.
    Compressibility enters only through Re_θ* (reference temperature method).
    H from this system IS Goldman's incompressible form factor Hi.
    """
    theta, H1 = y

    if theta < 1e-12:
        theta = 1e-12
    if H1 < 3.05:
        H1 = 3.05

    Me = float(Me_fn(s))
    dMe_ds = float(dMe_ds_fn(s))
    Re_unit_e = float(Re_unit_fn(s))
    nu_rat = float(nu_ratio_fn(s))

    T_ratio = T0_over_T(Me)
    inv_ue_due_ds = dMe_ds / (Me * T_ratio)

    H = H_from_H1(H1)

    # Re_θ using reference kinematic viscosity: Re_θ* = ue·θ/ν* = Re_θ_e / nu_ratio
    Re_theta = Re_unit_e * theta / nu_rat

    Cf = Cf_ludwieg_tillmann(H, Re_theta)
    CE = CE_head(H1)

    # Incompressible von Kármán (NO Me² term — that's the compressible form)
    dtheta = Cf / 2 - theta * (H + 2) * inv_ue_due_ds

    # Head's entrainment
    dH1 = (CE - H1 * Cf / 2 + H1 * theta * (H + 1) * inv_ue_due_ds) / theta

    return [dtheta, dH1]


def solve_bl(Me_spline, Re_chord, M_in, s_arr, T0=500.0):
    """Solve BL for one surface. Returns s, theta, H (=Hi), Me arrays."""
    Me_arr, ue_ratio, Re_unit_arr = build_edge_conditions(
        Me_spline, s_arr, Re_chord, M_in, T0
    )

    # Reference viscosity ratio ν*/νe along surface
    nu_ratio_arr = np.array([ref_nu_ratio(m, T0) for m in Me_arr])

    # Splines for interpolation
    dMe_spline = Me_spline.derivative()
    Re_unit_spline = CubicSpline(s_arr, Re_unit_arr)
    nu_ratio_spline = CubicSpline(s_arr, nu_ratio_arr)

    # Initial conditions: turbulent BL just downstream of LE
    s0 = s_arr[2]
    Re_unit_0 = float(Re_unit_spline(s0))
    nu_rat_0 = float(nu_ratio_spline(s0))
    Re_s0 = Re_unit_0 * s0 / nu_rat_0
    theta_0 = 0.036 * s0 * max(Re_s0, 10)**(-0.2)
    H_0 = 1.4
    H1_0 = H1_from_H(H_0)

    sol = solve_ivp(
        bl_ode,
        [s0, s_arr[-1]],
        [theta_0, H1_0],
        args=(Me_spline, dMe_spline, Re_unit_spline, nu_ratio_spline),
        t_eval=s_arr[2:],
        method='Radau',
        rtol=1e-6, atol=1e-10,
        max_step=0.005,
    )

    if not sol.success:
        print(f"  Warning: solver stopped at s/c={sol.t[-1]:.3f} -- {sol.message}")

    s_sol = sol.t
    theta_sol = sol.y[0]
    H1_sol = sol.y[1]
    H_sol = np.array([H_from_H1(h1) for h1 in H1_sol])
    Me_sol = Me_spline(s_sol)

    # H from the incompressible equations IS Goldman's Hi directly
    Hi_sol = H_sol

    return s_sol, theta_sol, H_sol, Hi_sol, Me_sol


# ── Main ──
def main():
    # Goldman's reference conditions
    M_in = 2.5
    Re_chord = 35000

    # Typical case from Goldman Figs 4, 5:
    # ν_lower = 34° → M_lower ≈ 2.14
    # ν_upper = 49° → M_upper ≈ 2.93
    M_lower = mach_from_prandtl_meyer(34.0)
    M_upper = mach_from_prandtl_meyer(49.0)
    print(f"Lower surface: nu=34 deg -> M={M_lower:.3f}")
    print(f"Upper surface: nu=49 deg -> M={M_upper:.3f}")

    # Arc-length grid (normalised by chord)
    s = np.linspace(0, 1, 501)

    # Build Mach distributions
    Me_lower = make_mach_distribution(M_in, M_lower, s)
    Me_upper = make_mach_distribution(M_in, M_upper, s)

    # Solve BL on each surface
    print("\nSolving lower surface BL...")
    s_lo, theta_lo, H_lo, Hi_lo, Me_lo = solve_bl(Me_lower, Re_chord, M_in, s)

    print("Solving upper surface BL...")
    s_up, theta_up, H_up, Hi_up, Me_up = solve_bl(Me_upper, Re_chord, M_in, s)

    print(f"\nLower surface: Hi_max = {np.max(Hi_lo):.2f} at s/c = {s_lo[np.argmax(Hi_lo)]:.3f}")
    print(f"Upper surface: Hi_max = {np.max(Hi_up):.2f} at s/c = {s_up[np.argmax(Hi_up)]:.3f}")

    # ── Plot 1: Surface Mach distributions (cf. Goldman Fig. 5) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Von Kármán + Head BL Solver — Goldman (1970) Validation\n'
        f'$M_{{in}}=2.5$, $\\beta_{{in}}=70°$, $Re={Re_chord}$, '
        f'$\\nu_l=34°\\ (M_l={M_lower:.2f})$, $\\nu_u=49°\\ (M_u={M_upper:.2f})$',
        fontsize=13
    )

    ax = axes[0, 0]
    ax.plot(s_lo, Me_lo, 'b-', lw=2, label='Lower surface')
    ax.plot(s_up, Me_up, 'r-', lw=2, label='Upper surface')
    ax.set_xlabel('Fraction of chord, $s/c$')
    ax.set_ylabel('Surface Mach number, $M_e$')
    ax.set_title('Surface Mach Distribution (cf. Goldman Fig. 5)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # ── Plot 2: Incompressible form factor Hi (cf. Goldman Fig. 4) ──
    ax = axes[0, 1]
    ax.plot(s_lo, Hi_lo, 'b-', lw=2, label='Lower surface')
    ax.plot(s_up, Hi_up, 'r-', lw=2, label='Upper surface')
    ax.axhline(1.8, color='gray', ls='--', alpha=0.5, label='Separation range')
    ax.axhline(2.4, color='gray', ls='--', alpha=0.5)
    ax.axhspan(1.8, 2.4, alpha=0.08, color='red')
    ax.set_xlabel('Fraction of chord, $s/c$')
    ax.set_ylabel('Incompressible form factor, $H_i$')
    ax.set_title('Form Factor Distribution (cf. Goldman Fig. 4)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(1.0, 3.0)

    # ── Plot 3: Momentum thickness ──
    ax = axes[1, 0]
    ax.plot(s_lo, theta_lo / s_lo[-1] * 1000, 'b-', lw=2, label='Lower surface')
    ax.plot(s_up, theta_up / s_up[-1] * 1000, 'r-', lw=2, label='Upper surface')
    ax.set_xlabel('Fraction of chord, $s/c$')
    ax.set_ylabel('$\\theta / c \\times 10^3$')
    ax.set_title('Momentum Thickness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # ── Plot 4: Displacement thickness ──
    ax = axes[1, 1]
    delta_star_lo = Hi_lo * theta_lo
    delta_star_up = Hi_up * theta_up
    ax.plot(s_lo, delta_star_lo / s_lo[-1] * 1000, 'b-', lw=2, label='Lower surface')
    ax.plot(s_up, delta_star_up / s_up[-1] * 1000, 'r-', lw=2, label='Upper surface')
    ax.set_xlabel('Fraction of chord, $s/c$')
    ax.set_ylabel('$\\delta^* / c \\times 10^3$')
    ax.set_title('Displacement Thickness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig('C:/Users/Martin/Active/FYPTurbine/quicktests/goldman_bl_validation.png', dpi=150)
    print("\nSaved: goldman_bl_validation.png")
    plt.close()

    # ── Plot 5: Hi_max sweep (cf. Goldman Figs. 8, 9) ──
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('$H_{i,max}$ vs Surface Prandtl-Meyer Angle (cf. Goldman Figs. 8, 9)', fontsize=13)

    # Sweep ν_lower at fixed ν_upper = 49°
    nu_lower_range = np.arange(18, 35, 2.0)
    Hi_max_lower = []
    for nu_l in nu_lower_range:
        Ml = mach_from_prandtl_meyer(nu_l)
        Me_sp = make_mach_distribution(M_in, Ml, s)
        try:
            s_sol, _, _, Hi, _ = solve_bl(Me_sp, Re_chord, M_in, s)
            Hi_max_lower.append(np.max(Hi))
        except Exception:
            Hi_max_lower.append(np.nan)
    ax1.plot(nu_lower_range, Hi_max_lower, 'bo-', lw=2, markersize=6)
    ax1.axhline(1.8, color='gray', ls='--', alpha=0.5)
    ax1.axhline(2.4, color='gray', ls='--', alpha=0.5)
    ax1.axhspan(1.8, 2.4, alpha=0.08, color='red')
    ax1.set_xlabel('Lower-surface Prandtl-Meyer angle, $\\nu_l$ (deg)')
    ax1.set_ylabel('$H_{i,max}$ on lower surface')
    ax1.set_title(f'$\\nu_u = 49°$ fixed (cf. Goldman Fig. 8)')
    ax1.grid(True, alpha=0.3)

    # Sweep ν_upper at fixed ν_lower = 34°
    nu_upper_range = np.arange(42, 55, 1.5)
    Hi_max_upper = []
    for nu_u in nu_upper_range:
        Mu = mach_from_prandtl_meyer(nu_u)
        Me_sp = make_mach_distribution(M_in, Mu, s)
        try:
            s_sol, _, _, Hi, _ = solve_bl(Me_sp, Re_chord, M_in, s)
            Hi_max_upper.append(np.max(Hi))
        except Exception:
            Hi_max_upper.append(np.nan)
    ax2.plot(nu_upper_range, Hi_max_upper, 'ro-', lw=2, markersize=6)
    ax2.axhline(1.8, color='gray', ls='--', alpha=0.5)
    ax2.axhline(2.4, color='gray', ls='--', alpha=0.5)
    ax2.axhspan(1.8, 2.4, alpha=0.08, color='red')
    ax2.set_xlabel('Upper-surface Prandtl-Meyer angle, $\\nu_u$ (deg)')
    ax2.set_ylabel('$H_{i,max}$ on upper surface')
    ax2.set_title(f'$\\nu_l = 34°$ fixed (cf. Goldman Fig. 9)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('C:/Users/Martin/Active/FYPTurbine/quicktests/goldman_Hi_sweep.png', dpi=150)
    print("Saved: goldman_Hi_sweep.png")
    plt.close()


if __name__ == '__main__':
    main()
