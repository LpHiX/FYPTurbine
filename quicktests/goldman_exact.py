"""
Goldman (1970) NASA TM X-2059 — exact replication of Figs 4, 5, 8, 9.

Uses the Method of Characteristics (from turbinemoc.py) to compute the
actual blade surface geometry and Mach distribution, then feeds it into
the von Karman + Head entrainment BL solver.

Goldman conditions: gamma=1.4, M_in=2.5, beta_in=70 deg, Re=35000.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from prop_components.turbinemoc import SupersonicTurbineMOC

# ══════════════════════════════════════════════════════════════════════
# Gas dynamics
# ══════════════════════════════════════════════════════════════════════
GAMMA = 1.4
GM1 = GAMMA - 1
GP1 = GAMMA + 1
PRANDTL = 0.72


def nu_pm(M):
    """Prandtl-Meyer angle (radians). Standalone for M_from_nu inversion."""
    Ms = np.sqrt(GP1 / 2 * M**2 / (1 + GM1 / 2 * M**2))
    return (np.pi / 4 * (np.sqrt(GP1 / GM1) - 1)
            + 0.5 * (np.sqrt(GP1 / GM1) * np.arcsin(np.clip(GM1 * Ms**2 - GAMMA, -1, 1))
                      + np.arcsin(np.clip(GP1 / Ms**2 - GAMMA, -1, 1))))


def M_from_nu(nu_target):
    """Invert Prandtl-Meyer function (radians in, Mach out)."""
    return brentq(lambda M: nu_pm(M) - nu_target, 1.001, 10.0)


def T0_over_T(M):
    return 1 + GM1 / 2 * M**2


def rho0_over_rho(M):
    return T0_over_T(M) ** (1 / GM1)


def mu_ratio_sutherland(T, T_ref):
    S = 110.4
    return (T / T_ref)**1.5 * (T_ref + S) / (T + S)


# ══════════════════════════════════════════════════════════════════════
# Blade surface from turbinemoc
# ══════════════════════════════════════════════════════════════════════

def build_blade_surface(M_inlet, M_surface, beta_inlet_deg, side='lower',
                        dv=0.001, M_other=None):
    """
    Build complete blade surface from inlet to outlet using SupersonicTurbineMOC.

    Returns:
        s_norm : arc length / chord  (0 to 1)
        Me     : edge Mach number at each point
        chord  : total arc length
    """
    # For the MOC solver, mach_lower is the lower-surface Mach and
    # mach_upper is the upper-surface Mach. When we only need one surface,
    # we still need to provide both — use M_other for the opposite surface.
    if side == 'lower':
        mach_lower = M_surface
        mach_upper = M_other if M_other is not None else M_inlet
    else:
        mach_upper = M_surface
        mach_lower = M_other if M_other is not None else M_inlet

    moc = SupersonicTurbineMOC(
        gamma=GAMMA,
        mach_inlet=M_inlet,
        mach_lower=mach_lower,
        mach_upper=mach_upper,
        beta_inlet_deg=beta_inlet_deg,
        dv=dv,
    )
    moc.generate()
    dist = moc.surface_mach_distribution(plot=False)

    surf = dist[side]
    return surf['s_norm'], surf['mach'], dist[f'chord_{side}']


# ══════════════════════════════════════════════════════════════════════
# Boundary layer solver (von Karman + Head, reference temperature)
# ══════════════════════════════════════════════════════════════════════

def H1_from_H(H):
    if H > 1.6:
        return 3.3 + 0.8234 * (H - 1.1)**(-1.287)
    else:
        return 3.3 + 1.5501 * (H - 0.6778)**(-3.064)


def H_from_H1(H1):
    if H1 > 100:
        return 1.2
    try:
        root = brentq(lambda H: H1_from_H(H) - H1, 1.2, 4.0)
        if isinstance(root, tuple):
            root = root[0]
        return float(root)
    except ValueError:
        return 1.4 if H1 > H1_from_H(1.4) else 3.5


def Cf_LT(H, Re_theta):
    if Re_theta < 10:
        Re_theta = 10
    return 0.246 * 10**(-0.678 * H) * Re_theta**(-0.268)


def CE_head(H1):
    if H1 <= 3.03:
        H1 = 3.03
    return 0.0306 * (H1 - 3.0)**(-0.6169)


def ref_nu_ratio(Me, T0):
    """Ratio of reference kinematic viscosity to edge kinematic viscosity."""
    Te = T0 / T0_over_T(Me)
    r_rec = 0.89
    alpha = r_rec * GM1 / 2 * Me**2
    Tstar_Te = 1 + 0.72 * alpha
    T_star = Te * Tstar_Te
    mu_star_over_mu_e = mu_ratio_sutherland(T_star, Te)
    rho_star_over_rho_e = 1.0 / Tstar_Te
    return mu_star_over_mu_e / rho_star_over_rho_e


def adiabatic_wall_temperature_ratio(Me, pr=PRANDTL):
    """Adiabatic-wall temperature ratio T_w/T_0."""
    recovery = pr ** (1.0 / 3.0)
    Te_T0 = 1.0 / T0_over_T(Me)
    return recovery + (1.0 - recovery) * Te_T0


def reference_temperature_ratio(Me, pr=PRANDTL):
    """Eckert reference-temperature ratio \bar{T}/T_0 from Sasman-Cresci Eq. (9)."""
    Te_T0 = 1.0 / T0_over_T(Me)
    Tw_T0 = adiabatic_wall_temperature_ratio(Me, pr=pr)
    return 0.5 * Tw_T0 + 0.22 * pr**(1.0 / 3.0) + (0.5 - 0.22 * pr**(1.0 / 3.0)) * Te_T0


def Cf_sasman_cresci(Hi, Re_theta_ref, Me, pr=PRANDTL):
    """Sasman-Cresci/Goldman Eq. (10) for Cf/2, returned as Cf."""
    Hi_eff = max(float(Hi), 1.02)
    Re_eff = max(float(Re_theta_ref), 10.0)
    Te_Tbar = 1.0 / reference_temperature_ratio(Me, pr=pr)
    Tbar_T0 = reference_temperature_ratio(Me, pr=pr)
    mu_bar_over_mu0 = mu_ratio_sutherland(Tbar_T0 * 500.0, 500.0)
    cf_over_2 = 0.123 * np.exp(-1.561 * Hi_eff) * Re_eff**(-0.268) * Te_Tbar * mu_bar_over_mu0**0.268
    return 2.0 * cf_over_2


def shear_integral_sasman_cresci(Hi, Cf):
    """Equilibrium shear-integral closure from Sasman-Cresci."""
    Hi_eff = max(float(Hi), 1.05)
    return 0.011 / Hi_eff + Cf / 2.0


def bl_ode(s, y, Me_fn, dMe_ds_fn, Re_unit_fn, nu_ratio_fn):
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
    Re_theta = Re_unit_e * theta / nu_rat

    Cf = Cf_LT(H, Re_theta)
    CE = CE_head(H1)

    dtheta = Cf / 2 - theta * (H + 2) * inv_ue_due_ds
    dH1 = (CE - H1 * Cf / 2 + H1 * theta * (H + 1) * inv_ue_due_ds) / theta

    return [dtheta, dH1]


def bl_ode_sasman_cresci(s, y, Me_fn, dMe_ds_fn, Re0, T0=500.0):
    """Reduced Sasman-Cresci form following exactly the 1966 paper."""
    f, Hi = y
    f = float(max(f, 1e-12))
    Hi = float(np.clip(Hi, 1.05, 4.5))

    Me = float(Me_fn(s))
    dMe_ds = float(dMe_ds_fn(s))

    Te_T0 = 1.0 / (1.0 + 0.2 * Me**2)
    pr = 0.72
    Taw_T0 = Te_T0 + pr**(1.0/3.0) * (1.0 - Te_T0)
    gw = Taw_T0  # adiabatic wall

    # Eq. 9: Eckert reference temperature ratio
    Tbar_T0 = 0.5 * Taw_T0 + 0.22 * pr**(1.0/3.0) + (0.5 - 0.22 * pr**(1.0/3.0)) * Te_T0
    
    S_T0 = 110.4 / T0
    mu_bar_mu0 = (Tbar_T0)**1.5 * (1.0 + S_T0) / (Tbar_T0 + S_T0)

    # A parameter (dimensionless when multiplied by c)
    cA = 0.123 * np.exp(-1.561 * Hi) * Me * Re0 * Te_T0 * mu_bar_mu0**0.268

    # Eq. 19: f evolution
    df = 1.268 * ( -(f / Me) * dMe_ds * (1.0 + gw * Hi) + cA )

    # Skin friction from Eq. 10
    cf_over_2 = 0.123 * np.exp(-1.561 * Hi) * f**(-0.268 / 1.268) * (Te_T0 / Tbar_T0) * mu_bar_mu0**0.268
    cf_over_2 = max(cf_over_2, 1e-10)

    # Eq. 20: Hi evolution
    dHi_pressure = - (1.0 / (2.0 * Me)) * dMe_ds * Hi * (Hi + 1.0)**2 * (Hi - 1.0) * \
                   (1.0 + (gw - 1.0) * (Hi**2 + 4.0 * Hi - 1.0) / ((Hi + 1.0) * (Hi + 3.0)))
                   
    dHi_shear = ((Hi**2 - 1.0) / f) * cA * \
                (Hi - 0.011 * (Hi + 1.0) * (Hi - 1.0)**2 / Hi**2 / cf_over_2 * (Te_T0 / Tbar_T0))

    dHi = float(np.clip(dHi_pressure + dHi_shear, -50.0, 50.0))

    return [df, dHi]


def solve_bl(s_norm, Me_arr, Re_chord, M_in, T0=500.0):
    """
    Solve BL for given surface Mach distribution.
    s_norm: arc length / chord (0 to 1)
    Me_arr: edge Mach at each point
    Returns: s, theta, Hi, Me (all arrays)
    """
    # Remove any duplicate s values and ensure monotonicity
    mask = np.diff(s_norm, prepend=-1) > 0
    s_norm = s_norm[mask]
    Me_arr = Me_arr[mask]

    # Smooth Mach distribution with spline
    Me_spline = PchipInterpolator(s_norm, Me_arr)
    dMe_spline = Me_spline.derivative()

    # Edge conditions
    T_in = T0 / T0_over_T(M_in)
    Me_eval = Me_spline(s_norm)

    Te = T0 / T0_over_T(Me_eval)
    ae_over_ain = np.sqrt(Te / T_in)
    ue_ratio = Me_eval / M_in * ae_over_ain
    rho_ratio = rho0_over_rho(M_in) / rho0_over_rho(Me_eval)
    mu_rat = np.array([mu_ratio_sutherland(t, T_in) for t in Te])

    Re_unit_arr = Re_chord * rho_ratio * ue_ratio / mu_rat
    nu_ratio_arr = np.array([ref_nu_ratio(m, T0) for m in Me_eval])

    Re_unit_spline = PchipInterpolator(s_norm, Re_unit_arr)
    nu_ratio_spline = PchipInterpolator(s_norm, nu_ratio_arr)

    # Initial conditions
    s0 = s_norm[3]
    Re_u0 = float(Re_unit_spline(s0))
    nu_r0 = float(nu_ratio_spline(s0))
    Re_s0 = Re_u0 * s0 / nu_r0
    theta_0 = 0.036 * s0 * max(Re_s0, 10)**(-0.2)
    H_0 = 1.71
    H1_0 = H1_from_H(H_0)

    s_eval = s_norm[3:]

    sol = solve_ivp(
        bl_ode,
        [s0, s_norm[-1]],
        [theta_0, H1_0],
        args=(Me_spline, dMe_spline, Re_unit_spline, nu_ratio_spline),
        t_eval=s_eval,
        method='Radau',
        rtol=1e-6, atol=1e-10,
        max_step=0.003,
    )

    if not sol.success:
        print(f"  Warning: solver stopped at s/c={sol.t[-1]:.3f} -- {sol.message}")

    s_sol = sol.t
    theta_sol = sol.y[0]
    H1_sol = sol.y[1]
    Hi_sol = np.array([H_from_H1(h1) for h1 in H1_sol])
    Me_sol = Me_spline(s_sol)

    return s_sol, theta_sol, Hi_sol, Me_sol


def solve_bl_sasman_cresci(s_norm, Me_arr, Re_chord, M_in, T0=500.0):
    """
    Solve BL with a Sasman-Cresci closure set.
    Returns: s, theta, Hi, Me (all arrays)
    """
    mask = np.diff(s_norm, prepend=-1) > 0
    s_norm = s_norm[mask]
    Me_arr = Me_arr[mask]

    Me_spline = PchipInterpolator(s_norm, Me_arr)
    dMe_spline = Me_spline.derivative()

    T_in = T0 / (1.0 + 0.2 * M_in**2)
    S = 110.4
    mu_in_mu0 = (T_in / T0)**1.5 * (1.0 + S / T0) / (T_in / T0 + S / T0)
    
    # Re0 = a0 * c / nu0
    Re0 = (Re_chord / M_in) * ((T0 / T_in)**3.0) * mu_in_mu0

    s0 = s_norm[3]
    Me0 = Me_spline(s0)
    Te0_T0 = 1.0 / (1.0 + 0.2 * Me0**2)
    T_in_T0 = 1.0 / (1.0 + 0.2 * M_in**2)
    
    rho_e_rho_in = (Te0_T0 / T_in_T0)**2.5
    u_e_u_in = (Me0 / M_in) * np.sqrt(Te0_T0 / T_in_T0)
    
    Te0 = T0 * Te0_T0
    mu_e_mu_in = (Te0 / T_in)**1.5 * (T_in + S) / (Te0 + S)
    
    Re_unit0 = Re_chord * rho_e_rho_in * u_e_u_in / mu_e_mu_in
    Re_s0 = Re_unit0 * s0
    
    theta_0_c = 0.036 * s0 * max(Re_s0, 10.0)**(-0.2)
    theta_bar_0_c = theta_0_c * (Te0_T0)**3.0
    f_0 = (Me0 * theta_bar_0_c * Re0)**1.268
    Hi_0 = 1.67

    s_eval = s_norm[3:]

    sol = solve_ivp(
        bl_ode_sasman_cresci,
        [s0, s_norm[-1]],
        [f_0, Hi_0],
        args=(Me_spline, dMe_spline, Re0, T0),
        t_eval=s_eval,
        method='Radau',
        rtol=1e-6, atol=1e-10,
        max_step=0.003,
    )

    if not sol.success:
        print(f"  Warning [SC]: solver stopped at s/c={sol.t[-1]:.3f} -- {sol.message}")

    s_sol = sol.t
    f_sol = sol.y[0]
    Hi_sol = sol.y[1]
    
    Me_sol = Me_spline(s_sol)
    Te_T0_sol = 1.0 / (1.0 + 0.2 * Me_sol**2)
    theta_bar_c_sol = f_sol**(1.0 / 1.268) / (Me_sol * Re0)
    theta_sol = theta_bar_c_sol / (Te_T0_sol**3.0)

    return s_sol, theta_sol, Hi_sol, Me_sol


# ══════════════════════════════════════════════════════════════════════
# Main: replicate Goldman Figures 4, 5, 8, 9
# ══════════════════════════════════════════════════════════════════════

def run_single_case(M_in, M_lower, M_upper, beta_deg, Re_chord):
    """Run BL on both surfaces for one design point."""
    s_lo, Me_lo, c_lo = build_blade_surface(
        M_in, M_lower, beta_deg, side='lower', M_other=M_upper
    )
    s_up, Me_up, c_up = build_blade_surface(
        M_in, M_upper, beta_deg, side='upper', M_other=M_lower
    )

    s_lo_bl, th_lo, Hi_lo, Me_lo_bl = solve_bl(s_lo, Me_lo, Re_chord, M_in)
    s_up_bl, th_up, Hi_up, Me_up_bl = solve_bl(s_up, Me_up, Re_chord, M_in)

    return (s_lo_bl, Hi_lo, Me_lo_bl, th_lo,
            s_up_bl, Hi_up, Me_up_bl, th_up)


def run_single_case_sasman_cresci(M_in, M_lower, M_upper, beta_deg, Re_chord):
    """Run Sasman-Cresci BL on both surfaces for one design point."""
    s_lo, Me_lo, _ = build_blade_surface(
        M_in, M_lower, beta_deg, side='lower', M_other=M_upper
    )
    s_up, Me_up, _ = build_blade_surface(
        M_in, M_upper, beta_deg, side='upper', M_other=M_lower
    )

    s_lo_bl, th_lo, Hi_lo, Me_lo_bl = solve_bl_sasman_cresci(s_lo, Me_lo, Re_chord, M_in)
    s_up_bl, th_up, Hi_up, Me_up_bl = solve_bl_sasman_cresci(s_up, Me_up, Re_chord, M_in)

    return (s_lo_bl, Hi_lo, Me_lo_bl, th_lo,
            s_up_bl, Hi_up, Me_up_bl, th_up)


def main():
    # ── Goldman conditions ──
    M_in = 2.5
    beta_deg = 70.0
    Re_chord = 35000

    # Typical case: nu_l=22 deg, nu_u=49 deg (matches Fig 5 M_lower ~ 1.84)
    nu_l_rad = np.deg2rad(22.0)
    nu_u_rad = np.deg2rad(49.0)
    M_lower = M_from_nu(nu_l_rad)
    M_upper = M_from_nu(nu_u_rad)
    print(f"Typical case: M_lower={M_lower:.3f} (nu=22), M_upper={M_upper:.3f} (nu=49)")

    results = run_single_case(M_in, M_lower, M_upper, beta_deg, Re_chord)
    s_lo, Hi_lo, Me_lo, th_lo = results[0], results[1], results[2], results[3]
    s_up, Hi_up, Me_up, th_up = results[4], results[5], results[6], results[7]

    results_sc = run_single_case_sasman_cresci(M_in, M_lower, M_upper, beta_deg, Re_chord)
    s_lo_sc, Hi_lo_sc, Me_lo_sc, th_lo_sc = results_sc[0], results_sc[1], results_sc[2], results_sc[3]
    s_up_sc, Hi_up_sc, Me_up_sc, th_up_sc = results_sc[4], results_sc[5], results_sc[6], results_sc[7]

    print(f"Lower surface: Hi_max = {np.max(Hi_lo):.3f} at s/c = {s_lo[np.argmax(Hi_lo)]:.3f}")
    print(f"Upper surface: Hi_max = {np.max(Hi_up):.3f} at s/c = {s_up[np.argmax(Hi_up)]:.3f}")
    print(f"Lower surface [SC]: Hi_max = {np.max(Hi_lo_sc):.3f} at s/c = {s_lo_sc[np.argmax(Hi_lo_sc)]:.3f}")
    print(f"Upper surface [SC]: Hi_max = {np.max(Hi_up_sc):.3f} at s/c = {s_up_sc[np.argmax(Hi_up_sc)]:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # Figure 5: Surface Mach number distribution
    # ═══════════════════════════════════════════════════════════════
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    ax5.plot(s_lo, Me_lo, 'b-', lw=2, label='Lower surface')
    ax5.plot(s_up, Me_up, 'r-', lw=2, label='Upper surface')
    ax5.set_xlabel('Fraction of chord', fontsize=12)
    ax5.set_ylabel('Surface Mach number, $M_e$', fontsize=12)
    ax5.set_title('Typical variation of surface Mach number with axial distance\n'
                   '(cf. Goldman Fig. 5)', fontsize=12)
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 1)
    ax5.set_ylim(1.5, 3.5)
    fig5.tight_layout()
    fig5.savefig('C:/Users/Martin/Active/FYPTurbine/quicktests/goldman_fig5.png', dpi=150)
    print("\nSaved: goldman_fig5.png")

    # ═══════════════════════════════════════════════════════════════
    # Figure 4: Incompressible form factor Hi
    # ═══════════════════════════════════════════════════════════════
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.plot(s_lo, Hi_lo, 'b-', lw=2, label='Lower surface (Head)')
    ax4.plot(s_up, Hi_up, 'r-', lw=2, label='Upper surface (Head)')
    ax4.plot(s_lo_sc, Hi_lo_sc, 'b--', lw=2, label='Lower surface (SC)')
    ax4.plot(s_up_sc, Hi_up_sc, 'r--', lw=2, label='Upper surface (SC)')
    ax4.axhspan(1.8, 2.4, alpha=0.08, color='red', label='Separation range')
    ax4.axhline(1.8, color='gray', ls='--', alpha=0.4)
    ax4.axhline(2.4, color='gray', ls='--', alpha=0.4)
    ax4.set_xlabel('Fraction of chord', fontsize=12)
    ax4.set_ylabel('Incompressible form factor, $H_i$', fontsize=12)
    ax4.set_title('Typical variation of incompressible form factor with axial distance\n'
                   '(cf. Goldman Fig. 4)', fontsize=12)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(1.3, 2.6)
    fig4.tight_layout()
    fig4.savefig('C:/Users/Martin/Active/FYPTurbine/quicktests/goldman_fig4.png', dpi=150)
    print("Saved: goldman_fig4.png")

    # ═══════════════════════════════════════════════════════════════
    # Figure 8: Hi_max on lower surface vs nu_l (nu_u=49 fixed)
    # ═══════════════════════════════════════════════════════════════
    nu_l_range_deg = np.arange(18, 35, 2.0)
    Hi_max_lower = []
    Hi_max_lower_sc = []
    print("\nSweeping nu_lower for Fig 8...")
    for nu_l_deg in nu_l_range_deg:
        Ml = M_from_nu(np.deg2rad(nu_l_deg))
        try:
            sl, Mel, cl = build_blade_surface(
                M_in, Ml, beta_deg, side='lower',
                M_other=M_from_nu(np.deg2rad(49.0))
            )
            s_bl, _, Hi_bl, _ = solve_bl(sl, Mel, Re_chord, M_in)
            hmax = np.max(Hi_bl)
            Hi_max_lower.append(hmax)
            
            s_bl_sc, _, Hi_bl_sc, _ = solve_bl_sasman_cresci(sl, Mel, Re_chord, M_in)
            hmax_sc = np.max(Hi_bl_sc)
            Hi_max_lower_sc.append(hmax_sc)
            
            print(f"  nu_l={nu_l_deg:.0f}: M_l={Ml:.3f}, Hi_max(Head)={hmax:.3f}, Hi_max(SC)={hmax_sc:.3f}")
        except Exception as e:
            print(f"  nu_l={nu_l_deg:.0f}: FAILED ({e})")
            Hi_max_lower.append(np.nan)
            Hi_max_lower_sc.append(np.nan)

    # Corresponding Mach numbers for secondary x-axis
    Ml_for_axis = [M_from_nu(np.deg2rad(n)) for n in nu_l_range_deg]

    fig8, ax8 = plt.subplots(figsize=(8, 6))
    ax8.plot(nu_l_range_deg, Hi_max_lower, 'bo-', lw=2, markersize=6, label='Head')
    ax8.plot(nu_l_range_deg, Hi_max_lower_sc, 'b^--', lw=2, markersize=6, label='Sasman-Cresci')
    ax8.axhspan(1.8, 2.4, alpha=0.08, color='red')
    ax8.axhline(1.8, color='gray', ls='--', alpha=0.4)
    ax8.axhline(2.4, color='gray', ls='--', alpha=0.4)
    ax8.set_xlabel('Lower-surface Prandtl-Meyer angle, $\\nu_l$ (deg)', fontsize=12)
    ax8.set_ylabel('Maximum $H_i$ on lower surface', fontsize=12)
    ax8.set_title('Effect of lower-surface Prandtl-Meyer angle on $H_{i,max}$\n'
                   '(cf. Goldman Fig. 8, $\\nu_u = 49°$)', fontsize=12)
    ax8.grid(True, alpha=0.3)
    ax8.legend(fontsize=11)
    # Secondary axis: Mach number
    ax8b = ax8.twiny()
    ax8b.set_xlim(ax8.get_xlim())
    tick_positions = nu_l_range_deg[::2]
    tick_labels = [f"{M_from_nu(np.deg2rad(n)):.2f}" for n in tick_positions]
    ax8b.set_xticks(tick_positions)
    ax8b.set_xticklabels(tick_labels)
    ax8b.set_xlabel('Lower-surface Mach number, $M_l$', fontsize=11)
    fig8.tight_layout()
    fig8.savefig('C:/Users/Martin/Active/FYPTurbine/quicktests/goldman_fig8.png', dpi=150)
    print("Saved: goldman_fig8.png")

    # ═══════════════════════════════════════════════════════════════
    # Figure 9: Hi_max on upper surface vs nu_u (nu_l=34 fixed)
    # ═══════════════════════════════════════════════════════════════
    nu_u_range_deg = np.arange(42, 55, 1.5)
    Hi_max_upper = []
    Hi_max_upper_sc = []
    print("\nSweeping nu_upper for Fig 9...")
    for nu_u_deg in nu_u_range_deg:
        Mu = M_from_nu(np.deg2rad(nu_u_deg))
        try:
            su, Meu, cu = build_blade_surface(
                M_in, Mu, beta_deg, side='upper',
                M_other=M_from_nu(np.deg2rad(34.0))
            )
            s_bl, _, Hi_bl, _ = solve_bl(su, Meu, Re_chord, M_in)
            hmax = np.max(Hi_bl)
            Hi_max_upper.append(hmax)
            
            s_bl_sc, _, Hi_bl_sc, _ = solve_bl_sasman_cresci(su, Meu, Re_chord, M_in)
            hmax_sc = np.max(Hi_bl_sc)
            Hi_max_upper_sc.append(hmax_sc)
            
            print(f"  nu_u={nu_u_deg:.1f}: M_u={Mu:.3f}, Hi_max(Head)={hmax:.3f}, Hi_max(SC)={hmax_sc:.3f}")
        except Exception as e:
            print(f"  nu_u={nu_u_deg:.1f}: FAILED ({e})")
            Hi_max_upper.append(np.nan)
            Hi_max_upper_sc.append(np.nan)

    fig9, ax9 = plt.subplots(figsize=(8, 6))
    ax9.plot(nu_u_range_deg, Hi_max_upper, 'ro-', lw=2, markersize=6, label='Head')
    ax9.plot(nu_u_range_deg, Hi_max_upper_sc, 'r^--', lw=2, markersize=6, label='Sasman-Cresci')
    ax9.axhspan(1.8, 2.4, alpha=0.08, color='red')
    ax9.axhline(1.8, color='gray', ls='--', alpha=0.4)
    ax9.axhline(2.4, color='gray', ls='--', alpha=0.4)
    ax9.set_xlabel('Upper-surface Prandtl-Meyer angle, $\\nu_u$ (deg)', fontsize=12)
    ax9.set_ylabel('Maximum $H_i$ on upper surface', fontsize=12)
    ax9.set_title('Effect of upper-surface Prandtl-Meyer angle on $H_{i,max}$\n'
                   '(cf. Goldman Fig. 9, $\\nu_l = 34°$)', fontsize=12)
    ax9.grid(True, alpha=0.3)
    ax9.legend(fontsize=11)
    ax9b = ax9.twiny()
    ax9b.set_xlim(ax9.get_xlim())
    tick_positions_u = nu_u_range_deg[::2]
    tick_labels_u = [f"{M_from_nu(np.deg2rad(n)):.2f}" for n in tick_positions_u]
    ax9b.set_xticks(tick_positions_u)
    ax9b.set_xticklabels(tick_labels_u)
    ax9b.set_xlabel('Upper-surface Mach number, $M_u$', fontsize=11)
    fig9.tight_layout()
    fig9.savefig('C:/Users/Martin/Active/FYPTurbine/quicktests/goldman_fig9.png', dpi=150)
    print("Saved: goldman_fig9.png")

    # ═══════════════════════════════════════════════════════════════
    # Combined overview plot
    # ═══════════════════════════════════════════════════════════════
    dstar_lo = Hi_lo * th_lo
    dstar_up = Hi_up * th_up
    dstar_lo_sc = Hi_lo_sc * th_lo_sc
    dstar_up_sc = Hi_up_sc * th_up_sc

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    fig.suptitle(
        'Goldman (1970) Replication --- MoC blade + von Karman/Head & Sasman-Cresci BL\n'
        f'$M_{{in}}=2.5$, $\\beta_{{in}}=70°$, $Re={Re_chord}$, $\\gamma=1.4$',
        fontsize=14
    )

    # Fig 5
    ax = axes[0, 0]
    ax.plot(s_lo, Me_lo, 'b-', lw=2, label='Lower')
    ax.plot(s_up, Me_up, 'r-', lw=2, label='Upper')
    ax.set_xlabel('Fraction of chord')
    ax.set_ylabel('Surface Mach number')
    ax.set_title('Fig. 5: Surface Mach distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Fig 4
    ax = axes[0, 1]
    ax.plot(s_lo, Hi_lo, 'b-', lw=2, label='Lower (Head)')
    ax.plot(s_up, Hi_up, 'r-', lw=2, label='Upper (Head)')
    ax.plot(s_lo_sc, Hi_lo_sc, 'b--', lw=2, label='Lower (SC)')
    ax.plot(s_up_sc, Hi_up_sc, 'r--', lw=2, label='Upper (SC)')
    ax.axhspan(1.8, 2.4, alpha=0.08, color='red')
    ax.axhline(1.8, color='gray', ls='--', alpha=0.4)
    ax.axhline(2.4, color='gray', ls='--', alpha=0.4)
    ax.set_xlabel('Fraction of chord')
    ax.set_ylabel('$H_i$')
    ax.set_title('Fig. 4: Incompressible form factor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Theta plot
    ax = axes[1, 0]
    ax.plot(s_lo, th_lo, 'b-', lw=2, label='Lower (Head)')
    ax.plot(s_up, th_up, 'r-', lw=2, label='Upper (Head)')
    ax.plot(s_lo_sc, th_lo_sc, 'b--', lw=2, label='Lower (SC)')
    ax.plot(s_up_sc, th_up_sc, 'r--', lw=2, label='Upper (SC)')
    ax.set_xlabel('Fraction of chord')
    ax.set_ylabel(r'Momentum thickness, $\theta / c$')
    ax.set_title(r'Momentum thickness $\theta / c$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Dstar plot
    ax = axes[1, 1]
    ax.plot(s_lo, dstar_lo, 'b-', lw=2, label='Lower (Head)')
    ax.plot(s_up, dstar_up, 'r-', lw=2, label='Upper (Head)')
    ax.plot(s_lo_sc, dstar_lo_sc, 'b--', lw=2, label='Lower (SC)')
    ax.plot(s_up_sc, dstar_up_sc, 'r--', lw=2, label='Upper (SC)')
    ax.set_xlabel('Fraction of chord')
    ax.set_ylabel(r'Displacement thickness, $\delta^* / c$')
    ax.set_title(r'Displacement thickness $\delta^* / c$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Fig 8
    ax = axes[2, 0]
    ax.plot(nu_l_range_deg, Hi_max_lower, 'bo-', lw=2, markersize=5, label='Head')
    ax.plot(nu_l_range_deg, Hi_max_lower_sc, 'b^--', lw=2, markersize=5, label='SC')
    ax.axhspan(1.8, 2.4, alpha=0.08, color='red')
    ax.axhline(1.8, color='gray', ls='--', alpha=0.4)
    ax.axhline(2.4, color='gray', ls='--', alpha=0.4)
    ax.set_xlabel('$\\nu_l$ (deg)')
    ax.set_ylabel('$H_{i,max}$ on lower surface')
    ax.set_title('Fig. 8: $H_{i,max}$ vs $\\nu_l$ ($\\nu_u=49°$)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Fig 9
    ax = axes[2, 1]
    ax.plot(nu_u_range_deg, Hi_max_upper, 'ro-', lw=2, markersize=5, label='Head')
    ax.plot(nu_u_range_deg, Hi_max_upper_sc, 'r^--', lw=2, markersize=5, label='SC')
    ax.axhspan(1.8, 2.4, alpha=0.08, color='red')
    ax.axhline(1.8, color='gray', ls='--', alpha=0.4)
    ax.axhline(2.4, color='gray', ls='--', alpha=0.4)
    ax.set_xlabel('$\\nu_u$ (deg)')
    ax.set_ylabel('$H_{i,max}$ on upper surface')
    ax.set_title('Fig. 9: $H_{i,max}$ vs $\\nu_u$ ($\\nu_l=34°$)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig('C:/Users/Martin/Active/FYPTurbine/quicktests/goldman_combined.png', dpi=150)
    print("\nSaved: goldman_combined.png")

    plt.close('all')


if __name__ == '__main__':
    main()
