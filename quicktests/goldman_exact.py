"""
Goldman (1970) NASA TM X-2059 — exact replication of Figs 4, 5, 8, 9.

Uses the Method of Characteristics (from turbinemoc.ipynb) to compute the
actual blade surface geometry and Mach distribution, then feeds it into
the von Karman + Head entrainment BL solver.

Goldman conditions: gamma=1.4, M_in=2.5, beta_in=70 deg, Re=35000.
"""

import numpy as np
from scipy.optimize import fsolve, brentq
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════
# Gas dynamics
# ══════════════════════════════════════════════════════════════════════
GAMMA = 1.4
GM1 = GAMMA - 1
GP1 = GAMMA + 1


def Mstar(M):
    return np.sqrt(GP1 / 2 * M**2 / (1 + GM1 / 2 * M**2))


def safe_asin_scalar(x):
    return np.arcsin(np.clip(x, -1, 1))


def nu_pm(M):
    """Prandtl-Meyer angle (radians)."""
    Ms = Mstar(M)
    return (np.pi / 4 * (np.sqrt(GP1 / GM1) - 1)
            + 0.5 * (np.sqrt(GP1 / GM1) * safe_asin_scalar(GM1 * Ms**2 - GAMMA)
                      + safe_asin_scalar(GP1 / Ms**2 - GAMMA)))


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
# Method of Characteristics — transition arc computation
# ══════════════════════════════════════════════════════════════════════

def safe_asin(x):
    return np.arcsin(np.clip(x, -1, 1))


def compute_transition_arc(M_inlet, M_surface, beta_inlet, dv=0.001, side='lower'):
    """
    MoC computation of one transition arc (inlet side).

    Returns:
        xs, ys : surface point coords (un-rotated frame), ordered from
                 vortex end → inlet end
        Machs  : Mach number at each surface point (same ordering)
        alpha  : half-angle of the circular arc on the inlet side
    """
    nu_in = nu_pm(M_inlet)
    nu_s = nu_pm(M_surface)

    if side == 'lower':
        delta_nu = nu_in - nu_s          # positive: flow decelerates
        sign = -1
    else:
        delta_nu = nu_s - nu_in          # positive: flow accelerates
        sign = +1

    alpha = beta_inlet - delta_nu        # half-angle for circular arc

    n_steps = int(np.ceil(delta_nu / dv))
    if n_steps < 2:
        n_steps = 2
    actual_dv = delta_nu / n_steps

    # u_i function (Mach angle from critical velocity ratio)
    if side == 'lower':
        u_i_fn = lambda Rs: -np.arcsin(np.sqrt(0.5 * GP1 * Rs**2 - 0.5 * GM1))
    else:
        u_i_fn = lambda Rs: np.arcsin(np.sqrt(0.5 * GP1 * Rs**2 - 0.5 * GM1))

    # Initial conditions at vortex region boundary
    R0 = 1 / Mstar(M_surface)
    phi_k1 = 0.0
    u_i_k1 = u_i_fn(R0)
    xlstar_k1 = 0.0
    ylstar_k1 = R0

    xs, ys, machs = [], [], []

    for k in range(n_steps, 0, -1):
        phi_k = delta_nu - k * actual_dv

        # Solve for Rstar at this characteristic
        if side == 'lower':
            fRk = 2 * nu_in - np.pi / 2 * (np.sqrt(GP1 / GM1) - 1) - 2 * k * actual_dv
        else:
            fRk = 2 * nu_in - np.pi / 2 * (np.sqrt(GP1 / GM1) - 1) + 2 * k * actual_dv

        def eq(Rs):
            return (np.sqrt(GP1 / GM1) * safe_asin(GM1 / Rs**2 - GAMMA)
                    + safe_asin(GP1 * Rs**2 - GAMMA) - fRk)

        Rstar = fsolve(eq, 0.5 if side == 'upper' else 0.9, full_output=False)[0]

        xkstar = -Rstar * np.sin(phi_k)
        ykstar = Rstar * np.cos(phi_k)

        u_i_k = u_i_fn(Rstar)
        mi_k = np.tan(0.5 * (phi_k + phi_k1) + 0.5 * (u_i_k + u_i_k1))
        mbar_k = np.tan(phi_k1)

        xsl = ((ylstar_k1 - mbar_k * xlstar_k1) - (ykstar - mi_k * xkstar)) / (mi_k - mbar_k)
        ysl = (mi_k * (ylstar_k1 - mbar_k * xlstar_k1) - mbar_k * (ykstar - mi_k * xkstar)) / (mi_k - mbar_k)

        phi_k1 = phi_k
        u_i_k1 = u_i_k
        xlstar_k1 = xsl
        ylstar_k1 = ysl

        xs.append(xsl)
        ys.append(ysl)

        # Mach at this characteristic intersection
        if side == 'lower':
            nu_local = nu_in - k * actual_dv
        else:
            nu_local = nu_in + k * actual_dv
        machs.append(M_from_nu(nu_local))

    return np.array(xs), np.array(ys), np.array(machs), alpha


def build_blade_surface(M_inlet, M_surface, beta_inlet_rad, side='lower', dv=0.001, M_other=None):
    """
    Build complete blade surface from inlet to outlet.

    Returns:
        s_norm : arc length / chord  (0 to 1)
        Me     : edge Mach number at each point
    """
    xs_t, ys_t, machs_t, alpha_in = compute_transition_arc(
        M_inlet, M_surface, beta_inlet_rad, dv=dv, side=side
    )
    
    straight_len = 0.0
    if M_other is not None:
        xs_o, ys_o, _, alpha_o = compute_transition_arc(
            M_inlet, M_other, beta_inlet_rad, dv=dv, side='upper' if side == 'lower' else 'lower'
        )
        def rot(x, y, ang):
            return x * np.cos(ang) - y * np.sin(ang), x * np.sin(ang) + y * np.cos(ang)
        
        x_in, _ = rot(xs_t[-1], ys_t[-1], alpha_in)
        x_oth, _ = rot(xs_o[-1], ys_o[-1], alpha_o)
        
        if side == 'upper':
            dx = x_oth - x_in
            straight_len = abs(dx / np.cos(beta_inlet_rad))

    # Transition arc is ordered: vortex end (index 0) -> inlet end (index -1)
    # For the blade surface (inlet -> outlet), reverse for inlet transition
    xs_inlet = xs_t[::-1]       # inlet end -> vortex end
    ys_inlet = ys_t[::-1]
    M_inlet_trans = machs_t[::-1]   # M near inlet -> M_surface

    # Add exact endpoint Machs for continuity
    M_inlet_trans = np.concatenate([[M_inlet], M_inlet_trans, [M_surface]])
    # Extrapolate surface coords slightly for endpoints
    xs_inlet = np.concatenate([
        [xs_inlet[0] + (xs_inlet[0] - xs_inlet[1]) * 0.1],
        xs_inlet,
        [xs_inlet[-1] + (xs_inlet[-1] - xs_inlet[-2]) * 0.1]
    ])
    ys_inlet = np.concatenate([
        [ys_inlet[0] + (ys_inlet[0] - ys_inlet[1]) * 0.1],
        ys_inlet,
        [ys_inlet[-1] + (ys_inlet[-1] - ys_inlet[-2]) * 0.1]
    ])

    # Vortex region: circular arc at radius R = 1/Mstar(M_surface)
    R_s = 1 / Mstar(M_surface)

    # Outlet transition: Goldman uses shorter outlet arcs so G_out < G_in.
    # For impulse, outlet has same Mach variation but compressed spatially.
    # Use 65% of inlet transition (Goldman's iterative procedure typically
    # yields outlet arcs 50-70% of inlet for displacement thickness correction).
    outlet_frac = 0.65
    n_out = max(int(len(xs_t) * outlet_frac), 3)
    # Outlet transition: vortex end -> outlet end (same Mach variation as inlet but reversed)
    # Resample to get evenly spaced Mach variation over shorter arc
    outlet_idx = np.linspace(0, len(xs_t) - 1, n_out).astype(int)
    xs_outlet = -xs_t[outlet_idx]     # x-reflected for outlet side
    ys_outlet = ys_t[outlet_idx]
    M_outlet_trans = machs_t[outlet_idx]  # vortex end -> inlet end

    # Add endpoints
    M_outlet_trans = np.concatenate([[M_surface], M_outlet_trans, [M_inlet]])
    xs_outlet = np.concatenate([
        [xs_outlet[0] + (xs_outlet[0] - xs_outlet[1]) * 0.1],
        xs_outlet,
        [xs_outlet[-1] + (xs_outlet[-1] - xs_outlet[-2]) * 0.1]
    ])
    ys_outlet = np.concatenate([
        [ys_outlet[0] + (ys_outlet[0] - ys_outlet[1]) * 0.1],
        ys_outlet,
        [ys_outlet[-1] + (ys_outlet[-1] - ys_outlet[-2]) * 0.1]
    ])

    # Rotate transitions to align with cascade angle
    def rot(x, y, ang):
        return x * np.cos(ang) - y * np.sin(ang), x * np.sin(ang) + y * np.cos(ang)

    xs_inlet, ys_inlet = rot(xs_inlet, ys_inlet, alpha_in)
    xs_outlet, ys_outlet = rot(xs_outlet, ys_outlet, -alpha_in)

    # Circular arc between inlet and outlet transitions
    angle_start = np.arctan2(xs_inlet[-1], ys_inlet[-1])
    angle_end = np.arctan2(xs_outlet[0], ys_outlet[0])

    n_arc = 80
    arc_angles = np.linspace(angle_start, angle_end, n_arc)
    xs_arc = R_s * np.sin(arc_angles)
    ys_arc = R_s * np.cos(arc_angles)
    M_arc = np.full(n_arc, M_surface)

    # Short inlet straight
    inlet_dir = np.array([np.sin(beta_inlet_rad), np.cos(beta_inlet_rad)])
    n_str = max(5, int(straight_len * 100))
    ts = np.linspace(straight_len, 0, n_str)
    xs_str_in = xs_inlet[0] + inlet_dir[0] * ts
    ys_str_in = ys_inlet[0] + inlet_dir[1] * ts
    M_str_in = np.full(n_str, M_inlet)

    outlet_dir = np.array([-np.sin(beta_inlet_rad), np.cos(beta_inlet_rad)])
    xs_str_out = xs_outlet[-1] + outlet_dir[0] * np.linspace(0, straight_len, n_str)
    ys_str_out = ys_outlet[-1] + outlet_dir[1] * np.linspace(0, straight_len, n_str)
    M_str_out = np.full(n_str, M_inlet)

    # Concatenate full surface
    xs_full = np.concatenate([xs_str_in, xs_inlet, xs_arc, xs_outlet, xs_str_out])
    ys_full = np.concatenate([ys_str_in, ys_inlet, ys_arc, ys_outlet, ys_str_out])
    Me_full = np.concatenate([M_str_in, M_inlet_trans, M_arc, M_outlet_trans, M_str_out])

    # Compute arc length
    dx = np.diff(xs_full)
    dy = np.diff(ys_full)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate([[0], np.cumsum(ds)])
    chord = s[-1]
    s_norm = s / chord

    return s_norm, Me_full, chord


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
        return brentq(lambda H: H1_from_H(H) - H1, 1.2, 4.0)
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
    Me_spline = CubicSpline(s_norm, Me_arr)
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

    Re_unit_spline = CubicSpline(s_norm, Re_unit_arr)
    nu_ratio_spline = CubicSpline(s_norm, nu_ratio_arr)

    # Initial conditions
    s0 = s_norm[3]
    Re_u0 = float(Re_unit_spline(s0))
    nu_r0 = float(nu_ratio_spline(s0))
    Re_s0 = Re_u0 * s0 / nu_r0
    theta_0 = 0.036 * s0 * max(Re_s0, 10)**(-0.2)
    H_0 = 1.4
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


# ══════════════════════════════════════════════════════════════════════
# Main: replicate Goldman Figures 4, 5, 8, 9
# ══════════════════════════════════════════════════════════════════════

def run_single_case(M_in, M_lower, M_upper, beta_deg, Re_chord):
    """Run BL on both surfaces for one design point."""
    beta_rad = np.deg2rad(beta_deg)

    s_lo, Me_lo, c_lo = build_blade_surface(M_in, M_lower, beta_rad, side='lower', M_other=M_upper)
    s_up, Me_up, c_up = build_blade_surface(M_in, M_upper, beta_rad, side='upper', M_other=M_lower)

    s_lo_bl, th_lo, Hi_lo, Me_lo_bl = solve_bl(s_lo, Me_lo, Re_chord, M_in)
    s_up_bl, th_up, Hi_up, Me_up_bl = solve_bl(s_up, Me_up, Re_chord, M_in)

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

    print(f"Lower surface: Hi_max = {np.max(Hi_lo):.3f} at s/c = {s_lo[np.argmax(Hi_lo)]:.3f}")
    print(f"Upper surface: Hi_max = {np.max(Hi_up):.3f} at s/c = {s_up[np.argmax(Hi_up)]:.3f}")

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
    print("Saved: goldman_fig5.png")

    # ═══════════════════════════════════════════════════════════════
    # Figure 4: Incompressible form factor Hi
    # ═══════════════════════════════════════════════════════════════
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.plot(s_lo, Hi_lo, 'b-', lw=2, label='Lower surface')
    ax4.plot(s_up, Hi_up, 'r-', lw=2, label='Upper surface')
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
    ax4.set_ylim(1.0, 3.0)
    fig4.tight_layout()
    fig4.savefig('C:/Users/Martin/Active/FYPTurbine/quicktests/goldman_fig4.png', dpi=150)
    print("Saved: goldman_fig4.png")

    # ═══════════════════════════════════════════════════════════════
    # Figure 8: Hi_max on lower surface vs nu_l (nu_u=49 fixed)
    # ═══════════════════════════════════════════════════════════════
    nu_l_range_deg = np.arange(18, 35, 2.0)
    Hi_max_lower = []
    print("\nSweeping nu_lower for Fig 8...")
    for nu_l_deg in nu_l_range_deg:
        Ml = M_from_nu(np.deg2rad(nu_l_deg))
        Ml_range = np.linspace(Ml, Ml, 1)  # just for print
        try:
            sl, Mel, cl = build_blade_surface(M_in, Ml, np.deg2rad(beta_deg), side='lower', M_other=M_from_nu(np.deg2rad(49.0)))
            s_bl, _, Hi_bl, _ = solve_bl(sl, Mel, Re_chord, M_in)
            hmax = np.max(Hi_bl)
            Hi_max_lower.append(hmax)
            print(f"  nu_l={nu_l_deg:.0f}: M_l={Ml:.3f}, Hi_max={hmax:.3f}")
        except Exception as e:
            print(f"  nu_l={nu_l_deg:.0f}: FAILED ({e})")
            Hi_max_lower.append(np.nan)

    # Corresponding Mach numbers for secondary x-axis
    Ml_for_axis = [M_from_nu(np.deg2rad(n)) for n in nu_l_range_deg]

    fig8, ax8 = plt.subplots(figsize=(8, 6))
    ax8.plot(nu_l_range_deg, Hi_max_lower, 'bo-', lw=2, markersize=6)
    ax8.axhspan(1.8, 2.4, alpha=0.08, color='red')
    ax8.axhline(1.8, color='gray', ls='--', alpha=0.4)
    ax8.axhline(2.4, color='gray', ls='--', alpha=0.4)
    ax8.set_xlabel('Lower-surface Prandtl-Meyer angle, $\\nu_l$ (deg)', fontsize=12)
    ax8.set_ylabel('Maximum $H_i$ on lower surface', fontsize=12)
    ax8.set_title('Effect of lower-surface Prandtl-Meyer angle on $H_{i,max}$\n'
                   '(cf. Goldman Fig. 8, $\\nu_u = 49°$)', fontsize=12)
    ax8.grid(True, alpha=0.3)
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
    print("\nSweeping nu_upper for Fig 9...")
    for nu_u_deg in nu_u_range_deg:
        Mu = M_from_nu(np.deg2rad(nu_u_deg))
        try:
            su, Meu, cu = build_blade_surface(M_in, Mu, np.deg2rad(beta_deg), side='upper', M_other=M_from_nu(np.deg2rad(34.0)))
            s_bl, _, Hi_bl, _ = solve_bl(su, Meu, Re_chord, M_in)
            hmax = np.max(Hi_bl)
            Hi_max_upper.append(hmax)
            print(f"  nu_u={nu_u_deg:.1f}: M_u={Mu:.3f}, Hi_max={hmax:.3f}")
        except Exception as e:
            print(f"  nu_u={nu_u_deg:.1f}: FAILED ({e})")
            Hi_max_upper.append(np.nan)

    fig9, ax9 = plt.subplots(figsize=(8, 6))
    ax9.plot(nu_u_range_deg, Hi_max_upper, 'ro-', lw=2, markersize=6)
    ax9.axhspan(1.8, 2.4, alpha=0.08, color='red')
    ax9.axhline(1.8, color='gray', ls='--', alpha=0.4)
    ax9.axhline(2.4, color='gray', ls='--', alpha=0.4)
    ax9.set_xlabel('Upper-surface Prandtl-Meyer angle, $\\nu_u$ (deg)', fontsize=12)
    ax9.set_ylabel('Maximum $H_i$ on upper surface', fontsize=12)
    ax9.set_title('Effect of upper-surface Prandtl-Meyer angle on $H_{i,max}$\n'
                   '(cf. Goldman Fig. 9, $\\nu_l = 34°$)', fontsize=12)
    ax9.grid(True, alpha=0.3)
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        'Goldman (1970) Replication --- MoC blade + von Karman/Head BL\n'
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
    ax.plot(s_lo, Hi_lo, 'b-', lw=2, label='Lower')
    ax.plot(s_up, Hi_up, 'r-', lw=2, label='Upper')
    ax.axhspan(1.8, 2.4, alpha=0.08, color='red')
    ax.axhline(1.8, color='gray', ls='--', alpha=0.4)
    ax.axhline(2.4, color='gray', ls='--', alpha=0.4)
    ax.set_xlabel('Fraction of chord')
    ax.set_ylabel('$H_i$')
    ax.set_title('Fig. 4: Incompressible form factor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(1.0, 3.0)

    # Fig 8
    ax = axes[1, 0]
    ax.plot(nu_l_range_deg, Hi_max_lower, 'bo-', lw=2, markersize=5)
    ax.axhspan(1.8, 2.4, alpha=0.08, color='red')
    ax.axhline(1.8, color='gray', ls='--', alpha=0.4)
    ax.axhline(2.4, color='gray', ls='--', alpha=0.4)
    ax.set_xlabel('$\\nu_l$ (deg)')
    ax.set_ylabel('$H_{i,max}$ on lower surface')
    ax.set_title('Fig. 8: $H_{i,max}$ vs $\\nu_l$ ($\\nu_u=49°$)')
    ax.grid(True, alpha=0.3)

    # Fig 9
    ax = axes[1, 1]
    ax.plot(nu_u_range_deg, Hi_max_upper, 'ro-', lw=2, markersize=5)
    ax.axhspan(1.8, 2.4, alpha=0.08, color='red')
    ax.axhline(1.8, color='gray', ls='--', alpha=0.4)
    ax.axhline(2.4, color='gray', ls='--', alpha=0.4)
    ax.set_xlabel('$\\nu_u$ (deg)')
    ax.set_ylabel('$H_{i,max}$ on upper surface')
    ax.set_title('Fig. 9: $H_{i,max}$ vs $\\nu_u$ ($\\nu_l=34°$)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig('C:/Users/Martin/Active/FYPTurbine/quicktests/goldman_combined.png', dpi=150)
    print("\nSaved: goldman_combined.png")
    plt.close('all')


if __name__ == '__main__':
    main()
