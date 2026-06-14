def get_sample_pump():
    isp = 250
    thrust = 5000
    g = 9.81
    mdot = thrust / (isp * g) * 0.5 # for OF=1
    density_keroesene = 800
    q = mdot / density_keroesene
    p = 50e5
    H = p / (density_keroesene * g)
    nq = 20
    n = nq * H**0.75 / q**0.5  # metric n_q = n*Q^0.5 / H^0.75  ->  n [rpm]
    return thrust, mdot, nq, n, p, density_keroesene


def get_sample_turbine():
    """Worked 'example turbine' for the Blade Height and Partial Admission
    section of the theory chapter (2_theory.tex).

    A bare first-principles calc, deliberately decoupled from the full Weiss
    turbine model so the numbers in the prose are reproducible by hand. It feeds
    the \\todo{addvalue} blanks around eq:turbine_blade_height: spouting velocity,
    blade speed, u/c3, the (sub-mm) full-admission blade height, and the height
    once partial admission is introduced.

    Physics: single impulse stage, all expansion in the nozzle so the rotor sees
    constant static pressure p3 = p_exit. The gas is fully expanded to p_exit at
    station 3, so the static state there follows from the isentropic relations and
    the spouting velocity carries the whole enthalpy drop.

    NOTE on c3: the prose paragraph just above quotes c3 ~ 470 m/s for a pressure
    ratio of 5. This worked example uses PR = 10 (Martin's spec), which gives
    c3 ~ 538 m/s. The example is computed consistently at PR = 10 throughout
    (the 470 in the first-pass hand formula was carried over from the PR = 5
    sentence). Reconcile the prose PR if you want one number across both.
    """
    import math

    # ---- inputs ----
    cp = 1000.0        # J/kg/K, air
    gamma = 1.4
    R = 287.0          # J/kg/K, air
    T01 = 300.0        # K, inlet total temperature
    PR = 10.0          # stage total-to-static pressure ratio p01/p_exit
    N = 20000.0        # rpm
    d_mean = 0.100     # m, mean (pitch) diameter
    beta_deg = 15.0    # deg, nozzle flow angle from tangential
    P = 2000.0         # W, target shaft power
    p_exit = 1.0e5     # Pa, exit (and rotor) static pressure
    zeta = 0.15        # partial admission ratio

    beta = math.radians(beta_deg)

    # ---- velocity triangle ----
    # spouting velocity: ideal isentropic expansion over the full pressure ratio
    c3 = math.sqrt(2.0 * cp * T01 * (1.0 - (1.0 / PR) ** ((gamma - 1.0) / gamma)))
    u = N * 2.0 * math.pi / 60.0 * (d_mean / 2.0)   # mean blade speed
    u_c3 = u / c3                                    # blade-jet speed ratio
    c3u = c3 * math.cos(beta)                        # swirl (tangential) component
    c3m = c3 * math.sin(beta)                        # meridional component

    # Euler work for a symmetric impulse blade: dh = 2 u (c3u - u)
    deltah_useful = 2.0 * u * (c3u - u)
    mdot = P / deltah_useful                         # mass flow to make target power

    # ---- station-3 static state (full nozzle expansion, p3 = p_exit) ----
    T3 = T01 - c3 ** 2 / (2.0 * cp)
    rho3 = p_exit / (R * T3)

    # ---- blade height ----
    H_full = mdot / (rho3 * c3m * d_mean)            # full admission, eq:..full_admission
    H_partial = mdot / (zeta * rho3 * c3m * d_mean)  # partial admission, eq:turbine_blade_height

    return {
        "c3": c3,
        "u": u,
        "u_c3": u_c3,
        "c3m": c3m,
        "deltah_useful": deltah_useful,
        "mdot": mdot,
        "T3": T3,
        "rho3": rho3,
        "H_full": H_full,
        "zeta": zeta,
        "H_partial": H_partial,
        "PR": PR,
        "P": P,
        "d_mean": d_mean,
        "beta_deg": beta_deg,
    }


def _diameter_stub():
    diameter = 0