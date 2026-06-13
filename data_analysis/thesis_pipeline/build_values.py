"""Assemble results.json from manual constants + computed analysis values.

This is the ONE place the thesis's numbers are gathered. Run it (FYPTurbine venv)
whenever the data, the model tuning, or a constant changes:

    python build_values.py        # -> report/generated/results.json

Then `export_latex.py` turns that JSON into LaTeX macros.

Sources, in order:
  manual_values.toml   given constants the code can't produce (as-built CAD,
                       FEA results, datasheet figures, fitted literature consts)
  build_design()       EVERYTHING the design models compute — the canonical
                       sized BarskePump and Turbine objects from
                       thesis_figures.pump() / thesis_figures.turbine_design()
  build_computed()     measured / analysis-derived results
"""
from __future__ import annotations

import sys
from pathlib import Path

from registry import Registry

HERE = Path(__file__).resolve().parent
MANUAL = HERE / "manual_values.toml"
OUT = HERE.parent.parent / "report" / "generated" / "results.json"

sys.path.insert(0, str(HERE.parent))  # data_analysis (thesis_figures handles the rest)


def build_design(reg: Registry) -> Registry:
    """Register every design-point number straight from the design models.

    Turbine: the SAME canonical instance thesis_figures.turbine_design() uses.
    Pump: BarskePump sized exactly as barskesizer.ipynb (0.3 kg/s / 20 bar /
    20k rpm, AS-DESIGNED Karassik seal f=0.07) — NOT thesis_figures.pump(),
    which carries the post-hoc fitted seal friction (f=0.014) and therefore
    predicts a higher efficiency than was designed to. Geometry is identical
    either way (seal friction only enters the loss budget); the performance
    numbers here are the genuine design predictions. A model change propagates
    to figures, prose macros and tables in one rebuild. Nothing is typed by hand.
    """
    import thesis_figures as tf
    from prop_components.barskepump import BarskePump
    from mech_components.bearing import Bearing
    from mech_components.mechanicalseal import MechanicalSeal

    # ---------------- pump (BarskePump sizing model, as-designed) ----------------
    tb = Bearing(d=10, D=22, series=619, visc=1.0, C_0_kN=1.27, submerged=False)
    bb = Bearing(d=10, D=22, series=619, visc=1.0, C_0_kN=1.27, submerged=True)
    seal = MechanicalSeal(OD_mm=19.5, ID_mm=15, BD_mm=14, F_sp=100)  # default f=0.07
    p = BarskePump(0.3, 20e5, 1000, 1.002e-6, 20000, tb, bb, seal)
    mm = 1000.0
    reg.add("pump_rpm_design", p.RPM, unit=r"\rpm", fmt=".0f",
            desc="design shaft speed")
    reg.add("pump_mdot_design", p.mdot_desired, unit=r"\kilo\gram\per\second", fmt=".1f",
            desc="design mass flow rate")
    reg.add("pump_q_design", p.Q_desired * 1000, unit=r"\liter\per\second", fmt=".1f",
            desc="design volume flow rate")
    reg.add("pump_pressure_design", p.p_actual / 1e5, unit=r"\bar", fmt=".0f",
            desc="design pressure rise")
    reg.add("pump_head_design", p.H_actual, unit=r"\meter", fmt=".0f",
            desc="design head rise")
    reg.add("pump_nq_design", p.n_q, fmt=".1f",
            desc="metric specific speed n_q = n sqrt(Q)/H^0.75 (rpm, m^3/s, m)")
    reg.add("pump_psi_design", p.head_coeff, fmt=".3f",
            desc="design head coefficient 2gH/u2^2 (Gulich)")
    reg.add("pump_eta_design", p.efficiency * 100, unit=r"\percent", fmt=".1f",
            desc="design-point overall efficiency (Barske loss model, as-designed Karassik seal f=0.07)")
    reg.add("pump_power_design", p.required_power, unit=r"\watt", fmt=".0f",
            desc="design required shaft power")
    reg.add("pump_torque_design", p.torque, unit=r"\newton\meter", fmt=".2f",
            desc="design shaft torque")
    reg.add("pump_pfric_design", p.power_friction, unit=r"\watt", fmt=".0f",
            desc="Barske disk/churning friction estimate at design")
    reg.add("pump_pmech_design", p.mechanical_loss, unit=r"\watt", fmt=".0f",
            desc="bearing + seal mechanical loss estimate at design")
    reg.add("pump_npshr_design", p.NPSHR, unit=r"\meter", fmt=".1f",
            desc="NPSHr estimate at design (suction specific speed n_s=150, Lobanoff)")
    # geometry
    reg.add("pump_d0", p.d_0 * mm, unit=r"\milli\meter", fmt=".2f",
            desc="suction eye diameter (sized for 1 m/s inlet velocity)")
    reg.add("pump_d1", p.d_1 * mm, unit=r"\milli\meter", fmt=".1f",
            desc="impeller inlet (blade leading-edge) diameter, 1.1 d0")
    reg.add("pump_b1", p.b_1 * mm, unit=r"\milli\meter", fmt=".1f",
            desc="blade height at impeller inlet")
    reg.add("pump_d2", p.d_2 * mm, unit=r"\milli\meter", fmt=".1f",
            desc="impeller tip diameter")
    reg.add("pump_b2", p.b_2 * mm, unit=r"\milli\meter", fmt=".1f",
            desc="blade height at impeller exit")
    reg.add("pump_blades", p.blade_number, fmt=".0f", desc="blade count")
    reg.add("pump_sax", p.s_ax * mm, unit=r"\milli\meter", fmt=".2f",
            desc="impeller-casing axial gap s_ax")
    reg.add("pump_bc", p.B * mm, unit=r"\milli\meter", fmt=".2f",
            desc="annular casing axial width b_c (= b2 + 2 s_ax)")
    reg.add("pump_hc", p.H * mm, unit=r"\milli\meter", fmt=".2f",
            desc="annular casing radial height h_c")
    reg.add("pump_d3_sized", p.d_3 * mm, unit=r"\milli\meter", fmt=".2f",
            desc="sizing-model diffuser throat (as-built TRUE throat = pump_throat 3.0)")
    reg.add("pump_d4_sized", p.d_4 * mm, unit=r"\milli\meter", fmt=".2f",
            desc="sizing-model diffuser exit diameter (2 d3)")
    # velocities / blade speeds
    reg.add("pump_u1_design", p.u_1, unit=r"\meter\per\second", fmt=".1f",
            desc="blade speed at impeller inlet, design")
    reg.add("pump_u2_design", p.u_2, unit=r"\meter\per\second", fmt=".1f",
            desc="blade tip speed at design")
    # structural (sizing-model estimates, not FEA)
    reg.add("pump_tip_force", p.tip_force, unit=r"\newton", fmt=".2f",
            desc="per-blade tip force from design torque")
    reg.add("pump_blade_stress", p.blade_stress / 1e6, unit=r"\mega\pascal", fmt=".2f",
            desc="centrifugal blade root stress estimate (sizing model, not FEA)")

    # ---------------- turbine (loss-consistent inert-gas sizing) ----------------
    t = tf.turbine_design()
    reg.add("turb_rpm_design", t.RPM, unit=r"\rpm", fmt=".0f",
            desc="turbine design shaft speed (common shaft)")
    reg.add("turb_power_design", t.P / 1000, unit=r"\kilo\watt", fmt=".1f",
            desc="turbine design shaft power")
    reg.add("turb_dmean", t.d_mean * mm, unit=r"\milli\meter", fmt=".0f",
            desc="turbine mean (pitch) diameter")
    reg.add("turb_blade_height", t.Height * mm, unit=r"\milli\meter", fmt=".1f",
            desc="rotor blade height (from continuity at DOA)")
    reg.add("turb_doa", t.doa, fmt=".2f", desc="degree of admission")
    reg.add("turb_beta3_design", tf.TRB["BETA_DEG"], unit=r"\degree", fmt=".0f",
            desc="rotor blade angle from tangential (design relative flow angle)")
    reg.add("turb_nozzles", t.nozzles, fmt=".0f", desc="number of nozzles")
    reg.add("turb_mdot_design", t.mdot, unit=r"\kilo\gram\per\second", fmt=".4f",
            desc="design air mass flow rate")
    reg.add("turb_t01_design", tf.TRB["T01_DES"], unit=r"\kelvin", fmt=".0f",
            desc="design turbine inlet total temperature (air)")
    reg.add("turb_pe_design", t.p_e / 1e5, unit=r"\bar", fmt=".1f",
            desc="design exit static pressure")
    reg.add("turb_p01_design", t.p01 / 1e5, unit=r"\bar", fmt=".2f",
            desc="design inlet total pressure (incl. nozzle total-pressure loss)")
    reg.add("turb_p03_design", t.p03 / 1e5, unit=r"\bar", fmt=".2f",
            desc="nozzle-exit total pressure (isentropic expansion requirement)")
    reg.add("turb_p3_design", t.p3 / 1e5, unit=r"\bar", fmt=".2f",
            desc="rotor-inlet static pressure at design")
    reg.add("turb_t3_design", t.T3, unit=r"\kelvin", fmt=".0f",
            desc="rotor-inlet static temperature at design")
    # velocity triangle
    reg.add("turb_u_design", t.u, unit=r"\meter\per\second", fmt=".1f",
            desc="mean blade speed at design")
    reg.add("turb_c3_design", t.c3, unit=r"\meter\per\second", fmt=".0f",
            desc="design nozzle-exit absolute velocity")
    reg.add("turb_c3u_design", t.c3u, unit=r"\meter\per\second", fmt=".0f",
            desc="design nozzle-exit swirl component")
    reg.add("turb_c3m_design", t.c3m, unit=r"\meter\per\second", fmt=".0f",
            desc="design nozzle-exit meridional component")
    reg.add("turb_c4_design", t.c4, unit=r"\meter\per\second", fmt=".0f",
            desc="design rotor-exit absolute velocity")
    reg.add("turb_m3_design", t.M3, fmt=".2f",
            desc="design nozzle-exit absolute Mach number")
    reg.add("turb_mw_design", t.Mw3, fmt=".2f",
            desc="design rotor-inlet relative Mach M_w3 (Goldman startable: < 1.408)")
    reg.add("turb_deltab_design", t.deltaB_deg, unit=r"\degree", fmt=".0f",
            desc="rotor relative turning angle (symmetric impulse blade)")
    reg.add("turb_u_c3_design", t.blade_jet_speed_ratio, fmt=".3f",
            desc="blade-jet speed ratio u/c3 at design")
    reg.add("turb_u_cs_design", t.u / (2 * t.deltah_s) ** 0.5, fmt=".3f",
            desc="blade-speed ratio u/c_s (isentropic spouting velocity)")
    # nozzle geometry
    reg.add("turb_athroat", t.A_throat * 1e6, unit=r"\square\milli\meter", fmt=".1f",
            desc="total geometric nozzle throat area")
    reg.add("turb_a3_design", t.A3 * 1e6, unit=r"\square\milli\meter", fmt=".1f",
            desc="total nozzle exit area")
    reg.add("turb_area_ratio", t.eps, fmt=".2f",
            desc="nozzle expansion area ratio A3/A_th")
    reg.add("turb_nozzle_throat_len", t.nozzle_throat_length * mm,
            unit=r"\milli\meter", fmt=".2f", desc="per-nozzle throat width")
    reg.add("turb_nozzle_exit_len", t.nozzle_exit_length * mm,
            unit=r"\milli\meter", fmt=".2f", desc="per-nozzle exit width")
    # losses / efficiency (Weiss model)
    reg.add("turb_kn_design", t.phi_n, fmt=".3f",
            desc="nozzle velocity coefficient k_N (Weiss, fn of M3)")
    reg.add("turb_kr_design", t.phi_r, fmt=".3f",
            desc="rotor velocity coefficient k_R (Weiss, fn of turning + Mw3)")
    reg.add("turb_pw_design", t.p_v, unit=r"\watt", fmt=".0f",
            desc="ventilation (partial-admission windage) power at design")
    reg.add("turb_eta_h_design", t.eta_h, fmt=".2f",
            desc="hydraulic (diagram) efficiency at design")
    reg.add("turb_eta_ts_design", t.eff_real, fmt=".2f",
            desc="design total-to-static efficiency (Weiss loss model, net of ventilation)")
    # gas properties
    reg.add("air_r", tf.TRB["R_GAS"], unit=r"\joule\per\kilo\gram\per\kelvin", fmt=".0f",
            desc="specific gas constant, air")
    reg.add("air_gam", tf.TRB["GAM"], fmt=".1f", desc="heat capacity ratio, air")
    return reg


def build_computed(reg: Registry) -> Registry:
    """Add numbers that come OUT of the analysis pipeline.

    TODO: import your real analysis (epump_io, barskepump, turbine_analysis …)
    and replace the literals below with the computed results — ideally as
    `uncertainties.ufloat`s via `reg.add_ufloat(...)` so the error bar travels
    with the number. The values here are seeded from the current findings so the
    pipeline runs end-to-end today; swap them for live computations one by one.

    Example of the intended pattern (uncomment once wired):

        import epump_io as ep
        from uncertainties import ufloat
        runs = ep.load_runs(ep.LOGDIR)
        psi_peak = compute_psi_peak(runs)          # returns a ufloat
        reg.add_ufloat("pump_psi_meas", psi_peak, desc="measured peak head coeff")
    """
    # Introduction numbers

    from quicknumbers.intro_ns import get_sample_pump
    samplepump_thrust, samplepump_mdot, samplepump_nq, samplepump_n, samplepump_p, samplepump_rho = get_sample_pump()
    reg.add("samplepump_thrust", samplepump_thrust/1000, unit=r"\kilo\newton", fmt=".0f", desc="sample pump for introduction thrust")
    reg.add("samplepump_mdot", samplepump_mdot, unit=r"\kilo\gram\per\second", fmt=".1f", desc="sample pump for introduction mdot")
    reg.add("samplepump_nq", samplepump_nq, unit="", fmt=".0f", desc="sample pump for introduction nq (metric: rpm, m^3/s, m — quoted bare, convention stated in prose)")
    reg.add("samplepump_n", samplepump_n, unit=r"\rpm", fmt=".0f", desc="sample pump for introduction shaft speed")
    reg.add("samplepump_p", samplepump_p / 1e5, unit=r"\bar", fmt=".0f", desc="sample pump for introduction discharge pressure")
    reg.add("samplepump_rho", samplepump_rho, unit=r"\kilo\gram\per\cubic\metre", fmt=".0f", desc="sample pump for introduction rho")


    # --- pump (seed values; replace with computed ufloats) ---
    reg.add("pump_psi_meas", 1.10, unc=0.05, fmt=".2f",
            desc="measured peak head coefficient")
    reg.add("pump_eta_peak", 19.0, unit=r"\percent", unc=2.0, fmt=".0f",
            desc="measured peak overall efficiency")
    reg.add("pump_eta_extrap", 27, unit=r"\percent",
            desc="Reynolds-extrapolated efficiency at 20k rpm (26-29 range)")
    reg.add("pump_churn_mult", 2.5, fmt=".1f",
            desc="disk/churning friction multiple over textbook")
    reg.add("pump_seal_f", 0.014, fmt=".3f",
            desc="fitted seal friction coefficient (water-film)")

    # --- turbine / coupled (seed values) ---
    reg.add("turb_mw_test", 1.58, fmt=".2f",
            desc="coupled-test relative inlet Mach (> starting limit -> unstarted)")
    reg.add("turb_start_rpm", 15300, unit="", desc="rotor-starting threshold speed [rpm]")
    reg.add("coupled_rpm", 5400, unit="", desc="coupled self-regulated speed [rpm]")
    reg.add("coupled_torque", 0.27, unit=r"\newton\meter", fmt=".2f",
            desc="measured coupled shaft torque")
    return reg


def main():
    reg = Registry.from_manual(MANUAL)   # hand-entered constants
    build_design(reg)                    # everything the design models compute
    build_computed(reg)                  # analysis outputs
    path = reg.to_json(OUT)
    print(f"wrote {path}  ({len(reg)} values)")


if __name__ == "__main__":
    main()
