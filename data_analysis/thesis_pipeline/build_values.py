"""Assemble results.json from manual constants + computed analysis values.

This is the ONE place the thesis's numbers are gathered. Run it (FYPTurbine venv)
whenever the data, the model tuning, or a constant changes:

    python build_values.py        # -> report/generated/results.json

Then `export_latex.py` turns that JSON into LaTeX macros.
"""
from __future__ import annotations

from pathlib import Path

from registry import Registry

HERE = Path(__file__).resolve().parent
MANUAL = HERE / "manual_values.toml"
OUT = HERE.parent.parent / "report" / "generated" / "results.json"


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
    reg.add("turb_mw_design", 1.33, fmt=".2f", desc="design relative inlet Mach")
    reg.add("turb_mw_test", 1.58, fmt=".2f",
            desc="coupled-test relative inlet Mach (> starting limit -> unstarted)")
    reg.add("turb_start_rpm", 15300, unit="", desc="rotor-starting threshold speed [rpm]")
    reg.add("coupled_rpm", 5400, unit="", desc="coupled self-regulated speed [rpm]")
    reg.add("coupled_torque", 0.27, unit=r"\newton\meter", fmt=".2f",
            desc="measured coupled shaft torque")
    return reg


def main():
    reg = Registry.from_manual(MANUAL)   # hand-entered constants
    build_computed(reg)                  # analysis outputs
    path = reg.to_json(OUT)
    print(f"wrote {path}  ({len(reg)} values)")


if __name__ == "__main__":
    main()
