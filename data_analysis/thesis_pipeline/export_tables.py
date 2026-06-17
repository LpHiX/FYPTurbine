r"""results.json -> report/generated/tables/*.tex  (booktabs design tables).

Each table is a spec below: an ordered list of rows mapping a registry key to
its print symbol (nomenclature per the thesis register). The value cell uses
the SAME \val* macro that prose uses (from export_latex.macro_name), so number
formatting, units and uncertainties have exactly one source — regenerating the
registry updates prose and tables identically.

Row forms:
    {"key": "pump_d2", "sym": r"$d_2$"}                    # desc from registry
    {"key": "pump_d1", "sym": r"$d_1$", "desc": "..."}     # print-quality desc override
    {"sym": "--", "desc": "working fluid", "val": "water"} # literal row, no registry key

Only the inner tabular is emitted — wrap it in your own \begin{table} with your
caption/label so placement and caption prose stay in your hands, e.g.:

    \begin{table}[t]
      \centering
      \caption{Pump design parameters.}
      \label{tab:pump_design}
      \input{generated/tables/tab_pump_design}
    \end{table}

    python export_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

from export_latex import macro_name

HERE = Path(__file__).resolve().parent
JSON = HERE.parent.parent / "report" / "generated" / "results.json"
OUTDIR = HERE.parent.parent / "report" / "generated" / "tables"


TABLES = {
    # System-level design point — orients the reader at the top of Ch3.
    "tab_design_point": [
        {"key": "pump_rpm_design", "sym": r"$n$", "desc": "design shaft speed"},
        {"key": "pump_q_design", "sym": r"$Q$", "desc": "pump design volume flow rate"},
        # {"key": "pump_head_design", "sym": r"$H$", "desc": "pump design head rise"},
        {"key": "pump_pressure_design", "sym": r"$\Delta p$", "desc": "pump design pressure rise"},
        {"key": "pump_nq_design", "sym": r"$n_q$", "desc": "pump specific speed (rpm, \\unit{\\cubic\\meter\\per\\second}, \\unit{\\meter})"},
        {"key": "turb_power_design", "sym": r"$P$", "desc": "turbine design shaft power"},
        {"key": "turb_p01_design", "sym": r"$p_{01}$", "desc": "turbine design inlet total pressure"},
        {"sym": "--", "desc": "pump working fluid", "val": "water"},
        {"sym": "--", "desc": "turbine working fluid", "val": "compressed air"},
    ],
    # Pump design summary — end of the pump design section.
    "tab_pump_design": [
        {"key": "pump_d1", "sym": r"$d_1$", "desc": "impeller inlet (blade leading-edge) diameter"},
        {"key": "pump_d2", "sym": r"$d_2$", "desc": "impeller tip diameter"},
        {"key": "pump_b1", "sym": r"$b_1$", "desc": "blade height at impeller inlet"},
        {"key": "pump_b2", "sym": r"$b_2$", "desc": "blade height at impeller exit"},
        {"key": "pump_blades", "sym": r"$z$", "desc": "blade count"},
        {"key": "pump_throat", "sym": r"$d_3$", "desc": "diffuser throat diameter (as built)"},
        {"key": "pump_d4_asbuilt", "sym": r"$d_4$", "desc": "diffuser exit diameter (as built)"},
        {"key": "pump_psi_design", "sym": r"$\psi$", "desc": "design head coefficient"},
        {"key": "pump_nq_design", "sym": r"$n_q$", "desc": "specific speed (rpm, \\unit{\\cubic\\meter\\per\\second}, \\unit{\\meter})"},
    ],
    # Turbine design summary — end of the turbine design section.
    "tab_turbine_design": [
        {"key": "turb_power_design", "sym": r"$P$", "desc": "design shaft power"},
        {"key": "turb_p01_design", "sym": r"$p_{01}$", "desc": "design inlet total pressure"},
        {"key": "turb_t01_design", "sym": r"$T_{01}$", "desc": "design inlet total temperature"},
        {"key": "turb_mdot_design", "sym": r"$\dot{m}$", "desc": "design air mass flow rate"},
        {"key": "turb_dmean", "sym": r"$d_m$", "desc": "mean (pitch) diameter"},
        {"key": "turb_blade_height", "sym": r"$b$", "desc": "rotor blade height"},
        {"key": "turb_doa", "sym": r"$\zeta$", "desc": "degree of admission"},
        {"key": "turb_beta3_design", "sym": r"$\beta_3$", "desc": "rotor blade angle (from tangential)"},
        {"key": "turb_nozzles", "sym": r"$z_N$", "desc": "number of nozzles"},
        {"key": "turb_athroat", "sym": r"$A_\mathrm{th}$", "desc": "total nozzle throat area"},
        {"key": "turb_c3_design", "sym": r"$c_3$", "desc": "design nozzle-exit velocity"},
        {"key": "turb_m3_design", "sym": r"$M_3$", "desc": "design nozzle-exit Mach number"},
        {"key": "turb_mw_design", "sym": r"$M_{w3}$", "desc": "design rotor-inlet relative Mach number"},
        {"key": "turb_eta_ts_design", "sym": r"$\eta_\mathrm{ts}$", "desc": "design total-to-static efficiency"},
    ],
}


# Predicted-vs-measured comparison (the quantitative anchor for Ch5).
# Each row: registry keys for the model-side and measured-side values; the
# percentage difference is computed here at build time from the registry, so
# it can never drift from the quoted numbers. "note" lands in the last column.
COMPARE_TABLES = {
    "tab_model_vs_meas": [
        {"qty": r"head coefficient $\psi$",
         "model": "pump_psi_design", "meas": "pump_psi_meas",
         "note": "Barske vs measured"},
        {"qty": r"overall efficiency $\eta$ at \valPumpRpmDesign",
         "model": "pump_eta_design", "meas": "pump_eta_extrap",
         "note": "Extrapolated"},
        {"qty": "shutoff churning power",
         "model": None, "meas": None, "model_txt": "Barske disk correlation",
         "meas_txt": r"\valPumpChurnMult$\times$", "delta_txt": r"$+150\,\%$",
         "note": "Measured at shutoff"},
        {"qty": "coupled shaft torque",
         "model": "turb_torque_started_test", "meas": "coupled_torque",
         "note": "Unstarted"},
    ],
}


def emit_compare_table(name: str, rows: list[dict], data: dict) -> str:
    lines = [
        f"% AUTO-GENERATED by export_tables.py ({name}) — DO NOT EDIT BY HAND.",
        "% Regenerate:  python build_values.py && python export_latex.py && python export_tables.py",
        r"\begin{tabular}{@{}lllrl@{}}",
        r"\toprule",
        r"Quantity & Predicted & Measured & $\Delta$ & Note \\",
        r"\midrule",
    ]
    for row in rows:
        if row.get("model") is not None:
            km, ks = row["model"], row["meas"]
            for k in (km, ks):
                if k not in data:
                    raise KeyError(f"{name}: registry key {k!r} missing")
            vm, vs = data[km]["val"], data[ks]["val"]
            delta = rf"${(vs - vm) / vm * 100:+.0f}\,\%$"
            model_cell = rf"\{macro_name(km)}"
            meas_cell = rf"\{macro_name(ks)}"
        else:
            model_cell, meas_cell = row["model_txt"], row["meas_txt"]
            delta = row["delta_txt"]
        lines.append(rf"{row['qty']} & {model_cell} & {meas_cell} & {delta} & "
                     rf"{row.get('note', '')} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def emit_table(name: str, rows: list[dict], data: dict) -> str:
    lines = [
        f"% AUTO-GENERATED by export_tables.py ({name}) — DO NOT EDIT BY HAND.",
        "% Regenerate:  python build_values.py && python export_latex.py && python export_tables.py",
        r"\begin{tabular}{@{}lll@{}}",
        r"\toprule",
        r"Symbol & Parameter & Value \\",
        r"\midrule",
    ]
    for row in rows:
        key = row.get("key")
        if key is not None:
            if key not in data:
                raise KeyError(f"{name}: registry key {key!r} not in {JSON.name} "
                               "— run build_values.py, or fix the spec")
            desc = row.get("desc") or data[key].get("desc", "")
            value = rf"\{macro_name(key)}"
        else:
            desc = row["desc"]
            value = row["val"]
        lines.append(rf"{row['sym']} & {desc} & {value} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    data = json.loads(JSON.read_text(encoding="utf-8"))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for name, rows in TABLES.items():
        path = OUTDIR / f"{name}.tex"
        path.write_text(emit_table(name, rows, data), encoding="utf-8")
        print(f"wrote {path}  ({len(rows)} rows)")
    for name, rows in COMPARE_TABLES.items():
        path = OUTDIR / f"{name}.tex"
        path.write_text(emit_compare_table(name, rows, data), encoding="utf-8")
        print(f"wrote {path}  ({len(rows)} rows, compare)")


if __name__ == "__main__":
    main()
