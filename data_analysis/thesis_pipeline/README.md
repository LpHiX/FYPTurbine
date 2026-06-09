# Thesis data → LaTeX pipeline

**Rule:** no datum is ever typed into a `.tex` file. Every number lives here, is
exported to LaTeX macros, and is referenced by name in the prose. Change a number
in one place → rebuild → it updates everywhere (text, tables, figures).

## Files

| File | Role |
|---|---|
| `manual_values.toml` | Hand-entered constants (geometry, FEA limits, wood/rig dims, material props). The "random numbers" go here — NOT in the analysis scripts. |
| `build_values.py` | Assembles `results.json` = manual constants + computed analysis values. The one place numbers are gathered. |
| `export_latex.py` | `results.json` → `report/generated/values.tex` (one `\val...` macro per number, siunitx-formatted). |
| `registry.py` | The value model (val / unit / unc / fmt / desc) + JSON I/O. |
| `figstyle.py` | Plot helpers enforcing the house style: `plot_tuned` (solid), `plot_theory` (dashed), `plot_data` (markers+errorbars), `spline_trend`. |
| `../thesis.mplstyle` | Matplotlib style — body font/size on every figure (Knoll's consistency requirement). |
| `uncertainty_example.py` | Worked template for propagating uncertainty with `uncertainties`/`unumpy`. |

## Build

From the repo root, with the FYPTurbine venv active:

```
python build_thesis.py          # values + compile
python build_thesis.py --figs   # also regenerate figures
```

or step by step (from this folder): `python build_values.py && python export_latex.py`.

## LaTeX side (one-time setup)

In `report/ltx/preamble.tex`:

```latex
\usepackage{siunitx}
\DeclareSIUnit\rpm{rpm}          % siunitx has no built-in rpm
\input{generated/values.tex}    % pulls in all \val... macros
```

Then in prose:

```latex
The measured head coefficient is \valPumpPsiMeas{} (design \valPumpPsiDesign{}),
at a coupled speed of \valCoupledRpm{}~rpm.
```

`\valPumpPsiMeas` expands to `\qty{1.10 +- 0.05}{}` → renders as 1.10 ± 0.05.

## Adding a value

- **A given constant** (not computed): add a `[table]` to `manual_values.toml`.
- **A computed result**: add a line in `build_values.build_computed()` — prefer
  `reg.add_ufloat(key, ufloat_result, ...)` so the uncertainty travels with it.

### Should units be baked in?
Store the number and the unit **separately** (the `unit` field). The exporter
emits a single `\qty{value}{unit}` macro so number and unit are locked together
and formatted by siunitx — but you never hardcode the unit string in prose, and
you can restyle units globally from the preamble. Bare numbers (`unit = ""`) emit
`\num{...}`.

## Tables
Generate them too — don't type data tables. Have `build_values.py` (or a small
`export_tables.py`) write `report/generated/tables/*.tex` from the same registry /
a DataFrame (`df.to_latex(...)`), and `\input{}` them.
