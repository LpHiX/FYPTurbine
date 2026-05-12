# FYP Report LaTeX Instructions

This directory contains the LaTeX source code for Martin's Final Year Project (FYP) report on the small-scale Barske-type turbopump.

## Project Structure
- `main.tex`: The root document. Do NOT write extensive text here.
- `ltx/`: Contains all modular `.tex` files (e.g., `0_abstract.tex`, `1_introduction.tex`, `preamble.tex`).
- `figs/`: Contains all images.
- `references.bib`: The BibTeX bibliography file exported from Zotero.

## Image Management
Images should be placed in their respective subdirectories within the `figs/` folder to prevent clutter:
- `figs/ch1_intro/`
- `figs/ch2_background/`
- `figs/ch3_methodology/`
- `figs/ch4_results/`
- `figs/ch5_discussion/`
- `figs/app/`

You only need to specify the filename in LaTeX because `\graphicspath` is configured in `preamble.tex`:
`\includegraphics[width=0.8\textwidth]{my_image.png}` (No need to write `figs/ch1_intro/my_image.png`).

## LaTeX Best Practices
- **Units:** Always use the `siunitx` package for units.
  - Correct: `\qty{20000}{\rpm}` or `\unit{\kg\per\cubic\m}`
  - Incorrect: `$20,000 \mathrm{RPM}$` or `$\mathrm{kg/m^3}$`
- **Cross-referencing:** Always use the `cleveref` package.
  - Correct: `\cref{fig:my_figure}` (Outputs: "fig. 1" or "equation 2") or `\Cref{tab:my_table}` at the start of a sentence.
  - Incorrect: `Figure \ref{fig:my_figure}`
- **Math/Vectors:** Use `\bm{}` for bold math (vectors, matrices) instead of `\mathbf{}`.
- **Formatting:** Avoid hardcoding spaces (`\\` or `\vspace` or `\noindent`) inside paragraph text. Let LaTeX handle document flow. (Note: `\vspace` is acceptable in custom title pages).

## Custom Macros
Custom macros are defined in `ltx/preamble.tex` to maintain consistency for complex acronyms:
- `\NPSHa{}` -> Outputs formatted $\text{NPSH}_\text{a}$
- `\NPSHr{}` -> Outputs formatted $\text{NPSH}_\text{r}$
