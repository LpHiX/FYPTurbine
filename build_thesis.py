"""One command to rebuild the thesis numbers, figures and PDF from source.

    python build_thesis.py            # values + LaTeX compile
    python build_thesis.py --figs     # also regenerate figures

Pipeline:  manual constants + analysis  ->  results.json  ->  values.tex
           figures (matched style)      ->  report/generated/figs/*.pdf
           latexmk                        ->  report/main.pdf

Run with the FYPTurbine venv so pandas/uncertainties/etc. are available.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPE = ROOT / "data_analysis" / "thesis_pipeline"
REPORT = ROOT / "report"
PY = sys.executable


def run(cmd, cwd=None):
    print(">", " ".join(map(str, cmd)), f"(cwd={cwd})" if cwd else "")
    subprocess.run(cmd, cwd=cwd, check=True)


def main(argv):
    run([PY, "build_values.py"], cwd=PIPE)      # -> results.json
    run([PY, "export_latex.py"], cwd=PIPE)      # -> values.tex
    run([PY, "export_tables.py"], cwd=PIPE)     # -> generated/tables/*.tex
    # if "--figs" in argv:
    run([PY, "thesis_figures.py"], cwd=ROOT / "data_analysis")
    # latexmk handles bib + reruns; swap for your build command if different.
    # run(["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"], cwd=REPORT)


if __name__ == "__main__":
    main(sys.argv[1:])
