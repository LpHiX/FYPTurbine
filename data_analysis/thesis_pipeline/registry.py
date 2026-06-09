"""Single source of truth for every number that appears in the thesis.

A `Value` carries a number, its unit (a siunitx unit string), an optional
1-sigma uncertainty, and a description. The `Registry` collects Values under
hierarchical snake_case keys and serialises to JSON, from which
`export_latex.py` emits one LaTeX macro per key.

Design rule: analysis code and manual constants both feed ONE registry, so the
LaTeX side has a single source — but they stay separate at the *source*
(computed values come from `build_values.build_computed`, given constants come
from `manual_values.toml`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Union

Number = Union[float, int, str]


def _to_native(x):
    """Coerce numpy scalars to plain Python so json.dump works."""
    if x is None:
        return None
    try:
        import numpy as np
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
    except ImportError:
        pass
    return x


@dataclass
class Value:
    val: Number                     # the number (or a string, for text labels)
    unit: str = ""                  # siunitx unit, e.g. r"\percent", r"\milli\meter"; "" = dimensionless
    unc: Optional[float] = None     # 1-sigma uncertainty, same unit, or None
    fmt: str = "g"                  # python format spec for the number, e.g. "g", ".2f", ".3g"
    desc: str = ""                  # what it is / where it came from


class Registry:
    def __init__(self):
        self._d: dict[str, Value] = {}

    def add(self, key, val, unit="", unc=None, fmt="g", desc=""):
        if key in self._d:
            raise KeyError(f"duplicate registry key: {key!r}")
        self._d[key] = Value(val=_to_native(val), unit=unit,
                             unc=_to_native(unc) if unc is not None else None,
                             fmt=fmt, desc=desc)
        return self

    def add_ufloat(self, key, u, unit="", fmt="g", desc=""):
        """Add an `uncertainties.ufloat` directly: val = nominal, unc = std dev."""
        self.add(key, u.nominal_value, unit=unit, unc=u.std_dev, fmt=fmt, desc=desc)

    def update(self, other: "Registry"):
        for k, v in other._d.items():
            if k in self._d:
                raise KeyError(f"duplicate registry key on merge: {k!r}")
            self._d[k] = v

    def __len__(self):
        return len(self._d)

    def to_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({k: asdict(v) for k, v in sorted(self._d.items())}, indent=2),
            encoding="utf-8")
        return path

    @classmethod
    def from_manual(cls, toml_path):
        """Load hand-entered constants. Each top-level [table] is one Value."""
        import tomllib
        reg = cls()
        data = tomllib.loads(Path(toml_path).read_text(encoding="utf-8"))
        for key, e in data.items():
            reg.add(key, e["val"], e.get("unit", ""), e.get("unc"),
                    e.get("fmt", "g"), e.get("desc", "manual constant"))
        return reg
