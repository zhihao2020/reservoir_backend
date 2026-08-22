"""Load a Peng-Robinson mixture from YAML.

Used by the EXAMPLE C1–C7+/CO2 deck. The same schema can point at a
future real card; this module does not invent Tc, Pc, or ω. Missing
file or required fields raise. Not FIM, not GEM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from reservoir_backend.eos.peng_robinson import EosMixture

DEFAULT_EXAMPLE_FLUID_YAML = Path(__file__).resolve().parent / "fluids" / "example_c1_c7plus_co2.yaml"

_REQUIRED_COMP = ("name", "Tc", "Pc", "omega")
_MISSING_FILE = (
    "EXAMPLE EOS YAML not found: {path}. "
    "A fluid deck with Tc, Pc, and omega is required; "
    "refusing to invent GEM/Jiyang criticals."
)
_MISSING_FIELD = (
    "EXAMPLE EOS YAML {path}: {where} missing required field {field}. "
    "Refusing to invent criticals."
)


def resolve_fluid_yaml(path: str | Path) -> Path:
    """Resolve a deck path (absolute, cwd, or ``eos/fluids/<name>``)."""
    raw = Path(path)
    if raw.is_file():
        return raw.resolve()
    packaged = DEFAULT_EXAMPLE_FLUID_YAML.parent / raw.name
    if packaged.is_file():
        return packaged
    raise FileNotFoundError(_MISSING_FILE.format(path=path))


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(_MISSING_FILE.format(path=path))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"EOS YAML {path} must be a mapping, not a generated card")
    return data


def load_eos_mixture_yaml(path: str | Path) -> EosMixture:
    """Build ``EosMixture`` from YAML. Errors if the file or Tc/Pc/ω is missing."""
    deck = Path(path)
    data = _read_mapping(deck)
    marker = data.get("marker")
    if marker is None or str(marker).strip() == "":
        raise ValueError(_MISSING_FIELD.format(path=deck, where="deck", field="marker"))
    marker_s = " ".join(str(marker).split())
    if "EXAMPLE" not in marker_s:
        raise ValueError(
            f"EOS YAML {deck}: marker must identify EXAMPLE parameters; "
            "refusing to treat an unlabeled deck as a GEM card."
        )
    comps = data.get("components")
    if not comps:
        raise ValueError(_MISSING_FIELD.format(path=deck, where="deck", field="components"))
    names: list[str] = []
    tc: list[float] = []
    pc: list[float] = []
    omega: list[float] = []
    mw: list[float] = []
    have_mw = True
    for i, row in enumerate(comps):
        if not isinstance(row, dict):
            raise ValueError(f"EOS YAML {deck}: components[{i}] must be a mapping")
        label = row.get("name") or f"components[{i}]"
        for field in _REQUIRED_COMP:
            if field not in row or row[field] is None:
                raise ValueError(_MISSING_FIELD.format(path=deck, where=f"component {label!r}", field=field))
        names.append(str(row["name"]))
        tc.append(float(row["Tc"]))
        pc.append(float(row["Pc"]))
        omega.append(float(row["omega"]))
        if row.get("Mw_g_mol") is None:
            have_mw = False
        else:
            mw.append(float(row["Mw_g_mol"]) / 1000.0)
    if "kij_pairs" not in data:
        raise ValueError(_MISSING_FIELD.format(path=deck, where="deck", field="kij_pairs"))
    nc = len(names)
    index = {n: i for i, n in enumerate(names)}
    kij = np.zeros((nc, nc), dtype=float)
    for pair in data["kij_pairs"] or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 3:
            raise ValueError(f"EOS YAML {deck}: each kij_pairs entry must be [name_i, name_j, k]")
        a, b, k = pair
        if a not in index or b not in index:
            raise ValueError(f"EOS YAML {deck}: kij_pairs names {a!r}, {b!r} must be in components")
        i, j = index[str(a)], index[str(b)]
        kij[i, j] = float(k)
        kij[j, i] = float(k)
    return EosMixture(
        names=tuple(names),
        Tc=np.array(tc, dtype=float),
        Pc=np.array(pc, dtype=float),
        omega=np.array(omega, dtype=float),
        kij=kij,
        Mw=np.array(mw, dtype=float) if have_mw and len(mw) == nc else None,
        marker=marker_s,
    )


def load_feed_z_yaml(path: str | Path) -> np.ndarray:
    """Load ``feed_z`` from the same deck. Errors if any component fraction is missing."""
    deck = Path(path)
    data = _read_mapping(deck)
    mix = load_eos_mixture_yaml(deck)
    feed = data.get("feed_z")
    if not isinstance(feed, dict) or not feed:
        raise ValueError(_MISSING_FIELD.format(path=deck, where="deck", field="feed_z"))
    z = []
    for name in mix.names:
        if name not in feed or feed[name] is None:
            raise ValueError(_MISSING_FIELD.format(path=deck, where="feed_z", field=name))
        z.append(float(feed[name]))
    arr = np.array(z, dtype=float)
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError(f"EOS YAML {deck}: feed_z must sum to a positive value")
    return arr / total
