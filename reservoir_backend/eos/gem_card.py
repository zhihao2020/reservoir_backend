"""Optional CMG-GEM-style text card → YAML EOS deck mapping.

Reads ``*EOS`` / ``*COMP`` (and companion ``*TCRIT``, ``*PCRIT``,
``*ACENTRIC`` / ``*AC``, ``*MW``, ``*BIN`` when present) and returns the
same mapping ``mixture_from_deck_dict`` / ``load_eos_mixture_yaml`` consume.

YAML remains the primary fluid path. This reader does not invent Tc, Pc,
or ω, does not import ``references/``, and does not restore cmg_harness.
Not FIM. Not field-validated. EXAMPLE cards only.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from reservoir_backend.eos.load import mixture_from_deck_dict
from reservoir_backend.eos.peng_robinson import EosMixture

DEFAULT_EXAMPLE_GEM_CARD = Path(__file__).resolve().parent / "fluids" / "example_c1_co2.gem"
EXAMPLE_C1_C7PLUS_CO2_GEM = Path(__file__).resolve().parent / "fluids" / "example_c1_c7plus_co2.gem"

_GEM_SOURCE = "EXAMPLE GEM card"
_MISSING_FILE = (
    "EXAMPLE GEM card not found: {path}. "
    "A fluid deck with Tc, Pc, and omega is required; "
    "refusing to invent GEM/Jiyang criticals."
)
_MISSING_KW = (
    "EXAMPLE GEM card {path}: missing required keyword {field}. "
    "Refusing to invent criticals."
)
_MISSING_COUNT = (
    "EXAMPLE GEM card {path}: {field} has {got} values for {need} components. "
    "Refusing to invent criticals."
)

_SECTION = {
    "EOS",
    "MODEL",
    "NC",
    "NCOMP",
    "COMP",
    "COMPNAME",
    "COMPNAMES",
    "TCRIT",
    "PCRIT",
    "ACENTRIC",
    "AC",
    "ACF",
    "OMEGA",
    "MW",
    "MOLWT",
    "MOL_WEIGHT",
    "BIN",
    "BIJ",
    "KIJ",
    "INUNIT",
    "DESC",
    "DESCRIPTION",
}
_FLAGS = {
    "PR",
    "PR78",
    "PENG_ROBINSON",
    "PENGROBINSON",
    "PENG",
    "SI",
    "FIELD",
    "METRIC",
    "LAB",
    "ENGLISH",
    "SRK",
    "SRK78",
    "RK",
}
_PR_MODELS = {"PR", "PR78", "PENG_ROBINSON", "PENGROBINSON", "PENG"}
_KPA_TO_PA = 1.0e3
_PSIA_TO_PA = 6894.757293168
_RANKINE_TO_K = 5.0 / 9.0


def resolve_gem_deck(path: str | Path) -> Path:
    """Resolve a GEM text-card path (absolute, cwd, or ``eos/fluids/<name>``)."""
    raw = Path(path)
    if raw.is_file():
        return raw.resolve()
    packaged = DEFAULT_EXAMPLE_GEM_CARD.parent / raw.name
    if packaged.is_file():
        return packaged
    raise FileNotFoundError(_MISSING_FILE.format(path=path))


def _norm(tok: str) -> str:
    return tok.lstrip("*").upper().replace("-", "_")


def _is_section(tok: str) -> bool:
    key = _norm(tok)
    if key in _FLAGS:
        return False
    if tok.startswith("*"):
        return True
    return key in _SECTION


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.split("**", 1)[0].replace(",", " ").strip()
        if not line:
            continue
        out.extend(shlex.split(line, posix=True))
    return out


def _sections(tokens: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if not _is_section(tok):
            i += 1
            continue
        key = _norm(tok)
        i += 1
        values: list[str] = []
        while i < n and not _is_section(tokens[i]):
            values.append(tokens[i])
            i += 1
        sections[key] = values
    return sections


def _is_float(tok: str) -> bool:
    try:
        float(tok)
    except ValueError:
        return False
    return True


def _floats(values: list[str], *, path: Path, field: str) -> list[float]:
    out: list[float] = []
    for tok in values:
        if not _is_float(tok):
            raise ValueError(
                f"EXAMPLE GEM card {path}: {field} value {tok!r} is not a number. "
                "Refusing to invent criticals."
            )
        out.append(float(tok))
    return out


def _names(sections: dict[str, list[str]], *, path: Path) -> list[str]:
    raw = sections.get("COMPNAME") or sections.get("COMPNAMES") or sections.get("COMP")
    if raw is None:
        raise ValueError(_MISSING_KW.format(path=path, field="*COMP / *COMPNAME"))
    names = [t for t in raw if not _is_float(t)]
    if not names and len(raw) == 1 and _is_float(raw[0]):
        raise ValueError(_MISSING_KW.format(path=path, field="*COMP / *COMPNAME"))
    if not names:
        raise ValueError(_MISSING_KW.format(path=path, field="*COMP / *COMPNAME"))
    return names


def _require_count(values: list[float], n: int, *, path: Path, field: str) -> list[float]:
    if len(values) != n:
        raise ValueError(_MISSING_COUNT.format(path=path, field=field, got=len(values), need=n))
    return values


def _units(sections: dict[str, list[str]], *, path: Path) -> str:
    raw = sections.get("INUNIT")
    if not raw:
        raise ValueError(
            f"EXAMPLE GEM card {path}: missing required keyword *INUNIT (*SI or *FIELD). "
            "Refusing to invent unit conversion for Tc/Pc."
        )
    unit = _norm(raw[0])
    if unit == "SI":
        return "SI"
    if unit == "FIELD":
        return "FIELD"
    raise ValueError(
        f"EXAMPLE GEM card {path}: *INUNIT {raw[0]!r} is not mapped; use *SI or *FIELD. "
        "Refusing to invent unit conversion."
    )


def _eos_is_pr(sections: dict[str, list[str]], *, path: Path) -> None:
    raw = sections.get("EOS") or sections.get("MODEL")
    if raw is None:
        raise ValueError(_MISSING_KW.format(path=path, field="*EOS"))
    model = _norm(raw[0]) if raw else ""
    if model not in _PR_MODELS:
        raise ValueError(
            f"EXAMPLE GEM card {path}: *EOS {raw[0] if raw else ''} is not Peng-Robinson. "
            "This path only maps *PR cards."
        )


def _kij_pairs(names: list[str], values: list[str], *, path: Path) -> list[list[Any]]:
    if not values:
        return []
    nums = _floats(values, path=path, field="*BIN")
    nc = len(names)
    tri = nc * (nc - 1) // 2
    pairs: list[list[Any]] = []
    if len(nums) == tri:
        k = 0
        for i in range(1, nc):
            for j in range(i):
                kij = nums[k]
                k += 1
                if kij != 0.0:
                    pairs.append([names[i], names[j], kij])
        return pairs
    if len(nums) == nc * nc:
        k = 0
        for i in range(nc):
            for j in range(nc):
                kij = nums[k]
                k += 1
                if i > j and kij != 0.0:
                    pairs.append([names[i], names[j], kij])
        return pairs
    raise ValueError(
        f"EXAMPLE GEM card {path}: *BIN has {len(nums)} values; "
        f"expected {tri} (lower triangle) or {nc * nc} (full matrix). "
        "Refusing to invent kij."
    )


def parse_gem_card(path: str | Path) -> dict[str, Any]:
    """Parse a GEM-style text card into the YAML EOS deck mapping.

    Required: ``*EOS`` (PR), ``*COMP`` / ``*COMPNAME``, ``*TCRIT``,
    ``*PCRIT``, ``*ACENTRIC`` (or ``*AC``), ``*INUNIT``, and an EXAMPLE
    label in the file. ``*MW`` and ``*BIN`` are optional. Missing file
    or required criticals raise; values are not invented.
    """
    deck = Path(path)
    if not deck.is_file():
        raise FileNotFoundError(_MISSING_FILE.format(path=path))
    text = deck.read_text(encoding="utf-8")
    if "EXAMPLE" not in text:
        raise ValueError(
            f"EXAMPLE GEM card {deck}: marker must identify EXAMPLE parameters; "
            "refusing to treat an unlabeled deck as a Jiyang/field card."
        )
    sections = _sections(_tokens(text))
    _eos_is_pr(sections, path=deck)
    names = _names(sections, path=deck)
    nc_raw = sections.get("NC") or sections.get("NCOMP")
    if nc_raw:
        nc_vals = _floats(nc_raw, path=deck, field="*NC")
        if len(nc_vals) != 1 or int(nc_vals[0]) != len(names):
            raise ValueError(
                f"EXAMPLE GEM card {deck}: *NC does not match *COMP count {len(names)}. "
                "Refusing to invent components."
            )
    unit = _units(sections, path=deck)
    if "TCRIT" not in sections:
        raise ValueError(_MISSING_KW.format(path=deck, field="*TCRIT (Tc)"))
    if "PCRIT" not in sections:
        raise ValueError(_MISSING_KW.format(path=deck, field="*PCRIT (Pc)"))
    ac_key = next((k for k in ("ACENTRIC", "AC", "ACF", "OMEGA") if k in sections), None)
    if ac_key is None:
        raise ValueError(_MISSING_KW.format(path=deck, field="*ACENTRIC / *AC (omega)"))
    n = len(names)
    tc_raw = _require_count(
        _floats(sections["TCRIT"], path=deck, field="*TCRIT"), n, path=deck, field="*TCRIT"
    )
    pc_raw = _require_count(
        _floats(sections["PCRIT"], path=deck, field="*PCRIT"), n, path=deck, field="*PCRIT"
    )
    omega = _require_count(
        _floats(sections[ac_key], path=deck, field="*ACENTRIC"), n, path=deck, field="*ACENTRIC"
    )
    if unit == "SI":
        tc = tc_raw
        pc = [p * _KPA_TO_PA for p in pc_raw]
    else:
        tc = [t * _RANKINE_TO_K for t in tc_raw]
        pc = [p * _PSIA_TO_PA for p in pc_raw]
    components: list[dict[str, Any]] = []
    mw_raw = None
    if "MW" in sections or "MOLWT" in sections or "MOL_WEIGHT" in sections:
        mw_key = "MW" if "MW" in sections else ("MOLWT" if "MOLWT" in sections else "MOL_WEIGHT")
        mw_raw = _require_count(
            _floats(sections[mw_key], path=deck, field="*MW"), n, path=deck, field="*MW"
        )
    for i, name in enumerate(names):
        row: dict[str, Any] = {"name": name, "Tc": tc[i], "Pc": pc[i], "omega": omega[i]}
        if mw_raw is not None:
            row["Mw_g_mol"] = mw_raw[i]
        components.append(row)
    desc = sections.get("DESC") or sections.get("DESCRIPTION")
    if desc:
        marker = " ".join(str(x) for x in desc)
        if "EXAMPLE" not in marker:
            marker = "EXAMPLE " + marker
    else:
        marker = (
            "EXAMPLE synthetic GEM-like snippet; NOT a Jiyang GEM card; "
            "NOT site-calibrated; NOT field-validated"
        )
    bin_vals = sections.get("BIN") or sections.get("BIJ") or sections.get("KIJ") or []
    return {
        "marker": marker,
        "components": components,
        "kij_pairs": _kij_pairs(names, bin_vals, path=deck),
    }


def load_eos_mixture_gem(path: str | Path) -> EosMixture:
    """Build ``EosMixture`` from a GEM-style text card via the YAML deck schema."""
    deck = Path(path)
    return mixture_from_deck_dict(parse_gem_card(deck), path=deck, source=_GEM_SOURCE)
