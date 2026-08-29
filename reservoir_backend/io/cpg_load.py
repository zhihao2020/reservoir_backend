"""Load CMG/Eclipse *GRID keyword subset and COORD/ZCORN snippets."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from reservoir_backend.exceptions import GridError
from reservoir_backend.grid.corner_point import CornerPointGrid

_REPEAT = re.compile(r"^(\d+)\*(.+)$")
_D_EXP = re.compile(r"([0-9.])[Dd]([+-]?\d+)")

_SECTION_KW = frozenset(
    {
        "GRID",
        "CART",
        "CARTESIAN",
        "CORNER",
        "CORNER-POINT",
        "CPG",
        "SPECGRID",
        "DIMENS",
        "NX",
        "NY",
        "NZ",
        "DX",
        "DY",
        "DZ",
        "DI",
        "DJ",
        "DK",
        "COORD",
        "ZCORN",
        "ACTNUM",
    }
)
_TYPE_CART = frozenset({"CART", "CARTESIAN"})
_TYPE_CPG = frozenset({"CORNER", "CORNER-POINT", "CPG"})
_AXIS_MOD = frozenset({"CON", "IVAR", "JVAR", "KVAR", "ALL"})
_DX_KEYS = frozenset({"DX", "DI"})
_DY_KEYS = frozenset({"DY", "DJ"})
_DZ_KEYS = frozenset({"DZ", "DK"})
_VAR_KEYS = frozenset(
    {
        "nx",
        "ny",
        "nz",
        "dx",
        "dy",
        "dz",
        "DX",
        "DY",
        "DZ",
        "coord",
        "COORD",
        "zcorn",
        "ZCORN",
        "actnum",
        "ACTNUM",
    }
)


def looks_like_coord_zcorn(text: str) -> bool:
    up = text.upper()
    return "*COORD" in up or "*ZCORN" in up or re.search(r"(?m)^\s*COORD\b", up) is not None


def looks_like_grid_deck(text: str) -> bool:
    up = text.upper()
    if looks_like_coord_zcorn(text):
        return True
    if "*GRID" in up or re.search(r"(?m)^\s*GRID\b", up) is not None:
        return True
    if re.search(r"(?m)^\s*\*?(?:SPECGRID|DIMENS|ACTNUM)\b", up) is not None:
        return True
    if re.search(r"(?m)^\s*\*?(?:DX|DY|DZ|DI|DJ|DK|NX|NY|NZ|CART|CARTESIAN)\b", up) is not None:
        return True
    return False


def _strip_comment(line: str) -> str:
    cut = len(line)
    for mark in ("--", "**", "!"):
        idx = line.find(mark)
        if idx >= 0:
            cut = min(cut, idx)
    return line[:cut]


def _fortran_float(token: str) -> float:
    return float(_D_EXP.sub(r"\1e\2", token))


def _expand_tokens(text: str) -> list[float]:
    out: list[float] = []
    for raw in text.replace(",", " ").split():
        tok = raw.strip()
        if not tok or tok == "/":
            continue
        if tok.startswith("*") and not _REPEAT.match(tok.lstrip("*")):
            continue
        m = _REPEAT.match(tok)
        if m:
            out.extend([_fortran_float(m.group(2))] * int(m.group(1)))
            continue
        try:
            out.append(_fortran_float(tok))
        except ValueError as exc:
            raise GridError("bad COORD/ZCORN token: " + tok) from exc
    return out


def _norm_kw(tok: str) -> str:
    return tok.lstrip("*").upper().replace("_", "-")


def _is_section_kw(tok: str) -> bool:
    return _norm_kw(tok) in _SECTION_KW


def _tokenize_deck(text: str) -> list[str]:
    tokens: list[str] = []
    for line in text.splitlines():
        for raw in _strip_comment(line).replace(",", " ").split():
            tok = raw.strip()
            if tok:
                tokens.append(tok)
    return tokens


def _first_ints(body: list[str], n: int) -> list[int]:
    out: list[int] = []
    for tok in body:
        name = _norm_kw(tok)
        if name in _TYPE_CART or name in _TYPE_CPG or name in _AXIS_MOD:
            continue
        try:
            out.append(int(round(_fortran_float(tok))))
        except ValueError:
            continue
        if len(out) >= n:
            break
    return out


def _set_counts(out: dict[str, Any], ints: list[int]) -> None:
    if len(ints) >= 3:
        out["nx"] = ints[0]
        out["ny"] = ints[1]
        out["nz"] = ints[2]


def _axis_values(body: list[str]) -> Any:
    start = 0
    if body and _norm_kw(body[0]) in _AXIS_MOD:
        start = 1
    vals = _expand_tokens(" ".join(body[start:]))
    if not vals:
        raise GridError("empty DX/DY/DZ (or DI/DJ/DK) array")
    if len(vals) == 1:
        return vals[0]
    return vals


def _apply_section(out: dict[str, Any], name: str, body: list[str]) -> None:
    if name == "GRID":
        for tok in body:
            t = _norm_kw(tok)
            if t in _TYPE_CART:
                out["type"] = "cartesian"
            elif t in _TYPE_CPG:
                out["type"] = "corner_point"
        _set_counts(out, _first_ints(body, 3))
        return
    if name in _TYPE_CART:
        out["type"] = "cartesian"
        _set_counts(out, _first_ints(body, 3))
        return
    if name in _TYPE_CPG:
        out["type"] = "corner_point"
        _set_counts(out, _first_ints(body, 3))
        return
    if name in {"SPECGRID", "DIMENS"}:
        _set_counts(out, _first_ints(body, 3))
        return
    if name == "NX":
        ints = _first_ints(body, 1)
        if ints:
            out["nx"] = ints[0]
        return
    if name == "NY":
        ints = _first_ints(body, 1)
        if ints:
            out["ny"] = ints[0]
        return
    if name == "NZ":
        ints = _first_ints(body, 1)
        if ints:
            out["nz"] = ints[0]
        return
    if name in _DX_KEYS:
        out["dx"] = _axis_values(body)
        return
    if name in _DY_KEYS:
        out["dy"] = _axis_values(body)
        return
    if name in _DZ_KEYS:
        out["dz"] = _axis_values(body)
        return
    if name == "COORD":
        out["coord"] = _expand_tokens(" ".join(body))
        return
    if name == "ZCORN":
        out["zcorn"] = _expand_tokens(" ".join(body))
        return
    if name == "ACTNUM":
        out["actnum"] = [int(round(v)) for v in _expand_tokens(" ".join(body))]
        return


def parse_grid_deck(text: str) -> dict[str, Any]:
    """CMG/Eclipse *GRID subset. COORD/ZCORN stay in Eclipse order."""
    tokens = _tokenize_deck(text)
    out: dict[str, Any] = {}
    i = 0
    ntok = len(tokens)
    while i < ntok:
        tok = tokens[i]
        if tok == "/":
            i += 1
            continue
        if not _is_section_kw(tok):
            i += 1
            continue
        name = _norm_kw(tok)
        i += 1
        body: list[str] = []
        while i < ntok and tokens[i] != "/" and not _is_section_kw(tokens[i]):
            body.append(tokens[i])
            i += 1
        if i < ntok and tokens[i] == "/":
            i += 1
        _apply_section(out, name, body)
    return out


def parse_grdecl_coord_zcorn(text: str) -> dict[str, list[float]]:
    """COORD and ZCORN arrays (Eclipse order). Ignores other *GRID keywords."""
    parsed = parse_grid_deck(text)
    found: dict[str, list[float]] = {}
    if "coord" in parsed:
        found["coord"] = [float(v) for v in parsed["coord"]]
    if "zcorn" in parsed:
        found["zcorn"] = [float(v) for v in parsed["zcorn"]]
    return found


def _pick(cfg: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in cfg and cfg[name] is not None:
            return cfg[name]
    return None


def _as_int(name: str, value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise GridError("grid." + name + " must be a positive integer") from exc
    if n <= 0:
        raise GridError("grid." + name + " must be a positive integer")
    return n


def _unwrap_mapping(data: Any, path: Path) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise GridError("grid sidecar must be a mapping: " + str(path))
    inner = data.get("grid")
    if isinstance(inner, dict) and not any(k in data for k in _VAR_KEYS):
        return inner
    return data


def load_grid_sidecar(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return _unwrap_mapping(json.loads(text), path)
    if suffix in {".yaml", ".yml"}:
        return _unwrap_mapping(yaml.safe_load(text), path)
    if looks_like_grid_deck(text) or suffix in {".grdecl", ".dat", ".inc"}:
        parsed = parse_grid_deck(text)
        if parsed:
            return parsed
    loaded = yaml.safe_load(text)
    return _unwrap_mapping(loaded, path)


def merge_grid_sidecar(grid_cfg: dict[str, Any], cfg_dir: Path) -> dict[str, Any]:
    raw_file = grid_cfg.get("file")
    if raw_file in (None, ""):
        return dict(grid_cfg)
    if isinstance(raw_file, dict):
        sidecar: dict[str, Any] = dict(raw_file)
    else:
        path = Path(str(raw_file))
        if not path.is_absolute():
            path = cfg_dir / path
        if not path.is_file():
            raise GridError("grid.file not found: " + str(path))
        sidecar = load_grid_sidecar(path)
    merged = dict(sidecar)
    for key, val in grid_cfg.items():
        if key == "file":
            continue
        merged[key] = val
    return merged


def _load_blob(path: Path, name: str) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.ascontiguousarray(np.load(path), dtype=float).ravel()
    text = path.read_text(encoding="utf-8")
    if looks_like_coord_zcorn(text) and suffix not in {".json", ".yaml", ".yml"}:
        parsed = parse_grdecl_coord_zcorn(text)
        key = "coord" if name.lower() == "coord" else "zcorn"
        if key not in parsed:
            raise GridError(str(path) + " has no " + name + " array")
        return np.ascontiguousarray(parsed[key], dtype=float)
    if suffix in {".json", ".yaml", ".yml"}:
        data = json.loads(text) if suffix == ".json" else yaml.safe_load(text)
        if isinstance(data, dict):
            inner = data.get("grid") if isinstance(data.get("grid"), dict) else data
            raw = _pick(inner, (name, name.upper(), "values", "data"))
            if raw is None:
                raise GridError(str(path) + " has no " + name + " array")
            return np.ascontiguousarray(np.asarray(raw, dtype=float).ravel())
        return np.ascontiguousarray(np.asarray(data, dtype=float).ravel())
    return np.ascontiguousarray(_expand_tokens(text), dtype=float)


def _cpg_array(raw: Any, name: str, n: int, cfg_dir: Path) -> np.ndarray:
    if raw is None:
        raise GridError("grid." + name + " is required for corner-point (length " + str(n) + ")")
    if isinstance(raw, (str, Path)):
        path = Path(str(raw))
        if not path.is_absolute():
            path = cfg_dir / path
        if not path.is_file():
            raise GridError("grid." + name + " file not found: " + str(path))
        arr = _load_blob(path, name)
    else:
        arr = np.ascontiguousarray(np.asarray(raw, dtype=float).ravel())
    if arr.size != n:
        raise GridError("grid." + name + " length " + str(int(arr.size)) + " != " + str(n))
    if not np.isfinite(arr).all():
        raise GridError("grid." + name + " must be finite")
    return arr


def _actnum_array(raw: Any, n: int, cfg_dir: Path) -> np.ndarray:
    if isinstance(raw, (str, Path)) and not str(raw).replace(".", "", 1).lstrip("-").isdigit():
        path = Path(str(raw))
        if not path.is_absolute():
            path = cfg_dir / path
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            parsed = parse_grid_deck(text)
            if "actnum" in parsed:
                raw = parsed["actnum"]
            else:
                raw = _expand_tokens(text)
    arr = np.ascontiguousarray(np.asarray(raw, dtype=float).ravel())
    if arr.size != n:
        raise GridError("grid.actnum length " + str(int(arr.size)) + " != " + str(n))
    return arr


def cpg_grid_from_cfg(
    grid_cfg: dict[str, Any],
    origin: tuple[float, float, float],
    cfg_dir: Path,
) -> CornerPointGrid:
    cfg = merge_grid_sidecar(grid_cfg, cfg_dir)
    nx = _as_int("nx", _pick(cfg, ("nx", "NX")))
    ny = _as_int("ny", _pick(cfg, ("ny", "NY")))
    nz = _as_int("nz", _pick(cfg, ("nz", "NZ")))
    coord_raw = _pick(cfg, ("coord", "COORD"))
    zcorn_raw = _pick(cfg, ("zcorn", "ZCORN"))
    missing = []
    if coord_raw is None:
        missing.append("coord")
    if zcorn_raw is None:
        missing.append("zcorn")
    if missing:
        raise GridError(
            "grid." + " and grid.".join(missing) + " is required for corner-point"
        )
    coord = _cpg_array(coord_raw, "coord", (nx + 1) * (ny + 1) * 6, cfg_dir)
    zcorn = _cpg_array(zcorn_raw, "zcorn", 8 * nx * ny * nz, cfg_dir)
    act_raw = _pick(cfg, ("actnum", "ACTNUM"))
    actnum = None if act_raw is None else _actnum_array(act_raw, nx * ny * nz, cfg_dir)
    return CornerPointGrid.from_coord_zcorn(
        nx, ny, nz, coord, zcorn, origin=origin, actnum=actnum
    )
