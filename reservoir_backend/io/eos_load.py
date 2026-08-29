"""Load a published PR card (YAML or Eclipse/OPM COMPAD keywords).

Numbers must come from a file. Missing Tc/Pc/ω is a hard error.
Does not invent Jiyang GEM criticals. Does not import ``references/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from reservoir_backend.eos.pr import PengRobinson

_BAR = 1.0e5  # PCRIT default: bar → Pa
_G_MOL = 1.0e-3  # MW g/mol → kg/mol


def load_eos_card(path: str | Path) -> PengRobinson:
    """YAML mapping or keyword text with TCRIT, PCRIT, ACF, MW, BIC."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"EOS card not found: {path}; refuse invented criticals")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml", ".json"}:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"EOS YAML {path} must be a mapping")
        return _from_mapping(data)
    return _from_keywords(text)


def _from_mapping(data: dict) -> PengRobinson:
    names = tuple(str(x) for x in (data.get("names") or data.get("components") or ()))
    tc = _arr(data, ("tc_k", "tc", "TCRIT"))
    pc = _arr(data, ("pc_pa", "pc"))
    if pc is None:
        pc_bar = _arr(data, ("pc_bar", "PCRIT"))
        if pc_bar is not None:
            pc = pc_bar * _BAR
    omega = _arr(data, ("omega", "acf", "ACF"))
    mw = _arr(data, ("mw_kg_mol", "mw"))
    if mw is None:
        mw_g = _arr(data, ("mw_g_mol", "MW"))
        if mw_g is not None:
            mw = mw_g * _G_MOL
    if names and tc is not None and len(names) != tc.size:
        raise ValueError("EOS names length must match Tc")
    if names is None or not names:
        if tc is None:
            raise ValueError("EOS card missing names and Tc")
        names = tuple(f"C{i+1}" for i in range(tc.size))
    return _build(names, tc, pc, omega, mw, data.get("kij") or data.get("BIC"))


def _arr(data: dict, keys: tuple[str, ...]) -> np.ndarray | None:
    for k in keys:
        if k in data and data[k] is not None:
            return np.asarray(data[k], dtype=float).ravel()
    return None


def _from_keywords(text: str) -> PengRobinson:
    blocks: dict[str, list[float]] = {}
    names: list[str] = []
    current = None
    skip_names = False
    for raw in text.splitlines():
        line = raw.split("--")[0].strip()
        if not line:
            continue
        key = line.split()[0].upper().replace("*", "")
        if key in {"CNAMES", "COMPNAME", "COMPS"}:
            current = "NAMES"
            skip_names = True
            rest = line.split(maxsplit=1)
            if len(rest) > 1 and rest[1] not in {"/", ""}:
                names.extend(rest[1].replace("/", "").split())
            continue
        if key in {"TCRIT", "PCRIT", "ACF", "MW", "BIC", "VCRIT"}:
            current = key
            skip_names = False
            rest = line.split()[1:]
            _eat_nums(blocks, current, rest)
            continue
        if key == "EOS":
            current = None
            continue
        if line == "/":
            current = None
            skip_names = False
            continue
        if current == "NAMES" or skip_names:
            if line != "/":
                names.extend(line.replace("/", "").split())
            continue
        if current:
            _eat_nums(blocks, current, line.replace("/", "").split())
    if "TCRIT" in blocks and not names:
        names = [f"C{i+1}" for i in range(len(blocks["TCRIT"]))]
    kij = blocks.get("BIC")
    return _build(
        tuple(names),
        np.asarray(blocks.get("TCRIT"), dtype=float) if "TCRIT" in blocks else None,
        np.asarray(blocks["PCRIT"], dtype=float) * _BAR if "PCRIT" in blocks else None,
        np.asarray(blocks.get("ACF"), dtype=float) if "ACF" in blocks else None,
        np.asarray(blocks["MW"], dtype=float) * _G_MOL if "MW" in blocks else None,
        kij,
    )


def _eat_nums(blocks: dict[str, list[float]], key: str, tokens: list[str]) -> None:
    vals: list[float] = []
    for tok in tokens:
        if tok == "/":
            break
        try:
            vals.append(float(tok))
        except ValueError:
            continue
    blocks.setdefault(key, []).extend(vals)


def _build(names, tc, pc, omega, mw, kij_raw) -> PengRobinson:
    if tc is None or pc is None or omega is None or mw is None:
        raise ValueError("EOS card needs Tc, Pc, acentric factor, and Mw; refuse invented values")
    tc = np.asarray(tc, dtype=float).ravel()
    pc = np.asarray(pc, dtype=float).ravel()
    omega = np.asarray(omega, dtype=float).ravel()
    mw = np.asarray(mw, dtype=float).ravel()
    n = tc.size
    if min(pc.size, omega.size, mw.size) != n:
        raise ValueError("EOS card columns must align")
    if np.any(tc <= 0.0) or np.any(pc <= 0.0) or np.any(mw <= 0.0):
        raise ValueError("Tc, Pc, Mw must be positive")
    kij = _kij_square(n, kij_raw)
    if len(names) != n:
        names = tuple(names) + tuple(f"C{i+1}" for i in range(len(names), n))
        names = names[:n]
    return PengRobinson(tc=tc, pc=pc, omega=omega, mw=mw, kij=kij, names=tuple(names))


def _kij_square(n: int, raw) -> np.ndarray:
    kij = np.zeros((n, n))
    if raw is None:
        return kij
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 2:
        if arr.shape != (n, n):
            raise ValueError("kij must be (nc, nc)")
        return arr
    flat = arr.ravel()
    if flat.size == n * n:
        return flat.reshape(n, n)
    # Lower triangle below diagonal: k21; k31 k32; ...
    need = n * (n - 1) // 2
    if flat.size != need:
        if np.allclose(flat, 0.0):
            return kij
        raise ValueError(f"BIC length {flat.size} != {need} (lower triangle) or {n*n}")
    k = 0
    for i in range(1, n):
        for j in range(i):
            kij[i, j] = kij[j, i] = float(flat[k])
            k += 1
    return kij
