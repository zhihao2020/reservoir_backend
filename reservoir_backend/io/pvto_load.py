"""Parse CMG/IMEX *PVTO / *PVTW / *PVDG / *PVT text into BlackOilPVT SI arrays.

Default units match ``BlackOilPVT.cmg_seawater``: FIELD psi, scf/stb, cP.
``*INUNIT *SI`` / ``*METRIC`` / ``*LAB``: CMG kPa, sm3/sm3, cP.
Undersaturated *PVTO branches are ignored (1-D saturated curve + optional *CO).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from reservoir_backend.physics.pvt import PSI, SCF_PER_STB

_D_EXP = re.compile(r"([0-9.])[Dd]([+-]?\d+)")
_REPEAT = re.compile(r"^(\d+)\*(.+)$")
_CMG_RE = re.compile(r"\*(?:PVTO|PVTW|PVDG|PVTG|PVT)\b", re.IGNORECASE)

_SECTION = frozenset(
    {
        "PVTO",
        "PVTW",
        "PVDG",
        "PVTG",
        "PVT",
        "INUNIT",
        "CO",
        "CW",
        "BWI",
        "REFPW",
        "PB",
        "SVISC",
        "VISW",
    }
)
_SKIP_MOD = frozenset({"GRAPH", "MODEL", "SATURATED", "UNDERSATURATED"})
_FIELD = frozenset({"FIELD", "ENGLISH"})
_SI = frozenset({"SI", "METRIC", "LAB", "KPA"})
_SCALAR_KW = frozenset({"CO", "CW", "BWI", "REFPW", "PB", "SVISC", "VISW"})
_TABLE_KEYS = frozenset(
    {"p_tab", "rs_tab", "bo_tab", "eg_tab", "muo_tab", "mug_tab", "p_w_tab", "bw_tab", "muw_tab"}
)
_SCALAR_KEYS = frozenset({"bw_ref", "cw", "pref_w", "mu_w", "co", "pb", "bo_ref", "pref_o"})


def looks_like_cmg_pvt(text: str) -> bool:
    return _CMG_RE.search(text) is not None


def _strip_comment(line: str) -> str:
    cut = len(line)
    for mark in ("--", "**", "!"):
        idx = line.find(mark)
        if idx >= 0:
            cut = min(cut, idx)
    return line[:cut]


def _fortran_float(token: str) -> float:
    return float(_D_EXP.sub(r"\1e\2", token))


def _norm_kw(tok: str) -> str:
    return tok.lstrip("*").upper().replace("_", "-")


def _is_kw(tok: str) -> bool:
    if not tok.startswith("*") and tok.upper() not in _SECTION:
        return False
    name = _norm_kw(tok)
    return name in _SECTION or name in _SKIP_MOD or name in _FIELD or name in _SI


def _line_tokens(line: str) -> list[str]:
    out: list[str] = []
    for raw in _strip_comment(line).replace(",", " ").split():
        tok = raw.strip()
        if tok and tok != "/":
            out.append(tok)
    return out


def _expand_nums(tokens: list[str]) -> list[float]:
    out: list[float] = []
    for tok in tokens:
        if _is_kw(tok):
            continue
        m = _REPEAT.match(tok)
        if m:
            out.extend([_fortran_float(m.group(2))] * int(m.group(1)))
            continue
        try:
            out.append(_fortran_float(tok))
        except ValueError:
            continue
    return out


def _unit_kind(name: str) -> str:
    n = _norm_kw(name)
    if n in _SI:
        return "si"
    return "field"


class _Conv:
    """FIELD (cmg_seawater) or CMG SI/METRIC/LAB."""

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def p(self, v: float) -> float:
        return float(v) * PSI if self.kind == "field" else float(v) * 1.0e3

    def rs(self, v: float) -> float:
        return float(v) * SCF_PER_STB if self.kind == "field" else float(v)

    def eg(self, v: float) -> float:
        return float(v) * SCF_PER_STB if self.kind == "field" else float(v)

    def mu(self, v: float) -> float:
        return float(v) * 1.0e-3

    def c(self, v: float) -> float:
        return float(v) / PSI if self.kind == "field" else float(v) / 1.0e3

    def bg_to_eg(self, bg: float) -> float:
        inv = 1.0 / max(float(bg), 1.0e-30)
        return inv * SCF_PER_STB if self.kind == "field" else inv


def _sort_by_p(
    p: list[float], *cols: list[float]
) -> tuple[list[float], ...]:
    order = np.argsort(np.asarray(p, dtype=float))
    packed = [np.asarray(p, dtype=float)[order].tolist()]
    for col in cols:
        packed.append(np.asarray(col, dtype=float)[order].tolist())
    return tuple(packed)


def parse_pvto(text: str) -> dict[str, Any]:
    """Return a mapping of BlackOilPVT table / water-scalar names in SI."""
    if not looks_like_cmg_pvt(text):
        raise ValueError("not CMG *PVTO/*PVTW/*PVDG/*PVT text")

    kind = "field"
    section: str | None = None
    oil_sat: list[list[float]] = []
    pvt_rows: list[list[float]] = []
    gas_rows: list[list[float]] = []
    water_rows: list[list[float]] = []
    raw_scalars: dict[str, float] = {}

    for line in text.splitlines():
        tokens = _line_tokens(line)
        if not tokens:
            continue
        i = 0
        while i < len(tokens) and _is_kw(tokens[i]):
            name = _norm_kw(tokens[i])
            i += 1
            if name in _SKIP_MOD:
                continue
            if name == "INUNIT":
                if i < len(tokens):
                    kind = _unit_kind(tokens[i])
                    i += 1
                continue
            if name in _FIELD or name in _SI:
                kind = _unit_kind(name)
                continue
            if name in _SCALAR_KW:
                rest = _expand_nums(tokens[i:])
                if rest:
                    raw_scalars[name] = rest[0]
                    i = len(tokens)
                else:
                    section = name
                continue
            if name in {"PVTO", "PVTW", "PVDG", "PVTG", "PVT"}:
                section = name
                continue
        nums = _expand_nums(tokens[i:])
        if not nums:
            continue
        if section in _SCALAR_KW:
            raw_scalars[section] = nums[0]
            section = None
            continue
        if section == "PVTO":
            if len(nums) >= 4:
                oil_sat.append(nums[:4])
            continue
        if section == "PVT":
            if len(nums) >= 5:
                pvt_rows.append(nums)
            continue
        if section in {"PVDG", "PVTG"}:
            if len(nums) >= 4:
                gas_rows.append([nums[0], nums[2], nums[3]])
            elif len(nums) >= 3:
                gas_rows.append(nums[:3])
            continue
        if section == "PVTW":
            water_rows.append(nums)
            continue

    conv = _Conv(kind)
    out: dict[str, Any] = {}

    if pvt_rows:
        p_tab: list[float] = []
        rs_tab: list[float] = []
        bo_tab: list[float] = []
        eg_tab: list[float] = []
        muo_tab: list[float] = []
        mug_tab: list[float] = []
        for row in pvt_rows:
            p_tab.append(conv.p(row[0]))
            rs_tab.append(conv.rs(row[1]))
            bo_tab.append(float(row[2]))
            if len(row) >= 6:
                eg_tab.append(conv.eg(row[3]))
                muo_tab.append(conv.mu(row[4]))
                mug_tab.append(conv.mu(row[5]))
            else:
                muo_tab.append(conv.mu(row[3]))
                mug_tab.append(conv.mu(row[4]))
        p_tab, rs_tab, bo_tab, muo_tab, mug_tab = _sort_by_p(
            p_tab, rs_tab, bo_tab, muo_tab, mug_tab
        )
        out["p_tab"] = p_tab
        out["rs_tab"] = rs_tab
        out["bo_tab"] = bo_tab
        out["muo_tab"] = muo_tab
        out["mug_tab"] = mug_tab
        if eg_tab:
            _, eg_tab = _sort_by_p(
                [conv.p(r[0]) for r in pvt_rows],
                [conv.eg(r[3]) for r in pvt_rows],
            )
            out["eg_tab"] = eg_tab
    elif oil_sat:
        p_tab, rs_tab, bo_tab, muo_tab = _sort_by_p(
            [conv.p(r[1]) for r in oil_sat],
            [conv.rs(r[0]) for r in oil_sat],
            [float(r[2]) for r in oil_sat],
            [conv.mu(r[3]) for r in oil_sat],
        )
        out["p_tab"] = p_tab
        out["rs_tab"] = rs_tab
        out["bo_tab"] = bo_tab
        out["muo_tab"] = muo_tab

    if gas_rows and "eg_tab" not in out:
        gp = [conv.p(r[0]) for r in gas_rows]
        raw_f = [float(r[1]) for r in gas_rows]
        gmu = [conv.mu(r[2]) for r in gas_rows]
        if raw_f and min(raw_f) > 1.0:
            geg = [conv.eg(v) for v in raw_f]
        else:
            geg = [conv.bg_to_eg(v) for v in raw_f]
        gp, geg, gmu = _sort_by_p(gp, geg, gmu)
        if "p_tab" in out:
            p_oil = np.asarray(out["p_tab"], dtype=float)
            out["eg_tab"] = np.interp(p_oil, gp, geg).tolist()
            out["mug_tab"] = np.interp(p_oil, gp, gmu).tolist()
        else:
            out["p_tab"] = gp
            out["eg_tab"] = geg
            out["mug_tab"] = gmu

    if water_rows:
        if len(water_rows) == 1 and len(water_rows[0]) >= 4:
            pref, bw, cw, visw = water_rows[0][:4]
            out["pref_w"] = conv.p(pref)
            out["bw_ref"] = float(bw)
            out["cw"] = conv.c(cw)
            out["mu_w"] = conv.mu(visw)
        else:
            pw: list[float] = []
            bw_t: list[float] = []
            muw: list[float] = []
            for row in water_rows:
                if len(row) >= 3:
                    pw.append(conv.p(row[0]))
                    bw_t.append(float(row[1]))
                    muw.append(conv.mu(row[2]))
                elif len(row) >= 2:
                    pw.append(conv.p(row[0]))
                    bw_t.append(float(row[1]))
            if pw:
                if muw and len(muw) == len(pw):
                    pw, bw_t, muw = _sort_by_p(pw, bw_t, muw)
                    out["muw_tab"] = muw
                else:
                    pw, bw_t = _sort_by_p(pw, bw_t)
                out["p_w_tab"] = pw
                out["bw_tab"] = bw_t

    if "CO" in raw_scalars:
        out["co"] = conv.c(raw_scalars["CO"])
    if "CW" in raw_scalars:
        out.setdefault("cw", conv.c(raw_scalars["CW"]))
    if "BWI" in raw_scalars:
        out.setdefault("bw_ref", float(raw_scalars["BWI"]))
    if "REFPW" in raw_scalars:
        out.setdefault("pref_w", conv.p(raw_scalars["REFPW"]))
    if "PB" in raw_scalars:
        out["pb"] = conv.p(raw_scalars["PB"])
        out["pref_o"] = out["pb"]
    if "SVISC" in raw_scalars:
        out.setdefault("mu_w", conv.mu(raw_scalars["SVISC"]))
    if "VISW" in raw_scalars:
        out.setdefault("mu_w", conv.mu(raw_scalars["VISW"]))

    if not any(k in out for k in _TABLE_KEYS | _SCALAR_KEYS):
        raise ValueError("CMG PVT text has no *PVTO/*PVTW/*PVDG/*PVT records")
    return out
