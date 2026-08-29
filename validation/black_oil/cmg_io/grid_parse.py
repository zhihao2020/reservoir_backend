"""Parse multi-time grid fields and well BHP from CMG IMEX .out text."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Field titles as they appear in IMEX ASCII .out
FIELD_TITLES = {
    "sw": "Water Saturation (fraction)",
    "so": "Oil Saturation (fraction)",
    "sg": "Gas Saturation (fraction)",
    "pressure": "Pressure (psi)",
}


def ft_to_m(x: float) -> float:
    return float(x) * 0.3048


def psi_to_pa(p: float) -> float:
    return float(p) * 6894.757293168


def parse_grid_series(
    out_path: Path | str,
    *,
    field: str,
    nx: int,
    ny: int,
    nz: int,
) -> list[tuple[float, NDArray[np.float64]]]:
    """Parse multi-time 3D grid fields from IMEX .out.

    Parameters
    ----------
    field:
        One of ``sw``, ``so``, ``sg``, ``pressure`` (case-insensitive),
        or a raw title substring match.

    Returns
    -------
    Sorted list of ``(time_days, array)`` with shape ``(nz, ny, nx)``.
    Pressure is returned in **psi** (caller converts). Saturations are fraction.
    """
    path = Path(out_path)
    text = path.read_text(encoding="latin-1", errors="ignore")
    key = str(field).strip().lower()
    title = FIELD_TITLES.get(key, str(field))

    by_t: dict[float, NDArray[np.float64]] = {}
    chunks = re.split(r"(?=Time\s*=\s*[0-9.]+)", text)
    for ch in chunks:
        mt = re.match(r"Time\s*=\s*([0-9.]+)", ch)
        if not mt:
            continue
        # title must appear near the start of the chunk (avoid nested tables)
        head = ch[:1200]
        if key == "pressure":
            # standalone "Pressure (psi)" only — not Bubble Point / Capillary / Offsets
            if not re.search(r"(?m)^\s*Pressure \(psi\)\s*$", head):
                continue
        elif title not in head:
            # saturation titles
            if key in ("sw", "so", "sg"):
                if not re.search(re.escape(title), head):
                    continue
            else:
                if title not in head:
                    continue
        time = float(mt.group(1))
        arr = _parse_plane_block(ch, nx=nx, ny=ny, nz=nz)
        if np.isfinite(arr).sum() <= 0:
            continue
        # first good map for this time wins (later tables can be other variables)
        if time not in by_t:
            by_t[time] = arr
            continue
        # prefer higher spatial variance (true reservoir pressure vs constants)
        old = by_t[time]
        if float(np.nanstd(arr)) > float(np.nanstd(old)):
            by_t[time] = arr
    return sorted(by_t.items())


def _parse_plane_block(
    chunk: str, *, nx: int, ny: int, nz: int
) -> NDArray[np.float64]:
    out = np.full((nz, ny, nx), np.nan, dtype=float)
    # whole-grid constant
    m_all = re.search(r"All values are\s+([0-9.E+-]+)", chunk[:1500])
    if m_all and "Plane K" not in chunk[:2000]:
        out[:, :, :] = float(m_all.group(1))
        return out

    for kplane in re.finditer(r"Plane K\s*=\s*(\d+)(.*?)(?=Plane K\s*=|\Z)", chunk, re.S):
        k = int(kplane.group(1))
        if not (1 <= k <= nz):
            continue
        body = kplane.group(2)
        if "All values are" in body[:200]:
            mval = re.search(r"All values are\s+([0-9.E+-]+)", body)
            if mval:
                out[k - 1, :, :] = float(mval.group(1))
            continue
        # Wide grids wrap: "I = 1..14" then J-rows, then "I = 15..21" then J-rows.
        i0 = 0
        for line in body.splitlines():
            mi = re.match(r"\s*I\s*=\s*(.+)$", line)
            if mi and "J=" not in line:
                cols = [int(x) for x in re.findall(r"\d+", mi.group(1))]
                if cols:
                    i0 = cols[0] - 1
                continue
            mj = re.match(r"\s*J=\s*(\d+)\s+(.+)$", line)
            if not mj:
                continue
            j = int(mj.group(1))
            if not (1 <= j <= ny):
                continue
            # CMG values: 3019.  3030.i  2972.p  1.23E+03  (trailing i/p = well flags)
            raw = re.findall(
                r"([+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[Ee][+-]?\d+)?)[a-zA-Z]?",
                mj.group(2),
            )
            for ii, v in enumerate(raw):
                i = i0 + ii
                if 0 <= i < nx:
                    out[k - 1, j - 1, i] = float(v)
    return out


def parse_bhp(out_path: Path | str) -> dict[float, tuple[float, float]]:
    """Injector/producer BHP (psi) keyed by report time (days)."""
    text = Path(out_path).read_text(encoding="latin-1", errors="ignore")
    by_t: dict[float, tuple[float, float]] = {}
    for m in re.finditer(
        r"Bottom Hole\s+psi\s+\+\s+([0-9.E+-]+)\s+\+\s+([0-9.E+-]+)", text
    ):
        start = max(0, m.start() - 800)
        window = text[start : m.start()]
        times = list(re.finditer(r"Time\s*=\s*([0-9.]+)", window))
        if not times:
            continue
        t = float(times[-1].group(1))
        by_t[t] = (float(m.group(1)), float(m.group(2)))
    return by_t


def parse_surface_rates_m3s(out_path: Path | str) -> dict[float, dict[str, float]]:
    """Water rates from IMEX field summary → m³/s (INJ +, PROD −).

    Reads ``Inst Surface Injection/Production Rates`` / ``Water STB/day``
    under each ``TIME:`` header. The first Water line after Time is usually
    *production* (often ~0) and must not be treated as injection.
    """
    text = Path(out_path).read_text(encoding="latin-1", errors="ignore")
    by_t: dict[float, dict[str, float]] = {}

    def stbday_to_m3s(x: float) -> float:
        return float(x) * 0.158987 / 86400.0

    chunks = re.split(r"(?=TIME:\s*[0-9.]+)", text)
    for ch in chunks:
        mt = re.match(r"TIME:\s*([0-9.]+)", ch)
        if not mt:
            continue
        t = float(mt.group(1))
        m_inj = re.search(
            r"Inst Surface Injection Rates.*?Water\s+STB/day\s+\+\s+"
            r"([0-9.Ee+-]+)\s+\+\s+([0-9.Ee+-]+)",
            ch,
            re.S,
        )
        m_prd = re.search(
            r"Inst Surface Production Rates.*?Water\s+STB/day\s+\+\s+"
            r"([0-9.Ee+-]+)\s+\+\s+([0-9.Ee+-]+)",
            ch,
            re.S,
        )
        if m_inj is None and m_prd is None:
            continue
        inj_stb = 0.0
        prod_stb = 0.0
        if m_inj is not None:
            a, b = abs(float(m_inj.group(1))), abs(float(m_inj.group(2)))
            inj_stb = max(a, b)
        if m_prd is not None:
            a, b = abs(float(m_prd.group(1))), abs(float(m_prd.group(2)))
            # producer column is the second well in these two-well decks
            prod_stb = b if b > 0.0 else a
        by_t[t] = {"INJ": stbday_to_m3s(inj_stb), "PROD": -stbday_to_m3s(prod_stb)}
    return by_t


def parse_liquid_rates_m3s(out_path: Path | str) -> dict[float, dict[str, float]]:
    """Injector water + producer oil+water (reservoir voidage) in m³/s.

    Two-well IMEX summaries: first numeric column after the zeros is usually
    the injector, the next is the producer. Liquid production is oil+water so
    an incompressible / slightly-compressible F has a real sink.
    """
    text = Path(out_path).read_text(encoding="latin-1", errors="ignore")
    by_t: dict[float, dict[str, float]] = {}

    def stbday_to_m3s(x: float) -> float:
        return float(x) * 0.158987 / 86400.0

    chunks = re.split(r"(?=TIME:\s*[0-9.]+)", text)
    for ch in chunks:
        mt = re.match(r"TIME:\s*([0-9.]+)", ch)
        if not mt:
            continue
        t = float(mt.group(1))
        m_inj_w = re.search(
            r"Inst Surface Injection Rates.*?Water\s+STB/day\s+\+\s+"
            r"([0-9.Ee+-]+)\s+\+\s+([0-9.Ee+-]+)",
            ch,
            re.S,
        )
        m_prd_o = re.search(
            r"Inst Surface Production Rates.*?Oil\s+STB/day\s+\+\s+"
            r"([0-9.Ee+-]+)\s+\+\s+([0-9.Ee+-]+)",
            ch,
            re.S,
        )
        m_prd_w = re.search(
            r"Inst Surface Production Rates.*?Water\s+STB/day\s+\+\s+"
            r"([0-9.Ee+-]+)\s+\+\s+([0-9.Ee+-]+)",
            ch,
            re.S,
        )
        if m_inj_w is None and m_prd_o is None:
            continue
        inj = 0.0
        if m_inj_w is not None:
            inj = max(abs(float(m_inj_w.group(1))), abs(float(m_inj_w.group(2))))
        oil = 0.0
        water = 0.0
        if m_prd_o is not None:
            a, b = abs(float(m_prd_o.group(1))), abs(float(m_prd_o.group(2)))
            oil = b if b > 0.0 else a
        if m_prd_w is not None:
            a, b = abs(float(m_prd_w.group(1))), abs(float(m_prd_w.group(2)))
            water = b if b > 0.0 else a
        by_t[t] = {"INJ": stbday_to_m3s(inj), "PROD": -stbday_to_m3s(oil + water)}
    return by_t
