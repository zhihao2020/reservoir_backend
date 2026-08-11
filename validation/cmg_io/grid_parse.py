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
        for jline in re.finditer(r"J=\s*(\d+)\s+(.+)", body):
            j = int(jline.group(1))
            if not (1 <= j <= ny):
                continue
            # CMG values: 3019.  3030.i  2972.p  1.23E+03  (trailing i/p = well flags)
            raw = re.findall(
                r"([+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[Ee][+-]?\d+)?)[a-zA-Z]?",
                jline.group(2),
            )
            for i, v in enumerate(raw[:nx]):
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
    """Best-effort water rates → m³/s (INJ positive, PROD negative)."""
    text = Path(out_path).read_text(encoding="latin-1", errors="ignore")
    by_t: dict[float, dict[str, float]] = {}

    def stbday_to_m3s(x: float) -> float:
        return x * 0.158987 / 86400.0

    for m in re.finditer(
        r"Time\s*=\s*([0-9.]+).*?Water\s+STB/day\s+\+\s+([0-9.E+-]+)\s+\+\s+([0-9.E+-]+)",
        text,
        re.S | re.I,
    ):
        t = float(m.group(1))
        a, b = float(m.group(2)), float(m.group(3))
        q1, q2 = stbday_to_m3s(a), stbday_to_m3s(b)
        by_t[t] = {"INJ": abs(q1), "PROD": -abs(q2) if abs(q2) > 0 else -abs(q1)}
    return by_t
