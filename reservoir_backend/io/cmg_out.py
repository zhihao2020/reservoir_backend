"""Parse IMEX ASCII .out grid fields (product copy; validation re-exports)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FIELD_TITLES = {
    "sw": "Water Saturation (fraction)",
    "so": "Oil Saturation (fraction)",
    "sg": "Gas Saturation (fraction)",
    "pressure": "Pressure (psi)",
}


def ft_to_m(x: float) -> float:
    return float(x) * 0.3048


def psi_to_pa(p: float | NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(p, dtype=float) * 6894.757293168


def parse_grid_series(
    out_path: Path | str,
    *,
    field: str,
    nx: int,
    ny: int,
    nz: int,
) -> list[tuple[float, NDArray[np.float64]]]:
    """Return sorted ``(time_days, array[nz, ny, nx])``; pressure in psi."""
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
        head = ch[:1200]
        if key == "pressure":
            if not re.search(r"(?m)^\s*Pressure \(psi\)\s*$", head):
                continue
        elif title not in head:
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
        if time not in by_t:
            by_t[time] = arr
            continue
        old = by_t[time]
        if float(np.nanstd(arr)) > float(np.nanstd(old)):
            by_t[time] = arr
    return sorted(by_t.items())


def _parse_plane_block(chunk: str, *, nx: int, ny: int, nz: int) -> NDArray[np.float64]:
    out = np.full((nz, ny, nx), np.nan, dtype=float)
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
            raw = re.findall(
                r"([+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[Ee][+-]?\d+)?)[a-zA-Z]?",
                mj.group(2),
            )
            for ii, v in enumerate(raw):
                i = i0 + ii
                if 0 <= i < nx:
                    out[k - 1, j - 1, i] = float(v)
    return out
