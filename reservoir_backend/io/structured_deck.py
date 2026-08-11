"""Self-contained loader for a small structured-deck keyword subset.

Reads industry-style plain-text decks used as *offline fixtures only*.
Does not import or execute any third-party simulator code. Public API names
are project-local and intentionally distinct from external libraries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.core.grid import Grid3D

# File keywords recognized in the *input text* (not Python public names).
_FILE_KEYWORDS = frozenset(
    {
        "DIMENS",
        "SPECGRID",
        "DX",
        "DY",
        "DZ",
        "PORO",
        "PERMX",
        "PERMY",
        "PERMZ",
        "ACTNUM",
    }
)

_FT_TO_M = 0.3048
# Include Eclipse-style repetitions such as 300*1000 in a single token.
_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|"
    r"(?:\d+\*)?[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|"
    r"\*|"
    r"/"
)


@dataclass(frozen=True)
class StructuredDeckBundle:
    """Parsed structured mesh plus optional property arrays."""

    grid: Grid3D
    porosity_field: NDArray[np.float64] | None = None
    permeability_x_md: NDArray[np.float64] | None = None
    permeability_y_md: NDArray[np.float64] | None = None
    permeability_z_md: NDArray[np.float64] | None = None
    source_path: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)


def load_structured_deck(
    path: str | Path,
    *,
    convert_length_ft_to_m: bool = False,
) -> StructuredDeckBundle:
    """Load a structured orthogonal mesh and basic properties from a deck file.

    Supported file keywords: DIMENS/SPECGRID, DX/DY/DZ, PORO, PERMX/Y/Z, ACTNUM.
    Corner-point geometry and schedule sections are ignored.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"structured deck not found: {file_path}. "
            "If this is under references/upstream, run: "
            "git submodule update --init --depth 1"
        )

    text = file_path.read_text(encoding="utf-8", errors="replace")
    sections = _extract_keyword_bodies(text)
    notes: list[str] = []

    dims = sections.get("DIMENS") or sections.get("SPECGRID")
    if dims is None:
        raise ValueError("deck is missing DIMENS or SPECGRID")
    nx, ny, nz = (int(v) for v in dims[:3])
    if min(nx, ny, nz) <= 0:
        raise ValueError("deck dimensions must be positive")

    n_cells = nx * ny * nz
    spacing_i = _axis_spacing_from_values(sections.get("DX"), n_cells, nx, ny, nz, axis="i")
    spacing_j = _axis_spacing_from_values(sections.get("DY"), n_cells, nx, ny, nz, axis="j")
    spacing_k = _axis_spacing_from_values(sections.get("DZ"), n_cells, nx, ny, nz, axis="k")

    length_scale = _FT_TO_M if convert_length_ft_to_m else 1.0
    if convert_length_ft_to_m:
        notes.append("converted DX/DY/DZ from feet to metres using 0.3048")
        spacing_i = spacing_i * length_scale
        spacing_j = spacing_j * length_scale
        spacing_k = spacing_k * length_scale

    active_mask = None
    if "ACTNUM" in sections:
        act = np.asarray(_expand_numeric_tokens(sections["ACTNUM"], n_cells), dtype=float)
        active_mask = act.reshape((nz, ny, nx)) > 0.0
        notes.append("applied ACTNUM active mask")

    grid = Grid3D(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=spacing_i,
        dy=spacing_j,
        dz=spacing_k,
        active_mask=active_mask,
    )

    porosity = _optional_property(sections, "PORO", n_cells, nz, ny, nx)
    permx = _optional_property(sections, "PERMX", n_cells, nz, ny, nx)
    permy = _optional_property(sections, "PERMY", n_cells, nz, ny, nx)
    permz = _optional_property(sections, "PERMZ", n_cells, nz, ny, nx)

    if porosity is None:
        notes.append("PORO not present")
    if permx is None:
        notes.append("PERMX not present")

    return StructuredDeckBundle(
        grid=grid,
        porosity_field=porosity,
        permeability_x_md=permx,
        permeability_y_md=permy,
        permeability_z_md=permz,
        source_path=str(file_path),
        notes=tuple(notes),
    )


def _optional_property(
    sections: dict[str, list[str]],
    key: str,
    n_cells: int,
    nz: int,
    ny: int,
    nx: int,
) -> NDArray[np.float64] | None:
    if key not in sections:
        return None
    values = _expand_numeric_tokens(sections[key], n_cells)
    return np.asarray(values, dtype=float).reshape((nz, ny, nx))


def _axis_spacing_from_values(
    tokens: list[str] | None,
    n_cells: int,
    nx: int,
    ny: int,
    nz: int,
    *,
    axis: str,
) -> NDArray[np.float64]:
    if tokens is None:
        # Default unit spacing when the deck omits an axis size keyword.
        count = {"i": nx, "j": ny, "k": nz}[axis]
        return np.ones(count, dtype=float)

    values = np.asarray(_expand_numeric_tokens(tokens, n_cells), dtype=float)
    if values.size == 1:
        count = {"i": nx, "j": ny, "k": nz}[axis]
        return np.full(count, float(values[0]), dtype=float)

    shaped = values.reshape((nz, ny, nx))
    if axis == "i":
        # Collapse to per-i spacing if constant across j,k; else mean.
        return _reduce_axis_spacing(shaped, axis=2, expected=nx)
    if axis == "j":
        return _reduce_axis_spacing(shaped, axis=1, expected=ny)
    return _reduce_axis_spacing(shaped, axis=0, expected=nz)


def _reduce_axis_spacing(
    shaped: NDArray[np.float64],
    *,
    axis: int,
    expected: int,
) -> NDArray[np.float64]:
    # Move target axis to the end, then take mean over the others.
    moved = np.moveaxis(shaped, axis, -1)
    reduced = moved.reshape(-1, expected).mean(axis=0)
    # If each slab is constant, mean is exact; otherwise still a tensor approximation.
    return np.asarray(reduced, dtype=float)


def _extract_keyword_bodies(text: str) -> dict[str, list[str]]:
    """Map uppercase file keywords to token lists (excluding the keyword itself)."""
    # Strip comments (-- ...)
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        if "--" in line:
            line = line[: line.index("--")]
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)

    tokens = _TOKEN_RE.findall(cleaned)
    sections: dict[str, list[str]] = {}
    current: str | None = None
    body: list[str] = []

    def flush() -> None:
        nonlocal current, body
        if current is not None:
            sections[current] = body
        current = None
        body = []

    for token in tokens:
        upper = token.upper()
        if upper in _FILE_KEYWORDS:
            flush()
            current = upper
            body = []
            continue
        if current is None:
            continue
        if token == "/":
            flush()
            continue
        body.append(token)

    flush()
    return sections


def _expand_numeric_tokens(tokens: Iterable[str], expected: int) -> list[float]:
    """Expand Eclipse-style `N*value` repetition into a flat float list."""
    values: list[float] = []
    token_list = list(tokens)
    i = 0
    while i < len(token_list):
        tok = token_list[i]
        if tok == "*":
            raise ValueError("orphaned '*' in deck numeric list")
        if "*" in tok and not tok.startswith("*"):
            # form N*value in one token
            count_str, value_str = tok.split("*", 1)
            count = int(count_str)
            value = float(value_str)
            values.extend([value] * count)
            i += 1
            continue
        if i + 2 < len(token_list) and token_list[i + 1] == "*":
            count = int(tok)
            value = float(token_list[i + 2])
            values.extend([value] * count)
            i += 3
            continue
        values.append(float(tok))
        i += 1

    if len(values) == 1 and expected > 1:
        return values * expected
    if len(values) != expected:
        # Allow shorter lists only when a single scalar was intended (handled above).
        if len(values) < expected and expected % len(values) == 0:
            reps = expected // len(values)
            return values * reps
        raise ValueError(f"expected {expected} numeric values, got {len(values)}")
    return values
