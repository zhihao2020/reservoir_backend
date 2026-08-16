"""Laboratory flow ports. Not oilfield Peaceman wells."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.exceptions import InvalidControl
from reservoir_backend.grid.cartesian import CartesianGrid


@dataclass
class FlowPort:
    """A rate- or pressure-controlled inlet / outlet occupying one or more cells."""

    name: str
    role: str  # injector | producer
    control: str  # rate | pressure
    cell_ids: NDArray[np.int64]
    sw_inj: float = 1.0
    wi_multiplier: float = 1.0
    use_productivity: bool = False

    def __post_init__(self) -> None:
        role = str(self.role).strip().lower()
        if role in {"inj", "injection", "injector"}:
            role = "injector"
        elif role in {"prod", "production", "producer"}:
            role = "producer"
        else:
            raise InvalidControl(f"unsupported port role: {self.role}")
        control = str(self.control).strip().lower()
        if control not in {"rate", "pressure"}:
            raise InvalidControl(f"unsupported port control: {self.control}")
        cells = np.unique(np.asarray(self.cell_ids, dtype=np.int64).ravel())
        if cells.size == 0:
            raise InvalidControl(f"port {self.name} has no cells")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "cell_ids", cells)

    @classmethod
    def at_point(
        cls,
        grid: CartesianGrid,
        name: str,
        role: str,
        control: str,
        xyz: tuple[float, float, float],
        radius_m: float = 0.0,
        sw_inj: float = 1.0,
        use_productivity: bool = False,
    ) -> FlowPort:
        """Map a physical port to the containing cell, or cells inside ``radius_m``."""
        if radius_m <= 0.0:
            cells = np.array([grid.locate_cell(*xyz)], dtype=np.int64)
        else:
            centers = grid.cell_centers()
            d2 = np.sum((centers - np.asarray(xyz, dtype=float)) ** 2, axis=1)
            cells = np.nonzero(d2 <= radius_m * radius_m)[0].astype(np.int64)
            if cells.size == 0:
                cells = np.array([grid.locate_cell(*xyz)], dtype=np.int64)
        return cls(
            name=name,
            role=role,
            control=control,
            cell_ids=cells,
            sw_inj=sw_inj,
            use_productivity=use_productivity,
        )

    @classmethod
    def column(
        cls,
        grid: CartesianGrid,
        name: str,
        role: str,
        control: str,
        x: float,
        y: float,
        sw_inj: float = 1.0,
        use_productivity: bool = False,
    ) -> FlowPort:
        """Perforate every layer at ``(x, y)`` so a layered K is actually driven."""
        z_edges = grid.edge_z()
        zs = 0.5 * (z_edges[:-1] + z_edges[1:])
        cells = np.array([grid.locate_cell(x, y, float(z)) for z in zs], dtype=np.int64)
        return cls(
            name=name,
            role=role,
            control=control,
            cell_ids=cells,
            sw_inj=sw_inj,
            use_productivity=use_productivity,
        )


def geometric_wi(grid: CartesianGrid, cell: int, permeability: float, radius_m: float = 0.0) -> float:
    """Small hole connection. Not Peaceman: no r_w log, no skin.

    Units m³ so that ``WI * λ * (p_bhp - p)`` is m³/s.
    Prefer :func:`half_cell_wi` for a cell-face port.
    """
    i, j, k = grid.ijk(int(cell))
    dx, dy, dz = float(grid.dx[i]), float(grid.dy[j]), float(grid.dz[k])
    h = min(dx, dy, dz)
    r = float(radius_m) if radius_m > 0.0 else 0.15 * h
    area = float(np.pi) * r * r
    length = 0.25 * h
    return float(permeability) * area / max(length, 1.0e-12)


def half_cell_wi(grid: CartesianGrid, cell: int, permeability: float) -> float:
    """Half-cell face connection. Not Peaceman: no r_w, no skin, no ln(re/rw).

    Vertical port: TPFA from the cell center to each horizontal face.
    Units m³ so that ``WI * λ * (p_bhp - p)`` is m³/s.
    """
    i, j, k = grid.ijk(int(cell))
    dx, dy, dz = float(grid.dx[i]), float(grid.dy[j]), float(grid.dz[k])
    perm = float(permeability)
    tx = perm * (dy * dz) / max(0.5 * dx, 1.0e-12)
    ty = perm * (dx * dz) / max(0.5 * dy, 1.0e-12)
    return tx + ty


def validate_port_controls(ports: list[FlowPort], control_kinds: dict[str, set[str]]) -> None:
    """A port may not be both rate- and pressure-controlled."""
    for port in ports:
        kinds = control_kinds.get(port.name, set())
        if "rate" in kinds and "pressure" in kinds:
            raise InvalidControl(
                f"port {port.name} cannot have both rate and pressure as controls"
            )
        if port.control == "rate" and "rate" not in kinds:
            raise InvalidControl(f"rate-controlled port {port.name} has no rate series")
        if port.control == "pressure" and "pressure" not in kinds:
            raise InvalidControl(f"pressure-controlled port {port.name} has no pressure series")
