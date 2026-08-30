"""Two continua of static rock. C_f only changes the fracture rock."""

from __future__ import annotations

from dataclasses import dataclass

from reservoir_backend.exceptions import InvalidPermeability
from reservoir_backend.physics.rock import Rock


@dataclass
class DualRock:
    """Matrix and fracture rocks on the same grid.

    ``matrix.permeability`` (k_m) and both porosities are V1-fixed.
    ``fracture.permeability`` is k_f^eff from C_f.
    """

    matrix: Rock
    fracture: Rock

    def __post_init__(self) -> None:
        if self.matrix.permeability.size != self.fracture.permeability.size:
            raise ValueError("matrix and fracture n_cells must match")

    @property
    def n_cells(self) -> int:
        return int(self.matrix.permeability.size)

    @classmethod
    def from_cf(
        cls,
        n_cells: int,
        *,
        k_matrix_m2: float,
        phi_matrix: float,
        cf_m2: float,
        phi_fracture: float,
    ) -> DualRock:
        """Build uniform continua. ``cf_m2`` is k_f^eff (m²), not a discrete-fracture k."""
        if float(cf_m2) <= 0.0:
            raise InvalidPermeability("C_f must be positive")
        return cls(
            matrix=Rock.uniform(n_cells, k=float(k_matrix_m2), phi=float(phi_matrix)),
            fracture=Rock.uniform(n_cells, k=float(cf_m2), phi=float(phi_fracture)),
        )

    def with_cf(self, cf_m2: float) -> DualRock:
        """Replace only fracture permeability. Matrix rock is unchanged."""
        n = self.n_cells
        return DualRock(
            matrix=self.matrix,
            fracture=Rock.uniform(
                n,
                k=float(cf_m2),
                phi=float(self.fracture.porosity[0]),
                kz=None if self.fracture.kz is None else float(self.fracture.kz[0]),
            ),
        )

    def with_matrix_permeability(self, k_matrix_m2: float) -> DualRock:
        """Replace inter-cell matrix permeability. Transfer still uses its own k_m."""
        n = self.n_cells
        return DualRock(
            matrix=Rock.uniform(
                n,
                k=float(k_matrix_m2),
                phi=float(self.matrix.porosity[0]),
                kz=None if self.matrix.kz is None else float(self.matrix.kz[0]),
            ),
            fracture=self.fracture,
        )
