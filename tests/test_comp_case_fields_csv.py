"""EXAMPLE case_run field CSV: per-cell p, z_CO2, and Sw. Not FIM, not GEM."""

import csv
from pathlib import Path

import numpy as np

from reservoir_backend.comp.case_run import FIELD_CSV_COLUMNS, write_fields_csv
from reservoir_backend.comp.step import CompFields
from reservoir_backend.eos import example_eight_component_mixture
from reservoir_backend.grid.cartesian import CartesianGrid


def test_write_fields_csv_has_p_and_z_co2(tmp_path: Path) -> None:
    grid = CartesianGrid.uniform((3.0, 5.0, 1.0), 1.0)
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    z = np.tile(np.array([0.40, 0.60]), (grid.n_cells, 1))
    z[-1, 1] = 0.70
    fields = CompFields(z=z, n=z.copy(), cells=[], p=np.full(grid.n_cells, 5.0e6))
    path = write_fields_csv(tmp_path / "fields.csv", grid, fields, mix)
    assert path.is_file()
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0].keys()) == FIELD_CSV_COLUMNS
    assert FIELD_CSV_COLUMNS == ("cell", "i", "j", "k", "p", "z_CO2", "Sw", "So", "Sg")
    assert "p" in rows[0] and "z_CO2" in rows[0] and "Sw" in rows[0]
    assert "cell" in rows[0] and "i" in rows[0] and "j" in rows[0]
    assert len(rows) == grid.n_cells
    assert float(rows[0]["p"]) == 5.0e6
    assert abs(float(rows[0]["z_CO2"]) - 0.60) < 1e-12
    assert abs(float(rows[-1]["z_CO2"]) - 0.70) < 1e-12
    assert float(rows[0]["Sw"]) == 0.0
    assert int(rows[-1]["cell"]) == grid.n_cells - 1


def test_write_fields_csv_sw_so_sg_sum_to_one(tmp_path: Path) -> None:
    grid = CartesianGrid.uniform((3.0, 5.0, 1.0), 1.0)
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    z = np.tile(np.array([0.40, 0.60]), (grid.n_cells, 1))
    fields = CompFields(z=z, n=z.copy(), cells=[], p=np.full(grid.n_cells, 5.0e6))
    sw = np.full(grid.n_cells, 0.25)
    so = np.full(grid.n_cells, 0.45)
    sg = np.full(grid.n_cells, 0.30)
    path = write_fields_csv(
        tmp_path / "fields.csv", grid, fields, mix, s_water=sw, s_oil=so, s_gas=sg
    )
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert "Sw" in rows[0] and "So" in rows[0] and "Sg" in rows[0]
    assert abs(float(rows[0]["Sw"]) - 0.25) < 1e-12
    for row in rows:
        assert abs(float(row["So"]) + float(row["Sg"]) + float(row["Sw"]) - 1.0) < 1e-12
