"""EXAMPLE 30-day HZ 1+4 case, monthly dt. Not FIM, not a Jiyang card."""

import csv
import io

from reservoir_backend.comp.case_run import (
    CASE_30DAY,
    DEFAULT_CASE,
    WELL_HISTORY_CSV_COLUMNS,
    WELL_HISTORY_PLACEHOLDER,
    format_metrics,
    load_case_yaml,
    main,
    produce_days_from_schedule,
)
from reservoir_backend.comp.cycle import SECONDS_PER_DAY
from reservoir_backend.comp.step import DT_MIN


def test_30day_case_yaml_is_monthly_example() -> None:
    cfg = load_case_yaml(CASE_30DAY)
    assert cfg["marker"] == "EXAMPLE"
    assert cfg["pattern"] == "hz_1inj4prod"
    assert cfg["n_cycles"] == 1
    assert cfg["fluid"]["eos_yaml"] == "example_c1_c7plus_co2.yaml"
    assert "gem_deck" not in cfg["fluid"]
    assert cfg["fluid"]["components"] == ["C1", "CO2"]
    assert cfg["grid"] == load_case_yaml(DEFAULT_CASE)["grid"]
    scfg = cfg["schedule"]
    assert float(scfg["dt_init_days"]) == 30.0
    assert float(scfg["dt_max_days"]) == 30.0
    total = (
        float(scfg["inject_days"])
        + float(scfg["soak_days"])
        + produce_days_from_schedule(scfg)
    )
    assert abs(total - 30.0) < 20.0 / SECONDS_PER_DAY + 1e-12
    text = CASE_30DAY.read_text(encoding="utf-8")
    assert "monthly" in text.lower()
    assert "not a Jiyang" in text
    assert cfg["output"]["well_history_csv"] == "results/hz_1inj4prod_30day_well_history.csv"


def test_30day_case_run_prints_nsteps_and_dt_min(tmp_path) -> None:
    """One 30-day month: 3 accepted steps, dt < DT_MIN no, fields written."""
    buf = io.StringIO()
    csv_path = tmp_path / "fields.csv"
    wells_path = tmp_path / "wells.csv"
    metrics = main(
        [str(CASE_30DAY), "--fields", str(csv_path), "--wells", str(wells_path)],
        stdout=buf,
    )
    text = buf.getvalue()
    assert "EXAMPLE case: hz_1inj4prod_30day" in text
    assert "not a Jiyang GEM card" in text
    assert "accepted nsteps" in text
    assert "dt < DT_MIN no" in text
    assert "well history csv" in text
    assert metrics["n_cycles"] == 1
    assert metrics["accepted_steps"] == 3
    assert metrics["underflow"] is False
    assert metrics["dt_below_dt_min"] is False
    assert metrics["min_dt_s"] is not None
    assert metrics["min_dt_s"] >= DT_MIN
    assert format_metrics(metrics) in text
    assert csv_path.is_file()
    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert "p" in rows[0] and "z_CO2" in rows[0]
    assert len(rows) == 15
    assert wells_path.is_file()
    assert metrics["well_history_csv"] == str(wells_path)
    with wells_path.open(encoding="utf-8", newline="") as fh:
        well_rows = list(csv.DictReader(fh))
    assert tuple(well_rows[0].keys()) == WELL_HISTORY_CSV_COLUMNS
    assert "rate_mol_s" in well_rows[0] and "bhp_pa" in well_rows[0]
    periods = {row["period"] for row in well_rows}
    wells = {row["well"] for row in well_rows}
    assert periods == {"inject", "soak", "produce"}
    assert wells == {"INJ", "PROD"}
    assert len(well_rows) == 6
    for row in well_rows:
        assert row["q_oil_m3_s"] == WELL_HISTORY_PLACEHOLDER
        assert row["q_gas_m3_s"] == WELL_HISTORY_PLACEHOLDER
        assert row["q_water_m3_s"] == WELL_HISTORY_PLACEHOLDER
        if row["period"] == "inject" and row["well"] == "INJ":
            assert float(row["rate_mol_s"]) > 0.0
            assert row["bhp_pa"] != WELL_HISTORY_PLACEHOLDER
            assert float(row["bhp_pa"]) > 0.0
        if row["period"] == "produce" and row["well"] == "PROD":
            assert row["bhp_pa"] != WELL_HISTORY_PLACEHOLDER
            assert float(row["bhp_pa"]) > 0.0
        if row["period"] == "soak":
            assert float(row["rate_mol_s"]) == 0.0
            assert row["bhp_pa"] == WELL_HISTORY_PLACEHOLDER
