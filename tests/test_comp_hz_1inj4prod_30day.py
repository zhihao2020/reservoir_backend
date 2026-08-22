"""EXAMPLE 30-day HZ 1+4 case, monthly dt. Not FIM, not a Jiyang card."""

import csv
import io

from reservoir_backend.comp.case_run import (
    CASE_30DAY,
    DEFAULT_CASE,
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


def test_30day_case_run_prints_nsteps_and_dt_min(tmp_path) -> None:
    """One 30-day month: 3 accepted steps, dt < DT_MIN no, fields written."""
    buf = io.StringIO()
    csv_path = tmp_path / "fields.csv"
    metrics = main([str(CASE_30DAY), "--fields", str(csv_path)], stdout=buf)
    text = buf.getvalue()
    assert "EXAMPLE case: hz_1inj4prod_30day" in text
    assert "not a Jiyang GEM card" in text
    assert "accepted nsteps" in text
    assert "dt < DT_MIN no" in text
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
