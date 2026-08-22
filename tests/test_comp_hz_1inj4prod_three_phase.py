"""EXAMPLE HZ 1+4 short cycle with aqueous Sw. Not FIM, not a Jiyang card."""

import csv
import io

from reservoir_backend.comp.case_run import (
    CASE_THREE_PHASE,
    DEFAULT_CASE,
    FIELD_CSV_COLUMNS,
    format_metrics,
    load_case_yaml,
    main,
)


def test_three_phase_case_yaml_is_example_with_sw() -> None:
    cfg = load_case_yaml(CASE_THREE_PHASE)
    assert cfg["marker"] == "EXAMPLE"
    assert cfg["pattern"] == "hz_1inj4prod"
    assert cfg["n_cycles"] == 1
    assert cfg["aqueous"] is True
    assert cfg["fluid"]["eos_yaml"] == "example_c1_c7plus_co2.yaml"
    assert "gem_deck" not in cfg["fluid"]
    assert cfg["fluid"]["components"] == ["C1", "CO2"]
    assert float(cfg["fluid"]["s_water"]) == 0.25
    assert cfg["grid"] == load_case_yaml(DEFAULT_CASE)["grid"]
    scfg = cfg["schedule"]
    two = load_case_yaml(DEFAULT_CASE)["schedule"]
    assert float(scfg["inject_days"]) == float(two["inject_days"])
    assert float(scfg["soak_days"]) == float(two["soak_days"])
    assert float(scfg["produce_seconds"]) == float(two["produce_seconds"])
    text = CASE_THREE_PHASE.read_text(encoding="utf-8")
    assert "EXAMPLE" in text
    assert "not a Jiyang" in text
    assert "s_water" in text
    assert "gem_deck" not in text.lower()


def test_three_phase_case_run_writes_sw_and_accepted_steps(tmp_path) -> None:
    """Short three-phase cycle: Sw on CSV, accepted steps, So+Sg+Sw=1."""
    buf = io.StringIO()
    csv_path = tmp_path / "fields.csv"
    metrics = main([str(CASE_THREE_PHASE), "--fields", str(csv_path)], stdout=buf)
    text = buf.getvalue()
    assert "EXAMPLE case: hz_1inj4prod_three_phase" in text
    assert "not a Jiyang GEM card" in text
    assert "accepted nsteps" in text
    assert metrics["n_cycles"] == 1
    assert metrics["accepted_steps"] >= 3
    assert metrics["underflow"] is False
    r0, r1 = metrics["cycles"][0]["inject_R"]
    assert r0 > 0.0
    assert r1 < r0
    assert format_metrics(metrics) in text
    assert csv_path.is_file()
    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0].keys()) == FIELD_CSV_COLUMNS
    assert "p" in rows[0] and "z_CO2" in rows[0] and "Sw" in rows[0]
    assert len(rows) == 15
    for row in rows:
        so = float(row["So"])
        sg = float(row["Sg"])
        sw = float(row["Sw"])
        assert 0.0 <= sw <= 1.0
        assert abs(so + sg + sw - 1.0) < 1e-9
    assert metrics["fields_csv"] == str(csv_path)
