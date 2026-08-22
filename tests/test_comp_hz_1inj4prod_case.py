"""Smoke the EXAMPLE HZ 1+4 YAML case entry. Not FIM, not GEM."""

import ast
import csv
import io
from pathlib import Path

from reservoir_backend.comp.case_run import (
    DEFAULT_CASE,
    format_metrics,
    load_case_yaml,
    main,
    run_example_case,
)


def test_case_yaml_is_example_not_gem() -> None:
    cfg = load_case_yaml(DEFAULT_CASE)
    assert cfg["marker"] == "EXAMPLE"
    assert cfg["pattern"] == "hz_1inj4prod"
    assert cfg["n_cycles"] == 2
    text = DEFAULT_CASE.read_text(encoding="utf-8")
    assert "not a Jiyang GEM" in text
    assert "Not wired into FIM" in text
    assert float(cfg["rock"]["k_matrix_m2"]) == 1.0e-18
    assert float(cfg["rock"]["k_streak_m2"]) == 1.0e-12
    assert cfg["fluid"]["eos_yaml"] == "example_c1_c7plus_co2.yaml"


def test_case_run_module_does_not_import_fi_or_references() -> None:
    path = Path(__file__).resolve().parents[1] / "reservoir_backend" / "comp" / "case_run.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    assert not any(n.startswith("references") or n == "references" for n in names)
    assert not any("solver.fi" in n or n == "reservoir_backend.solver" or n.startswith("reservoir_backend.solver.") for n in names)
    assert not any(n.startswith("reservoir_backend.twin") or n.startswith("reservoir_backend.cli") for n in names)


def test_example_case_prints_metrics(tmp_path: Path) -> None:
    """Run the YAML entry; printed/returned inject/produce ||R|| and nsteps exist."""
    buf = io.StringIO()
    csv_path = tmp_path / "fields.csv"
    metrics = main([str(DEFAULT_CASE), "--fields", str(csv_path)], stdout=buf)
    text = buf.getvalue()
    assert "EXAMPLE case: hz_1inj4prod_two_cycle" in text
    assert "not a Jiyang GEM card" in text
    assert "cycle 1 inject ||R||" in text
    assert "cycle 1 produce ||R||" in text
    assert "cycle 2 inject ||R||" in text
    assert "cycle 2 produce ||R||" in text
    assert "accepted nsteps" in text
    assert "underflow" in text
    assert metrics["n_cycles"] == 2
    assert metrics["accepted_steps"] >= 2
    assert metrics["underflow"] is False
    assert "inject_R" in metrics["cycles"][0]
    assert "produce_R" in metrics["cycles"][0]
    assert format_metrics(metrics) in text
    assert csv_path.is_file()
    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert "p" in rows[0] and "z_CO2" in rows[0]
    assert len(rows) == 15
    assert metrics["fields_csv"] == str(csv_path)
