from pathlib import Path

from reservoir_backend.cli.main import main


def test_validate_lab_case(tmp_path: Path) -> None:
    code = main(["validate", "config/lab_30cm.yaml", "--output", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "validate.json").is_file()


def test_harness_journal_empty() -> None:
    code = main(["harness", "journal", "--threshold", "1.0"])
    assert code == 0
