from pathlib import Path

from reservoir_backend.cli.main import main


def test_apply_demo_writes_fields(tmp_path: Path) -> None:
    out = tmp_path / "lab"
    code = main(["apply", "config/lab_apply.yaml", "--demo", "--output", str(out)])
    assert code == 0
    assert (out / "apply.json").is_file()
    assert (out / "k_mean.npy").is_file()
    assert (out / "observations.csv").is_file()
    assert (out / "figures" / "posterior_fields_xz.png").is_file()
