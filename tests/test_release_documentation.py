from __future__ import annotations

from pathlib import Path


DOCS = Path("docs")


def test_docs_directory_exists() -> None:
    assert DOCS.exists()


def test_architecture_doc_exists() -> None:
    assert (DOCS / "architecture.md").exists()


def test_numerical_methods_doc_exists() -> None:
    assert (DOCS / "numerical_methods.md").exists()


def test_case_configuration_doc_exists() -> None:
    assert (DOCS / "case_configuration.md").exists()


def test_cli_usage_doc_exists() -> None:
    assert (DOCS / "cli_usage.md").exists()


def test_validation_and_profiling_doc_exists() -> None:
    assert (DOCS / "validation_and_profiling.md").exists()


def test_limitations_and_roadmap_doc_exists() -> None:
    assert (DOCS / "limitations_and_roadmap.md").exists()


def test_release_checklist_doc_exists() -> None:
    assert (DOCS / "release_checklist.md").exists()


def test_module_matrix_doc_exists() -> None:
    assert (DOCS / "module_matrix.md").exists()


def test_readme_mentions_cli() -> None:
    assert "CLI Usage" in _read("README.md")


def test_readme_mentions_finite_volume() -> None:
    text = _read("README.md").lower()
    assert "finite-volume" in text


def test_readme_mentions_combined_transport() -> None:
    text = _read("README.md")
    assert "combined capillary + gravity transport" in text


def test_numerical_methods_mentions_tpfa() -> None:
    assert "TPFA" in _read("docs/numerical_methods.md")


def test_numerical_methods_mentions_explicit_update() -> None:
    text = _read("docs/numerical_methods.md").lower()
    assert "explicit finite-volume update" in text


def test_case_configuration_mentions_all_cases() -> None:
    text = _read("docs/case_configuration.md")
    for case in [
        "demo_case.yaml",
        "multisignal_case.yaml",
        "capillary_case.yaml",
        "capillary_gradient_case.yaml",
        "gravity_case.yaml",
        "combined_case.yaml",
    ]:
        assert case in text


def test_cli_usage_mentions_dry_run() -> None:
    assert "--dry-run" in _read("docs/cli_usage.md")


def test_validation_doc_mentions_585_passed() -> None:
    assert "585 passed" in _read("docs/validation_and_profiling.md")


def test_limitations_mentions_no_black_oil() -> None:
    text = _read("docs/limitations_and_roadmap.md").lower()
    assert "black-oil" in text


def test_release_checklist_mentions_pytest() -> None:
    assert "pytest -q" in _read("docs/release_checklist.md")


def test_requirement_traceability_still_exists() -> None:
    text = _read("specs/10_requirement_traceability.md")
    assert "release candidate documentation" in text


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")
