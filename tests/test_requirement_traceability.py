from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_requirement_traceability_file_exists() -> None:
    assert (ROOT / "specs" / "10_requirement_traceability.md").exists()


def test_requirement_traceability_contains_core_requirements() -> None:
    text = (ROOT / "specs" / "10_requirement_traceability.md").read_text(encoding="utf-8")
    for phrase in ["电阻率反演", "三维压力场", "三维饱和度场", "参数场融合", "full pipeline demo"]:
        assert phrase in text


def test_requirement_traceability_marks_udp_deferred() -> None:
    text = (ROOT / "specs" / "10_requirement_traceability.md").read_text(encoding="utf-8")
    assert "UDP" in text
    assert "Deferred" in text
    assert "通讯协议未知" in text


def test_cpp_migration_spec_exists() -> None:
    assert (ROOT / "specs" / "09_cpp_migration_spec.md").exists()


def test_cpp_migration_spec_says_no_cpp_now() -> None:
    text = (ROOT / "specs" / "09_cpp_migration_spec.md").read_text(encoding="utf-8").lower()
    assert "do not implement c++ now" in text
    assert "pybind11" in text
