from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "black_oil_pvt_architecture.md"
SUMMARY_JSON = ROOT / "accuracy_reports" / "black_oil_pvt_architecture_summary.json"
SUMMARY_MD = ROOT / "accuracy_reports" / "black_oil_pvt_architecture_summary.md"


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def test_black_oil_doc_exists() -> None:
    assert DOC.exists()


def test_black_oil_doc_contains_bo_bw_bg() -> None:
    text = _doc()
    assert "Bo" in text
    assert "Bw" in text
    assert "Bg" in text


def test_black_oil_doc_contains_rs_rv() -> None:
    text = _doc()
    assert "Rs" in text
    assert "Rv" in text


def test_black_oil_doc_contains_bubble_point() -> None:
    assert "bubble point" in _doc().lower()


def test_black_oil_doc_contains_phase_appearance_disappearance() -> None:
    text = _doc().lower()
    assert "phase appearance" in text
    assert "phase disappearance" in text


def test_black_oil_doc_contains_surface_rates() -> None:
    assert "surface rates" in _doc().lower()


def test_black_oil_doc_contains_well_controls() -> None:
    assert "well controls" in _doc().lower()


def test_black_oil_doc_contains_schedule_restart_report_step() -> None:
    text = _doc().lower()
    assert "schedule" in text
    assert "restart" in text
    assert "report step" in text


def test_black_oil_doc_contains_state_variables() -> None:
    assert "state variables" in _doc().lower()


def test_black_oil_doc_contains_limitations() -> None:
    assert "Limitations" in _doc()


def test_black_oil_doc_contains_non_claims() -> None:
    assert "Non-Claims" in _doc()


def test_black_oil_doc_does_not_implement_solver() -> None:
    text = _doc()
    assert "No black-oil solver implemented." in text
    assert "No PVT table parser implemented." in text


def test_black_oil_doc_does_not_claim_equivalence() -> None:
    text = _doc()
    assert "commercial simulator equivalence" in text
    assert "OPM Flow" in text


def test_black_oil_summary_json_exists() -> None:
    assert SUMMARY_JSON.exists()


def test_black_oil_summary_markdown_exists() -> None:
    assert SUMMARY_MD.exists()


def test_black_oil_summary_json_serializable() -> None:
    json.dumps(json.loads(SUMMARY_JSON.read_text(encoding="utf-8")))


def test_black_oil_summary_contains_next_steps() -> None:
    data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    assert data["success"] is True
    assert data["next_steps"]
