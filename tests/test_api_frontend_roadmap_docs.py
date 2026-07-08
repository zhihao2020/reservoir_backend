from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "api_frontend_integration_roadmap.md"
SUMMARY_JSON = ROOT / "accuracy_reports" / "api_frontend_integration_roadmap_summary.json"
SUMMARY_MD = ROOT / "accuracy_reports" / "api_frontend_integration_roadmap_summary.md"


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def test_api_frontend_roadmap_doc_exists() -> None:
    assert DOC.exists()


def test_rest_api_roadmap() -> None:
    assert "REST API Roadmap" in _doc()


def test_project_case_run_endpoints() -> None:
    text = _doc()
    assert "GET /projects" in text
    assert "GET /cases/{case_id}" in text
    assert "GET /runs/{run_id}" in text


def test_result_manifest_serving() -> None:
    text = _doc()
    assert "Result Manifest Serving" in text
    assert "GET /results/{result_id}" in text


def test_report_serving() -> None:
    text = _doc()
    assert "Report Serving" in text
    assert "GET /reports/{report_id}" in text


def test_frontend_field_contract() -> None:
    text = _doc()
    assert "Frontend Field Contract" in text
    assert "pressure fields" in text
    assert "water cut curves" in text


def test_auth_security_placeholder() -> None:
    text = _doc()
    assert "Auth / Security Placeholder" in text
    assert "authentication mode" in text


def test_udp_deferred_rationale() -> None:
    text = _doc()
    assert "UDP Deferred Rationale" in text
    assert "UDP remains outside the current integration path" in text


def test_no_implementation_claim() -> None:
    text = _doc()
    assert "No REST service implementation." in text
    assert "No frontend implementation." in text
    assert "No UDP implementation." in text


def test_summary_json_exists() -> None:
    assert SUMMARY_JSON.exists()


def test_summary_markdown_exists() -> None:
    assert SUMMARY_MD.exists()


def test_summary_json_serializable() -> None:
    json.dumps(json.loads(SUMMARY_JSON.read_text(encoding="utf-8")))


def test_summary_contains_non_claims() -> None:
    data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    assert data["success"] is True
    assert "No frontend implementation claim." in data["non_claims"]
