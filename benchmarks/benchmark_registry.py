"""Unified benchmark registry for completed hardening reports.

The registry reads existing benchmark summary JSON files and open-source
reference fixtures. It does not run solvers, alter benchmark behavior, parse
upstream decks, or create runtime dependencies on OPM/MRST.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.reference_case_loader import load_open_source_reference_cases


BENCHMARK_SPECS = {
    "saturation_inversion_benchmark": {
        "file": "saturation_inversion_benchmark_summary.json",
        "module_id": "M2",
        "task_id": "TASK-046",
        "requirement": "saturation inversion",
    },
    "pressure_solver_benchmark": {
        "file": "pressure_solver_benchmark_summary.json",
        "module_id": "M3",
        "task_id": "TASK-047",
        "requirement": "pressure field reconstruction",
    },
    "saturation_transport_benchmark": {
        "file": "saturation_transport_benchmark_summary.json",
        "module_id": "M4",
        "task_id": "TASK-048",
        "requirement": "saturation transport",
    },
    "capillary_gravity_benchmark": {
        "file": "capillary_gravity_benchmark_summary.json",
        "module_id": "M4",
        "task_id": "TASK-049",
        "requirement": "capillary and gravity transport",
    },
    "three_phase_benchmark": {
        "file": "three_phase_benchmark_summary.json",
        "module_id": "M4",
        "task_id": "TASK-050",
        "requirement": "simplified three-phase WOG",
    },
    "parameter_fusion_benchmark": {
        "file": "parameter_fusion_benchmark_summary.json",
        "module_id": "M5",
        "task_id": "TASK-051",
        "requirement": "parameter field fusion",
    },
}

FORBIDDEN_CLAIMS = [
    "full SPE1 reproduction",
    "full SPE10 reproduction",
    "OPM Flow equivalent",
    "MRST equivalent",
    "commercial simulator equivalent",
    "black-oil validation",
    "complete black-oil model",
    "history matching implemented",
    "automatic calibration implemented",
]


def load_benchmark_summary(path: str | Path) -> dict:
    """Load one benchmark summary JSON file."""
    target = Path(path)
    if not target.exists():
        return {
            "missing": True,
            "path": str(target),
            "success": False,
            "warnings": [f"missing benchmark summary: {target}"],
        }
    data = json.loads(target.read_text(encoding="utf-8"))
    data["missing"] = False
    data["path"] = str(target)
    return data


def collect_benchmark_summaries(report_dir: str | Path = "accuracy_reports") -> dict[str, dict]:
    """Collect required benchmark summaries from a report directory."""
    root = Path(report_dir)
    return {
        benchmark_id: load_benchmark_summary(root / spec["file"])
        for benchmark_id, spec in BENCHMARK_SPECS.items()
    }


def load_open_source_reference_metadata() -> list[dict]:
    """Load already-extracted open-source reference metadata."""
    fixture = load_open_source_reference_cases()
    references = []
    for case in fixture.get("cases", []):
        source = case.get("source", {})
        name = case.get("case_name", "")
        references.append(
            {
                "reference_name": name,
                "project": source.get("project"),
                "path": source.get("path"),
                "url": source.get("url"),
                "reference_type": classify_reference_type(
                    {
                        "case_name": name,
                        "source": f"{source.get('project', '')} {source.get('path', '')}",
                        "is_exact_reproduction": False,
                        "limitations": [case.get("adapted_benchmark_use", "")],
                    }
                ),
                "is_exact_reproduction": False,
                "runtime_dependency": False,
                "adapted_benchmark_use": case.get("adapted_benchmark_use", ""),
            }
        )
    return references


def classify_validation_level(case: dict) -> str:
    """Classify a benchmark case into the project validation-level taxonomy."""
    name = str(case.get("case_name", "")).lower()
    source = str(case.get("source", "")).lower()
    text = f"{name} {source}"
    if "manufactured" in text:
        return "manufactured_solution"
    if "analytical" in text or "archie_formula" in text:
        return "analytical"
    if "metadata" in text or "property" in text or "sanity" in name:
        return "property_metadata_sanity"
    if any(token in text for token in ("smoothing", "segregation", "monotonicity", "buckley", "areal", "front", "confidence")):
        return "trend_validation"
    if any(token in text for token in ("stability", "boundedness", "cfl")):
        return "stability_validation"
    if "opm" in text or "mrst" in text:
        return "adapted_open_source_reference"
    return "diagnostic_sanity"


def classify_reference_type(case: dict) -> str:
    """Classify exact/adapted/reference-only status for a case."""
    if bool(case.get("is_exact_reproduction", False)):
        return "exact reproduction"
    name = str(case.get("case_name", "")).lower()
    source = str(case.get("source", "")).lower()
    limitations = " ".join(str(item).lower() for item in case.get("limitations", []))
    text = f"{name} {source} {limitations}"
    if "reference note" in text or "reference context" in text or "simpleincomptpfa" in text:
        return "reference context only"
    if "metadata" in text or "property" in text or "sanity" in text:
        return "property metadata sanity only"
    if "opm" in text or "mrst" in text or "adapted" in text:
        return "adapted reference"
    return "internal benchmark"


def build_benchmark_registry(report_dir: str | Path = "accuracy_reports") -> dict:
    """Build the unified benchmark registry from existing summaries."""
    report_root = Path(report_dir)
    summaries = collect_benchmark_summaries(report_root)
    benchmark_entries = []
    all_cases = []
    missing = 0
    for benchmark_id, data in summaries.items():
        spec = BENCHMARK_SPECS[benchmark_id]
        if data.get("missing"):
            missing += 1
            entry = _missing_entry(benchmark_id, spec, report_root / spec["file"])
            benchmark_entries.append(entry)
            continue
        cases = _cases_for_summary(benchmark_id, data, spec)
        all_cases.extend(cases)
        validation_levels = sorted({case["validation_level"] for case in cases})
        reference_types = sorted({case["reference_type"] for case in cases})
        entry = {
            "benchmark_id": benchmark_id,
            "benchmark_name": data.get("benchmark_name", benchmark_id),
            "module_id": spec["module_id"],
            "task_id": spec["task_id"],
            "summary_json_path": str(report_root / spec["file"]),
            "summary_markdown_path": str(report_root / spec["file"].replace(".json", ".md")),
            "success": bool(data.get("success", False)),
            "num_cases": int(data.get("num_cases", len(cases))),
            "num_passed": int(data.get("num_passed", sum(case["success"] for case in cases))),
            "num_failed": int(data.get("num_failed", sum(not case["success"] for case in cases))),
            "main_metrics": _main_metrics(data),
            "validation_levels": validation_levels,
            "reference_types": reference_types,
            "has_nan": bool(data.get("has_nan", False)),
            "has_inf": bool(data.get("has_inf", False)),
            "limitations": sorted({limitation for case in cases for limitation in case["limitations"]}),
            "cases": cases,
        }
        benchmark_entries.append(entry)
    overclaim_warnings = _scan_overclaims(benchmark_entries)
    success = (
        missing == 0
        and all(entry["success"] for entry in benchmark_entries)
        and not any(entry["has_nan"] or entry["has_inf"] for entry in benchmark_entries)
        and len(overclaim_warnings) == 0
    )
    modules = sorted({entry["module_id"] for entry in benchmark_entries} | {"M8"})
    registry = {
        "benchmark_registry_name": "unified_benchmark_registry",
        "success": bool(success),
        "num_benchmark_summaries": len(benchmark_entries),
        "num_benchmark_cases": len(all_cases),
        "num_passed_cases": int(sum(case["success"] for case in all_cases)),
        "num_failed_cases": int(sum(not case["success"] for case in all_cases)),
        "num_missing_summaries": int(missing),
        "modules_covered": modules,
        "requirements_covered": sorted({spec["requirement"] for spec in BENCHMARK_SPECS.values()} | {"benchmark registry"}),
        "benchmarks": benchmark_entries,
        "open_source_references": load_open_source_reference_metadata(),
        "overclaim_warnings": overclaim_warnings,
        "limitations": [
            "Registry reads existing benchmark summaries and reference fixtures only.",
            "OPM/MRST materials are adapted references or context only, with no runtime dependency.",
            "Registry does not claim full SPE1/SPE10 reproduction, OPM Flow equivalence, MRST integration, commercial simulator equivalence, or black-oil validation.",
        ],
        "recommendations": [
            "Keep registry generation as a required evidence index after benchmark hardening stages.",
            "Add future benchmark summaries to BENCHMARK_SPECS rather than scattering independent validation reports.",
            "Preserve exact/adapted/reference-only wording in reports and docs.",
        ],
    }
    return _jsonable(registry)


def write_benchmark_registry_reports(registry: dict, output_dir: str | Path = "accuracy_reports") -> None:
    """Write registry JSON and Markdown reports."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "benchmark_registry_summary.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    lines = [
        "# Benchmark Registry Summary",
        "",
        f"- success: {registry['success']}",
        f"- num_benchmark_summaries: {registry['num_benchmark_summaries']}",
        f"- num_benchmark_cases: {registry['num_benchmark_cases']}",
        f"- num_passed_cases: {registry['num_passed_cases']}",
        f"- num_failed_cases: {registry['num_failed_cases']}",
        f"- num_missing_summaries: {registry['num_missing_summaries']}",
        f"- modules_covered: {', '.join(registry['modules_covered'])}",
        "",
        "## Summary Table",
        "",
        "| Benchmark | Module | Task | Success | Cases | Validation levels | Reference types |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in registry["benchmarks"]:
        lines.append(
            "| {benchmark_id} | {module_id} | {task_id} | {success} | {num_cases} | {levels} | {refs} |".format(
                benchmark_id=entry["benchmark_id"],
                module_id=entry["module_id"],
                task_id=entry["task_id"],
                success=entry["success"],
                num_cases=entry["num_cases"],
                levels=", ".join(entry["validation_levels"]),
                refs=", ".join(entry["reference_types"]),
            )
        )
    lines.extend(["", "## Open-Source References", ""])
    for reference in registry["open_source_references"]:
        lines.append(
            f"- {reference['reference_name']}: {reference['project']} `{reference['path']}`; "
            f"type={reference['reference_type']}; runtime_dependency={reference['runtime_dependency']}; "
            f"exact_reproduction={reference['is_exact_reproduction']}"
        )
    lines.extend(["", "## Limitations", ""])
    for limitation in registry["limitations"]:
        lines.append(f"- {limitation}")
    lines.extend(["", "## Overclaim Warnings", ""])
    if registry["overclaim_warnings"]:
        for warning in registry["overclaim_warnings"]:
            lines.append(f"- {warning['affected_file_or_entry']}: {warning['claim']}; {warning['recommended_fix']}")
    else:
        lines.append("- None")
    (root / "benchmark_registry_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark_registry(output_dir: str | Path = "accuracy_reports") -> dict:
    """Build and write the unified benchmark registry."""
    registry = build_benchmark_registry(output_dir)
    write_benchmark_registry_reports(registry, output_dir)
    return registry


def _missing_entry(benchmark_id: str, spec: dict, path: Path) -> dict:
    return {
        "benchmark_id": benchmark_id,
        "benchmark_name": benchmark_id,
        "module_id": spec["module_id"],
        "task_id": spec["task_id"],
        "summary_json_path": str(path),
        "summary_markdown_path": str(path.with_suffix(".md")),
        "success": False,
        "num_cases": 0,
        "num_passed": 0,
        "num_failed": 0,
        "main_metrics": {},
        "validation_levels": [],
        "reference_types": [],
        "has_nan": False,
        "has_inf": False,
        "limitations": ["summary missing"],
        "cases": [],
        "warnings": [f"missing benchmark summary: {path}"],
    }


def _cases_for_summary(benchmark_id: str, data: dict, spec: dict) -> list[dict]:
    if isinstance(data.get("cases"), list):
        raw_cases = data["cases"]
    elif benchmark_id == "saturation_inversion_benchmark":
        raw_cases = _saturation_inversion_cases(data)
    else:
        raw_cases = []
    cases = []
    for raw in raw_cases:
        case = {
            "case_name": raw.get("case_name", "unnamed_case"),
            "module_id": spec["module_id"],
            "validation_level": classify_validation_level(raw),
            "reference_type": classify_reference_type(raw),
            "source": raw.get("source", "internal benchmark"),
            "is_exact_reproduction": bool(raw.get("is_exact_reproduction", False)),
            "success": bool(raw.get("success", False)),
            "key_metrics": raw.get("key_metrics", _main_metrics(raw)),
            "limitations": list(raw.get("limitations", [])),
            "warnings": list(raw.get("warnings", [])),
        }
        cases.append(_jsonable(case))
    return cases


def _saturation_inversion_cases(data: dict) -> list[dict]:
    return [
        {
            "case_name": "archie_formula_analytical",
            "source": "internal Archie analytical formula",
            "is_exact_reproduction": False,
            "success": bool(data.get("archie_formula_error", 1.0) <= 1.0e-12),
            "key_metrics": {"archie_formula_error": data.get("archie_formula_error")},
            "limitations": ["analytical Archie law only; not commercial petrophysical interpretation"],
            "warnings": [],
        },
        {
            "case_name": "noise_sensitivity",
            "source": "internal saturation inversion noise sensitivity",
            "is_exact_reproduction": False,
            "success": bool(data.get("noise_sensitivity", {}).get("bounded", False)),
            "key_metrics": data.get("noise_sensitivity", {}),
            "limitations": ["synthetic noise sensitivity only"],
            "warnings": [],
        },
        {
            "case_name": "uncertainty_weighted_fusion",
            "source": "internal inversion fusion diagnostic",
            "is_exact_reproduction": False,
            "success": bool(np.isfinite(data.get("fusion_error", np.nan))),
            "key_metrics": {"fusion_error": data.get("fusion_error")},
            "limitations": ["deterministic confidence/uncertainty fusion; no Bayesian inversion"],
            "warnings": [],
        },
        {
            "case_name": "clipping_report",
            "source": "internal inversion clipping diagnostic",
            "is_exact_reproduction": False,
            "success": bool(data.get("clipping_checked", False)),
            "key_metrics": {"clipping_checked": data.get("clipping_checked")},
            "limitations": ["clipping report only; no automatic calibration"],
            "warnings": [],
        },
    ]


def _main_metrics(data: dict) -> dict:
    excluded = {
        "benchmark_name",
        "success",
        "num_cases",
        "num_passed",
        "num_failed",
        "cases",
        "warnings",
        "recommendations",
        "has_nan",
        "has_inf",
        "open_source_references_used",
        "missing",
        "path",
    }
    return {key: value for key, value in data.items() if key not in excluded}


def _scan_overclaims(entries: list[dict]) -> list[dict]:
    text_targets: list[tuple[str, str]] = []
    for entry in entries:
        text_targets.append((entry["benchmark_id"], json.dumps(entry, ensure_ascii=False)))
    doc_paths = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "numerical_accuracy.md",
        REPO_ROOT / "docs" / "limitations_and_roadmap.md",
        REPO_ROOT / "docs" / "function_benchmark_matrix.md",
        REPO_ROOT / "docs" / "module_matrix.md",
    ]
    for path in doc_paths:
        if path.exists():
            text_targets.append((str(path.relative_to(REPO_ROOT)), path.read_text(encoding="utf-8")))
    warnings = []
    for target, text in text_targets:
        for claim in FORBIDDEN_CLAIMS:
            for match in re.finditer(re.escape(claim), text, flags=re.IGNORECASE):
                if _is_negated(text, match.start()):
                    continue
                warnings.append(
                    {
                        "affected_file_or_entry": target,
                        "claim": claim,
                        "recommended_fix": "Rephrase as a non-goal or adapted/reference-only limitation.",
                    }
                )
    return warnings


def _is_negated(text: str, index: int) -> bool:
    context = text[max(0, index - 80) : index].lower()
    negators = [
        "no ",
        "not ",
        "without ",
        "does not claim ",
        "do not claim ",
        "is not ",
        "不是",
        "不声称",
        "不得声称",
        "未",
    ]
    return any(negator in context for negator in negators)


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


if __name__ == "__main__":
    print(json.dumps(run_benchmark_registry(), indent=2))
