#!/usr/bin/env python3
"""Check that path-like references in active docs exist on disk.

Scans markdown under docs/, specs/, README.md, STATUS.md, and
requirements/ for backtick-wrapped paths that look like repository
files or directories. Historical archive pages are skipped.

This is a lightweight hard check to catch doc drift after package
layout changes. Semantic consistency still relies on QMD search plus
human review.

Usage:
    python scripts/check_doc_code_consistency.py
    python scripts/check_doc_code_consistency.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DOC_GLOBS = (
    "README.md",
    "STATUS.md",
    "docs/**/*.md",
    "specs/**/*.md",
    "requirements/**/*.md",
)

SKIP_PARTS = {"archive"}

# Historical audit snapshots; path drift is expected.
SKIP_DOCS = {"docs/digital_twin_repository_audit.md"}

# Backtick paths that look like repo-relative file/dir references.
PATH_RE = re.compile(
    r"`("
    r"(?:reservoir_backend|tests|scripts|examples|benchmarks|docs|specs|"
    r"requirements|references|accuracy_reports|validation_reports|profiling_reports|"
    r"results|validation)"
    r"/[^`\s]+"
    r")`"
)

# Patterns that intentionally use wildcards / placeholders.
WILDCARD_MARKERS = ("*", "...", "<", ">", "{", "}")

# Known non-literal fragments to ignore (prose / partial paths).
IGNORE_EXACT = {
    "results/",
    "results/examples/two_layer/",
    "accuracy_reports/",
    "validation_reports/",
    "profiling_reports/",
    "docs/",
    "specs/",
    "tests/",
    "scripts/",
    "examples/",
    "benchmarks/",
    "requirements/",
    "references/",
    "reservoir_backend/",
    "validation/",
    "experiments/",
    "examples/lab_v1/",
}


def iter_doc_files() -> list[Path]:
    files: list[Path] = []
    for pattern in DOC_GLOBS:
        for path in REPO_ROOT.glob(pattern):
            if not path.is_file():
                continue
            rel_parts = path.relative_to(REPO_ROOT).parts
            if any(part in SKIP_PARTS for part in rel_parts):
                continue
            if path.relative_to(REPO_ROOT).as_posix() in SKIP_DOCS:
                continue
            files.append(path)
    return sorted(set(files))


def path_exists(ref: str) -> bool:
    """Return True if ref exists, allowing mild suffix flexibility."""
    # Prose shorthand: path.json/md means path.json or path.md
    if ref.endswith(".json/md"):
        base = ref[: -len(".json/md")]
        return path_exists(f"{base}.json") or path_exists(f"{base}.md")
    if ref.endswith(".*/"):
        return False

    candidate = REPO_ROOT / ref
    if candidate.exists():
        return True
    # Directory referenced without trailing slash is fine if parent exists as dir.
    if ref.endswith("/"):
        return (REPO_ROOT / ref.rstrip("/")).is_dir()
    # Allow referring to a Python package via import-style trailing module without .py
    py_file = REPO_ROOT / f"{ref}.py"
    if py_file.exists():
        return True
    init_file = REPO_ROOT / ref / "__init__.py"
    if init_file.exists():
        return True
    return False


def should_skip(ref: str) -> bool:
    if ref in IGNORE_EXACT:
        return True
    if ref.startswith("results/"):
        return True
    if any(marker in ref for marker in WILDCARD_MARKERS):
        return True
    # Ignore pure extension globs like accuracy_reports/foo_summary.*
    if ref.endswith(".*") or ref.endswith(".*/"):
        return True
    # Ignore incomplete fragment paths ending with /
    if ref.count("/") == 0:
        return True
    return False


def check_file(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rel_doc = path.relative_to(REPO_ROOT).as_posix()
    findings: list[dict[str, object]] = []
    for match in PATH_RE.finditer(text):
        ref = match.group(1).strip()
        # Strip trailing punctuation sometimes glued into backticks
        ref = ref.rstrip(".,;:)")
        if should_skip(ref):
            continue
        if path_exists(ref):
            continue
        line_no = text.count("\n", 0, match.start()) + 1
        findings.append(
            {
                "doc": rel_doc,
                "line": line_no,
                "path": ref,
            }
        )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON findings",
    )
    args = parser.parse_args(argv)

    docs = iter_doc_files()
    findings: list[dict[str, object]] = []
    for doc in docs:
        findings.extend(check_file(doc))

    if args.json:
        print(json.dumps({"checked_docs": len(docs), "missing": findings}, indent=2))
    else:
        print(f"Checked {len(docs)} active markdown files under {REPO_ROOT}")
        if not findings:
            print("OK: no missing path references found.")
        else:
            print(f"Found {len(findings)} missing path reference(s):")
            for item in findings:
                print(f"  {item['doc']}:{item['line']}: `{item['path']}`")

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
