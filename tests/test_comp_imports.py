"""Import-graph guard: comp/ must stay off FIM and off references/."""

import ast
import sys
from pathlib import Path

import reservoir_backend.comp


def _iter_comp_py() -> list[Path]:
    root = Path(reservoir_backend.comp.__file__).resolve().parent
    return sorted(root.rglob("*.py"))


def _imported_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names.append(mod)
            for alias in node.names:
                names.append(f"{mod}.{alias.name}" if mod else alias.name)
    return names


def test_comp_does_not_import_fi_or_references() -> None:
    forbidden_hits: list[str] = []
    for path in _iter_comp_py():
        for name in _imported_names(path):
            if name.startswith("references") or name == "references":
                forbidden_hits.append(f"{path.name}: {name}")
            if name.startswith("reservoir_backend.solver.fi") or name.endswith(".solver.fi"):
                forbidden_hits.append(f"{path.name}: {name}")
            if name == "reservoir_backend.solver" or name.startswith("reservoir_backend.solver."):
                forbidden_hits.append(f"{path.name}: {name}")
    assert forbidden_hits == []
    assert "reservoir_backend.solver.fi" not in sys.modules
