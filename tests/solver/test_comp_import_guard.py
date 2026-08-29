"""Product tree must not import references/ or banned FIM identifiers."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
PRODUCT = ROOT / "reservoir_backend"
BANNED = (
    "CompositionalMultiphaseFVM",
    "SimulatorFullyImplicit",
    "FIBlackoilModel",
    "AppleyardChop",
)
_IMPORT_REFS = re.compile(r"^\s*(?:from|import)\s+references\b", re.MULTILINE)


def _py_files() -> list[Path]:
    return [p for p in PRODUCT.rglob("*.py") if p.is_file()]


def test_no_import_references() -> None:
    hits = []
    for path in _py_files():
        text = path.read_text(encoding="utf-8")
        if _IMPORT_REFS.search(text):
            hits.append(str(path.relative_to(ROOT)))
    assert hits == []


def test_no_banned_fim_names() -> None:
    hits = []
    for path in _py_files():
        text = path.read_text(encoding="utf-8")
        for name in BANNED:
            if name in text:
                hits.append(f"{path.relative_to(ROOT)}:{name}")
    assert hits == []
