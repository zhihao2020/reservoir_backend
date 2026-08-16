"""Attempt tree: keep / prune / backtrack. Breakthroughs are score jumps."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_JOURNAL = (
    Path(__file__).resolve().parents[3] / "black_oil" / "validation" / "cmg_harness" / "journal" / "attempts.jsonl"
)


@dataclass
class Attempt:
    id: str
    parent: str | None
    case: str
    knobs: dict
    probe: str
    scores: dict
    J: float
    decision: str  # keep | prune | backtrack
    reason: str
    wave: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class Journal:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or DEFAULT_JOURNAL)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[Attempt] = []
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                raw = json.loads(line)
                self.rows.append(Attempt(**raw))

    def next_id(self) -> str:
        return f"t{len(self.rows) + 1:03d}"

    def append(self, attempt: Attempt) -> Attempt:
        self.rows.append(attempt)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(attempt.as_dict()) + "\n")
        return attempt

    def by_id(self, aid: str) -> Attempt | None:
        for row in self.rows:
            if row.id == aid:
                return row
        return None

    def children(self, aid: str) -> list[Attempt]:
        return [r for r in self.rows if r.parent == aid]

    def best(self, *, case: str | None = None, decision: str = "keep") -> Attempt | None:
        cand = [r for r in self.rows if r.decision == decision and np_finite(r.J)]
        if case is not None:
            cand = [r for r in cand if r.case == case]
        if not cand:
            return None
        return min(cand, key=lambda r: r.J)

    def decide(self, child_j: float, parent: Attempt | None, *, eps: float = 0.02) -> tuple[str, str]:
        if parent is None or not np_finite(parent.J):
            return "keep", "no parent"
        if child_j > parent.J + eps:
            return "backtrack", f"J {child_j:.3f} > parent {parent.J:.3f}+{eps}"
        if child_j < parent.J - 1.0e-12:
            return "keep", f"J improved {parent.J - child_j:.3f} vs parent"
        return "keep", "J flat vs parent"


def np_finite(x: float) -> bool:
    try:
        return bool(x == x and abs(float(x)) != float("inf"))
    except (TypeError, ValueError):
        return False


def breakthroughs(
    journal: Journal,
    *,
    threshold: float = 1.0,
    case: str | None = None,
) -> dict:
    """First crossing of ``threshold``, and the largest keep-vs-parent drop."""
    rows = journal.rows
    if case is not None:
        rows = [r for r in rows if r.case == case]
    first = None
    for row in rows:
        if row.decision == "keep" and np_finite(row.J) and row.J < threshold:
            first = row
            break
    biggest = None
    biggest_dj = 0.0
    for row in rows:
        if row.decision != "keep" or row.parent is None or not np_finite(row.J):
            continue
        parent = journal.by_id(row.parent)
        if parent is None or not np_finite(parent.J):
            continue
        dj = parent.J - row.J
        if dj > biggest_dj:
            biggest_dj = dj
            biggest = row
    return {
        "threshold": threshold,
        "first_below": None if first is None else first.as_dict(),
        "largest_drop": None if biggest is None else {**biggest.as_dict(), "delta_J": biggest_dj},
    }
