from pathlib import Path

from reservoir_backend.validation.cmg_harness.journal import Attempt, Journal, breakthroughs


def test_journal_tree_backtrack_and_breakthrough(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    j = Journal(path)
    a0 = j.append(
        Attempt("t001", None, "lab_layers", {"n_ensemble": 8}, "ok", {"hold": 1.4}, 1.40, "keep", "seed")
    )
    dec, _reason = j.decide(1.10, a0)
    assert dec == "keep"
    j.append(
        Attempt("t002", "t001", "lab_layers", {"n_ensemble": 12}, "ok", {"hold": 0.90}, 0.90, "keep", "improved")
    )
    dec_bad, _ = j.decide(1.50, j.by_id("t002"))
    assert dec_bad == "backtrack"
    j.append(
        Attempt("t003", "t002", "lab_layers", {"n_ensemble": 24}, "ok", {"hold": 1.5}, 1.50, "backtrack", "worse")
    )
    j.append(
        Attempt("t004", "t001", "fault", {"n_ensemble": 8}, "prune:no_flood", {}, 9.0, "prune", "no_flood")
    )
    best = j.best(case="lab_layers")
    assert best is not None and best.id == "t002"
    hits = breakthroughs(j, threshold=1.0, case="lab_layers")
    assert hits["first_below"]["id"] == "t002"
    assert hits["largest_drop"]["id"] == "t002"
    assert hits["largest_drop"]["delta_J"] > 0.4
    reloaded = Journal(path)
    assert len(reloaded.rows) == 4
    assert reloaded.children("t001")[0].id == "t002"
