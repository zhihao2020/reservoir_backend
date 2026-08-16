"""CMG invert harness: multi-case scores, prune, journal backtracking.

Pass bar is gauges / hold-out / field p-Sw / breakthrough — not K_CMG.
"""

from reservoir_backend.validation.cmg_harness.catalog import CaseSpec, get_case, list_cases
from reservoir_backend.validation.cmg_harness.journal import Journal, breakthroughs
from reservoir_backend.validation.cmg_harness.score import Score, combine_j

__all__ = ["CaseSpec", "Journal", "Score", "breakthroughs", "combine_j", "get_case", "list_cases"]
