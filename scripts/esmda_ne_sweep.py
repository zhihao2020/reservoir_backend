"""Gate 10: ES-MDA posterior stability vs Ne for scalar C_f."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.inverse.ensemble_size import candidate_sizes, posterior_spread, recommend_ne
from reservoir_backend.synthetic import make_scalar_cf_twin


def main() -> int:
    rows = []
    for ne in candidate_sizes():
        case = make_scalar_cf_twin(ensemble_size=ne, assimilation_steps=2, seed=5, t_end=6.0, n_times=2)
        post = case.twin.calibrate()
        stats = posterior_spread(post.ensemble.theta_members)
        stats["ne"] = ne
        stats["err"] = abs(float(post.theta[0]) - float(case.theta_true[0]))
        stats["misfit"] = float(post.misfit[-1])
        rows.append(stats)
        print(json.dumps(stats))
    pick = recommend_ne(rows)
    print(json.dumps({"recommend_ne": pick, "rows": rows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
