"""Named invert presets. AutoGluon idea: hide knobs, expose quality/time.

These are ensemble-design defaults, not a search over K or a clone of F.
"""

from __future__ import annotations

PRESETS: dict[str, dict[str, float | int]] = {
    "fast": {"n_ensemble": 8, "n_assimilations": 2, "prior_std": 0.8, "inflation": 1.02},
    "balanced": {"n_ensemble": 24, "n_assimilations": 4, "prior_std": 0.8, "inflation": 1.02},
    "strict": {"n_ensemble": 40, "n_assimilations": 6, "prior_std": 1.0, "inflation": 1.04},
}


def preset_names() -> list[str]:
    return list(PRESETS)


def knobs_for(name: str) -> dict[str, float | int]:
    key = str(name).strip().lower()
    if key not in PRESETS:
        raise ValueError(f"unknown invert preset {name!r}; choose from {preset_names()}")
    return dict(PRESETS[key])


def portfolio_candidates(seed: int, *, time_limit_s: float | None = None) -> list[tuple[str, dict[str, float | int | str]]]:
    """Multi-algorithm portfolio. Same F; different assimilators from the methods notes."""
    s = int(seed)
    items: list[tuple[str, dict[str, float | int | str]]] = [
        ("es", {"n_ensemble": 12, "n_assimilations": 1, "prior_std": 0.8, "inflation": 1.0, "algorithm": "es", "seed": s}),
        (
            "esmda",
            {
                "n_ensemble": 12,
                "n_assimilations": 4,
                "prior_std": 0.8,
                "inflation": 1.02,
                "algorithm": "esmda",
                "seed": s + 1,
            },
        ),
    ]
    if time_limit_s is None or float(time_limit_s) >= 20.0:
        items.append(
            (
                "esmda_geo",
                {
                    "n_ensemble": 12,
                    "n_assimilations": 4,
                    "prior_std": 0.8,
                    "inflation": 1.02,
                    "algorithm": "esmda_geo",
                    "seed": s + 2,
                },
            )
        )
        items.append(
            (
                "ies",
                {
                    "n_ensemble": 12,
                    "n_assimilations": 4,
                    "prior_std": 0.8,
                    "inflation": 1.0,
                    "algorithm": "ies",
                    "seed": s + 3,
                },
            )
        )
        items.append(
            (
                "esmda_rs",
                {
                    "n_ensemble": 12,
                    "n_assimilations": 6,
                    "prior_std": 1.0,
                    "inflation": 1.02,
                    "algorithm": "esmda_rs",
                    "seed": s + 4,
                },
            )
        )
    return items
