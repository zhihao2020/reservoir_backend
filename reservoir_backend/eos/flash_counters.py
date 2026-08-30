"""Flash call counters for one accepted timestep / a whole run."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FlashStats:
    n_flash_main: int = 0
    n_flash_thermo_jac: int = 0
    n_flash_line_search: int = 0
    n_stability: int = 0
    n_stability_bypass: int = 0
    n_warm_start: int = 0
    n_warm_fallback: int = 0
    n_cells: int = 0

    def add(self, other: FlashStats) -> None:
        self.n_flash_main += other.n_flash_main
        self.n_flash_thermo_jac += other.n_flash_thermo_jac
        self.n_flash_line_search += other.n_flash_line_search
        self.n_stability += other.n_stability
        self.n_stability_bypass += other.n_stability_bypass
        self.n_warm_start += other.n_warm_start
        self.n_warm_fallback += other.n_warm_fallback
        self.n_cells += other.n_cells


_STATS = FlashStats()


def stats() -> FlashStats:
    return _STATS


def reset_stats() -> None:
    global _STATS
    _STATS = FlashStats()


def bump(**kwargs: int) -> None:
    for key, value in kwargs.items():
        setattr(_STATS, key, getattr(_STATS, key) + int(value))
