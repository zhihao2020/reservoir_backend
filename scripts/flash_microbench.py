"""Gate 5 flash microbenchmark: 10k states, FastPR vs reference.

Hard gate: T_fast < T_reference / 20 before re-running 10³ DPDP.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.eos.example import example_c1_nc10
from reservoir_backend.eos.flash_backend import FastPRBackend, ReferencePRBackend


def main() -> int:
    eos = example_c1_nc10()
    rng = np.random.default_rng(0)
    n = 10_000
    p = rng.uniform(3.0e6, 2.5e7, size=n)
    z1 = rng.uniform(0.1, 0.9, size=n)
    z = np.stack((z1, 1.0 - z1), axis=1)
    ref = ReferencePRBackend()
    fast = FastPRBackend()
    t0 = time.perf_counter()
    a = ref.evaluate_batch(eos, p[:200], 350.0, z[:200])
    t_ref_small = time.perf_counter() - t0
    t_ref = t_ref_small * (n / 200.0)
    t1 = time.perf_counter()
    b = fast.evaluate_batch(eos, p, 350.0, z)
    t_fast = time.perf_counter() - t1
    dv = float(np.max(np.abs(a.vapor_frac - fast.evaluate_batch(eos, p[:200], 350.0, z[:200]).vapor_frac)))
    ratio = t_ref / max(t_fast, 1.0e-9)
    print(f"N={n}")
    print(f"T_reference_est={t_ref:.4f}s (from 200 states × {n/200:.0f})")
    print(f"T_fast={t_fast:.4f}s")
    print(f"speedup={ratio:.1f}x")
    print(f"max |ΔV| on 200={dv:.3e}")
    print(f"gate_20x={'PASS' if ratio >= 20.0 else 'FAIL'}")
    _ = b
    return 0 if ratio >= 20.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
