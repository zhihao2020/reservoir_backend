"""Published EXAMPLE binary: methane + n-decane. Not a Jiyang GEM card.

Criticals: Reid, Prausnitz, Poling, The Properties of Gases and Liquids,
5th ed., McGraw-Hill, 2001, Appendix A.
Binary k_ij(C1–nC10) ≈ 0.049: Katz & Firoozabadi, JPT 1978; tabulated in
Whitson & Brulé, Phase Behavior, SPE Monograph 20, 2000.
Critical volumes (GEM *VCRIT only): OPM 1D_COMP.DATA METHANE/DECANE
0.09863 / 0.60980 m3/kmol. The cubic itself does not use Vc.
"""

from __future__ import annotations

import numpy as np

from reservoir_backend.eos.pr import PengRobinson

EXAMPLE_NAMES = ("C1", "nC10")


def example_c1_nc10() -> PengRobinson:
    """C1–nC10 Peng–Robinson. SI: K, Pa, kg/mol."""
    kij = np.array([[0.0, 0.049], [0.049, 0.0]], dtype=float)
    return PengRobinson(
        tc=np.array([190.564, 617.70], dtype=float),
        pc=np.array([4.5992e6, 2.103e6], dtype=float),
        omega=np.array([0.01142, 0.490], dtype=float),
        mw=np.array([16.0425e-3, 142.282e-3], dtype=float),
        kij=kij,
        names=EXAMPLE_NAMES,
    )
