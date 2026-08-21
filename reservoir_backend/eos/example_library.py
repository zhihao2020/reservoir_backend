"""EXAMPLE eight-component PR library (public literature, not a field card).

Component set: C1, C2, C3, nC4, nC5, nC6, example_C7plus, CO2.

Criticals (Tc, Pc, ω) are published textbook / NIST values for the named
pure species. ``example_C7plus`` is published n-decane (nC10) used as a
single EXAMPLE C7+ pseudo — not a Jiyang / 济阳 crude cut and not a GEM card.

Binary interaction parameters are common published CO2–n-alkane k_ij
(Danesh 1998; Whitson & Brulé 2000 compilation). Hydrocarbon–hydrocarbon
k_ij = 0. Symmetric van der Waals BIP.

STATUS: standalone example flash, not wired, example PR params.
Not site-calibrated. Not field-validated.
"""

from __future__ import annotations

import numpy as np

from reservoir_backend.eos.peng_robinson import EosMixture

# Tests assert this token so the table cannot be mistaken for a GEM card.
EXAMPLE_LIBRARY_MARKER = (
    "EXAMPLE / public literature — standalone example flash, not wired, "
    "example PR params; NOT a Jiyang GEM card; NOT site-calibrated"
)

# Poling, Prausnitz, O'Connell, The Properties of Gases and Liquids, 5th ed.
# (2001) and NIST Chemistry WebBook. Pc in Pa, Tc in K.
_EXAMPLE_SPECIES: tuple[tuple[str, float, float, float], ...] = (
    ("C1", 190.564, 4.5992e6, 0.01142),
    ("C2", 305.32, 4.8722e6, 0.0995),
    ("C3", 369.83, 4.2471e6, 0.1523),
    ("nC4", 425.12, 3.7960e6, 0.2002),
    ("nC5", 469.70, 3.3700e6, 0.2515),
    ("nC6", 507.60, 3.0250e6, 0.3013),
    # Published n-decane as an EXAMPLE C7+ stand-in (not Jiyang crude).
    ("example_C7plus", 617.70, 2.1030e6, 0.4884),
    ("CO2", 304.1282, 7.3773e6, 0.2236),
)

# EXAMPLE published typical CO2–alkane k_ij. Index matches _EXAMPLE_SPECIES.
# CO2–C1 0.105, CO2–C2 0.130, CO2–C3 0.125, CO2–nC4..nC6 0.115, CO2–nC10 0.100.
_CO2_K_IJ = {
    "C1": 0.105,
    "C2": 0.130,
    "C3": 0.125,
    "nC4": 0.115,
    "nC5": 0.115,
    "nC6": 0.115,
    "example_C7plus": 0.100,
}

# EXAMPLE synthetic feed (mole fractions). Not a field composition.
_EXAMPLE_FEED = {
    "C1": 0.45,
    "C2": 0.05,
    "C3": 0.05,
    "nC4": 0.03,
    "nC5": 0.02,
    "nC6": 0.02,
    "example_C7plus": 0.18,
    "CO2": 0.20,
}


def example_eight_component_mixture() -> EosMixture:
    """Return the EXAMPLE 8-component library. Public literature, not GEM."""
    names = tuple(row[0] for row in _EXAMPLE_SPECIES)
    tc = np.array([row[1] for row in _EXAMPLE_SPECIES], dtype=float)
    pc = np.array([row[2] for row in _EXAMPLE_SPECIES], dtype=float)
    omega = np.array([row[3] for row in _EXAMPLE_SPECIES], dtype=float)
    nc = len(names)
    kij = np.zeros((nc, nc), dtype=float)
    index = {name: i for i, name in enumerate(names)}
    i_co2 = index["CO2"]
    for name, k in _CO2_K_IJ.items():
        j = index[name]
        kij[i_co2, j] = k
        kij[j, i_co2] = k
    return EosMixture(names=names, Tc=tc, Pc=pc, omega=omega, kij=kij, marker=EXAMPLE_LIBRARY_MARKER)


def example_feed_z() -> np.ndarray:
    """EXAMPLE 8-component feed mole fractions (not a Jiyang / GEM composition)."""
    names = [row[0] for row in _EXAMPLE_SPECIES]
    z = np.array([_EXAMPLE_FEED[name] for name in names], dtype=float)
    return z / z.sum()
