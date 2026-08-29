"""Keyword gates for the Jiyang-pattern GEM CO2 huff-n-puff ruler. Does not run GEM."""

from pathlib import Path

import pytest

from validation.jiyang.cmg_co2_hnp.build_deck import OUT_DAT, build_deck, self_check


def test_build_deck_locks_wells_card_and_horizon() -> None:
    src = Path(r"D:\Tool\CMG\GEM\2024.20\TPL\spr\gmspr003.dat")
    if not src.is_file():
        pytest.skip("GEM TPL gmspr003.dat not installed")
    path = build_deck()
    self_check(path)
    assert path == OUT_DAT
    text = path.read_text(encoding="latin-1")
    assert "*WELL 1 'INJ'" in text
    assert "*WELL 2 'P1'" in text
    assert "*WELL 3 'P2'" in text
    assert "*WELL 4 'P3'" in text
    assert "*WELL 5 'P4'" in text
    assert "73.773" in text
    assert "45.992" in text
    assert "21.03" in text
    assert "304.128" in text
    assert "190.564" in text
    assert "617.7" in text
    assert "gmspr003.dat" in text.lower() or "GMSPR003" in text.upper()
    assert "not a Jiyang" in text.lower() or "NOT A JIYANG" in text.upper()
    assert "*GRID *CART 21 21 5" in text
    assert text.count("4 11 3") >= 1
    assert text.count("18 19 3") >= 1
    assert "*DATE 2000 1 1" in text
    assert "*DATE 2001 4 1" in text
    assert "cnpc" not in text.lower()
