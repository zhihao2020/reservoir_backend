from reservoir_backend.io.units import to_m2, to_m3_s, to_metres, to_pa, to_seconds


def test_lab_units() -> None:
    assert abs(to_metres(300.0, "mm") - 0.3) < 1.0e-15
    assert abs(to_seconds(30.0, "min") - 1800.0) < 1.0e-12
    assert to_pa(1.0, "bar") == 1.0e5
    assert to_m2(1.0, "mD") > 0.0
    assert to_m3_s(1.0, "mL/min") == 1.0e-6 / 60.0
