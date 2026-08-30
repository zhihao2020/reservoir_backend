import numpy as np
import pytest

from reservoir_backend.exceptions import InvalidPermeability
from reservoir_backend.inverse.log_conductivity import LogConductivityParameterization


def test_scalar_cf_roundtrip() -> None:
    p = LogConductivityParameterization()
    cf = np.array([2.5e-13])
    m = p.encode(cf)
    assert m.size == 1
    out = p.decode(m)
    assert out == pytest.approx(cf)


def test_rejects_nonpositive_cf() -> None:
    p = LogConductivityParameterization()
    with pytest.raises(InvalidPermeability):
        p.encode(-1.0e-12)
    with pytest.raises(InvalidPermeability):
        p.encode(0.0)


def test_decode_never_negative() -> None:
    p = LogConductivityParameterization()
    cf = p.decode(np.array([-40.0]))
    assert cf[0] > 0.0
    assert np.isfinite(cf[0])
