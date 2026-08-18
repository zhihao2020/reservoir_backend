import numpy as np
import pytest

from reservoir_backend.inverse.structure import CATALOG, run_structure_search, should_search_structure, specs_from_names
from reservoir_backend.io.case import load_case
from reservoir_backend.validation.synthetic import make_two_layer_waterflood


def test_unknown_structure_name_errors() -> None:
    with pytest.raises(ValueError, match="unknown structure"):
        specs_from_names(["mystery"])


def test_region_map_does_not_search_unless_asked() -> None:
    assert should_search_structure(has_region_map=True, search_structure=None, candidates=None) is False
    assert should_search_structure(has_region_map=True, search_structure=True, candidates=None) is True
    assert should_search_structure(has_region_map=False, search_structure=True, candidates=None) is True
    assert should_search_structure(has_region_map=False, search_structure=None, candidates=None) is False


def test_lab_apply_transport_is_implicit() -> None:
    twin = load_case("config/lab_apply.yaml")
    assert twin.physics.implicit_transport is True


def test_structure_search_runs_catalog_and_picks_a_winner() -> None:
    case = make_two_layer_waterflood(
        n=(6, 4, 4), n_times=4, t_end=400.0, seed=2, history_frac=0.75, noise_p=4.0e2, noise_s=0.01
    )
    case.twin.inverse.n_ensemble = 8
    case.twin.inverse.n_assimilations = 2
    post, rows = run_structure_search(case.twin, specs=[CATALOG["z1"], CATALOG["z2"]])
    by_name = {r["name"]: r for r in rows}
    assert set(by_name) == {"z1", "z2"}
    assert any(r.get("selected") for r in rows)
    assert np.isfinite(post.holdout_rmse)
    assert case.twin.parameterization.n_params in {1, 2}
    # Two-layer truth should not make the 2-region hold-out much worse than homogeneous.
    assert by_name["z2"]["holdout_rmse"] < 3.0 * max(by_name["z1"]["holdout_rmse"], 1.0e-6)
