from reservoir_backend.physics.conductivity import FractureConductivityModel
from reservoir_backend.physics.dual_rock import DualRock


def test_cf_only_changes_fracture_rock() -> None:
    dual = DualRock.from_cf(4, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=4.0e-13, phi_fracture=0.02)
    assert dual.n_cells == 4
    km0 = dual.matrix.permeability.copy()
    phi_m0 = dual.matrix.porosity.copy()
    updated = dual.with_cf(9.0e-13)
    assert updated.fracture.permeability[0] == 9.0e-13
    assert updated.matrix.permeability[0] == km0[0]
    assert updated.matrix.porosity[0] == phi_m0[0]


def test_conductivity_dual_rock_does_not_paint_matrix() -> None:
    model = FractureConductivityModel(
        n_cells=3, fracture_mask=__import__("numpy").array([True, True, True]), k_matrix_m2=2.0e-15
    )
    dual = model.dual_rock(5.0e-13, phi_matrix=0.10, phi_fracture=0.03)
    assert dual.matrix.permeability[0] == 2.0e-15
    assert dual.fracture.permeability[0] == 5.0e-13
