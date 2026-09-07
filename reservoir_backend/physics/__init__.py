from reservoir_backend.physics.capillary import (
    BrooksCorey,
    NoCapillary,
    TableCapillary,
    VanGenuchten,
    capillary_from_name,
)
from reservoir_backend.physics.relperm import CoreyThreePhase, CoreyTwoPhase, TableTwoPhase
from reservoir_backend.physics.dual_rock import DualRock
from reservoir_backend.physics.geomech import CartesianElasticity, GeomechSpec, geomech_from_cfg
from reservoir_backend.physics.rock import Rock, exp_permeability, log_permeability

__all__ = [
    "BrooksCorey",
    "CoreyThreePhase",
    "CoreyTwoPhase",
    "NoCapillary",
    "TableCapillary",
    "TableTwoPhase",
    "CartesianElasticity",
    "DualRock",
    "GeomechSpec",
    "Rock",
    "geomech_from_cfg",
    "VanGenuchten",
    "capillary_from_name",
    "exp_permeability",
    "log_permeability",
]
