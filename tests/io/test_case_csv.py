from pathlib import Path

import yaml

from reservoir_backend.io.case import load_case


def test_load_controls_and_observations_from_csv(tmp_path: Path) -> None:
    ctrl = tmp_path / "controls.csv"
    obs = tmp_path / "observations.csv"
    ctrl.write_text(
        "time_s,port,kind,value\n"
        "0,INJ,rate,3e-8\n"
        "10,INJ,rate,3e-8\n"
        "0,INJ,composition,0.8\n"
        "10,INJ,composition,0.8\n"
        "0,PROD,pressure,1.0e5\n"
        "10,PROD,pressure,1.0e5\n",
        encoding="utf-8",
    )
    obs.write_text(
        "time_s,sensor,kind,value,sigma,holdout\n"
        "0,P1,pressure,1.05e5,2000,0\n"
        "10,P1,pressure,1.06e5,2000,0\n"
        "0,S1,saturation,0.22,0.04,1\n"
        "10,S1,saturation,0.25,0.04,1\n",
        encoding="utf-8",
    )
    yml = tmp_path / "case.yaml"
    yml.write_text(
        yaml.safe_dump(
            {
                "geometry": {"size_m": [0.16, 0.08, 0.08]},
                "grid": {"spacing_m": 0.04},
                "physics": {"model": "two_phase_immiscible", "capillary": "none"},
                "ports": [
                    {"name": "INJ", "role": "injector", "control": "rate", "x": 0.02, "y": 0.04, "z": 0.04},
                    {"name": "PROD", "role": "producer", "control": "pressure", "x": 0.14, "y": 0.04, "z": 0.04},
                ],
                "sensors": [
                    {"name": "P1", "kind": "pressure", "x": 0.06, "y": 0.04, "z": 0.04, "sigma": 2000},
                    {"name": "S1", "kind": "saturation", "x": 0.10, "y": 0.04, "z": 0.04, "sigma": 0.04},
                ],
                "inverse": {"parameterization": "region", "n_regions": 2, "max_iter": 6},
                "experiment": {
                    "controls": "controls.csv",
                    "observations": "observations.csv",
                    "holdout_sensors": ["S1"],
                    "history_end_s": 10,
                },
            }
        ),
        encoding="utf-8",
    )
    twin = load_case(yml)
    assert twin.inverse.max_iter == 6
    assert len(twin.experiment.controls) >= 3
    assert any(o.holdout for o in twin.experiment.observations)
    assert any(o.sensor_name == "P1" for o in twin.experiment.observations)
