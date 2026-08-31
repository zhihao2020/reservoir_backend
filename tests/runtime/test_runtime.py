import json
from pathlib import Path

import numpy as np

from reservoir_backend.io.case import load_case
from reservoir_backend.io.udp_api import TwinUDPProtocol
from reservoir_backend.runtime.command_queue import CommandQueue
from reservoir_backend.runtime.field_store import FieldStore
from reservoir_backend.runtime.twin_runtime import TwinRuntime


def test_update_control_appends_series() -> None:
    twin = load_case("examples/lab_v1/case_dev.yaml")
    rt = TwinRuntime(twin, field_folder="results/fields_test")
    n0 = len(twin.experiment.controls)
    out = rt.update_control("INJ", "rate", 1.5e-4, 7.0)
    assert out["ok"] is True
    rates = [c for c in twin.experiment.controls if c.port_name == "INJ" and c.kind == "rate"]
    assert rates
    times = np.asarray(rates[0].times_s, dtype=float)
    vals = np.asarray(rates[0].values, dtype=float)
    assert 7.0 in times
    assert float(vals[times == 7.0][0]) == 1.5e-4
    assert len(twin.experiment.controls) >= n0


def test_observe_skips_reused_times() -> None:
    twin = load_case("examples/lab_v1/case_dev.yaml")
    rt = TwinRuntime(twin, field_folder="results/fields_test")
    a = rt.observe(sensor_id="P001", kind="pressure", value=1.21e7, sigma=2.0e3, time_s=1.0)
    b = rt.observe(sensor_id="P001", kind="pressure", value=1.22e7, sigma=2.0e3, time_s=1.0)
    assert a["reused"] is False
    assert b["reused"] is True


def test_udp_observe_sensor_id_goes_to_runtime() -> None:
    twin = load_case("examples/lab_v1/case_dev.yaml")
    rt = TwinRuntime(twin, field_folder="results/fields_test")
    proto = TwinUDPProtocol(runtime=rt)
    payload = {
        "cmd": "observe",
        "time_s": 2.0,
        "sensor_id": "S001",
        "kind": "sw",
        "value": 0.46,
        "sigma": 0.025,
    }
    out = json.loads(proto.handle_bytes(json.dumps(payload).encode("utf-8")))
    assert out["ok"] is True
    assert out["sensor_id"] == "S001"


def test_field_store_writes_npz(tmp_path: Path) -> None:
    store = FieldStore(tmp_path)
    meta = store.write(
        {"pressure_fracture": np.ones(4), "sw": np.zeros(4), "Cf_mean": np.array([1.0e-12])},
        time_s=3.0,
        metadata={"pressure_source": "fast", "saturations_held": True, "last_full_time_s": 0.0},
    )
    assert meta["frame_id"] == 1
    assert Path(meta["path"]).is_file()
    assert meta["pressure_source"] == "fast"
    assert meta["saturations_held"] is True


def test_command_queue_fifo() -> None:
    q = CommandQueue()
    q.push("update_control", {"port": "INJ"})
    q.push("observe", {"sensor_id": "P001"})
    assert len(q) == 2
    assert q.pop().name == "update_control"
    assert q.pop().name == "observe"
    assert q.pop() is None


def test_replay_experiment_appends_without_newton(tmp_path: Path) -> None:
    from reservoir_backend.runtime.replay import replay_experiment

    report = replay_experiment(Path("experiments/EXP001"), output=tmp_path)
    assert report["n_observations_appended"] > 0
    assert (tmp_path / "replay.json").is_file()
    assert report["snapshot"] is not None
    assert report["snapshot"]["transport"] == "npz"
    assert Path(report["snapshot"]["path"]).is_file()
    assert report["snapshot"]["pressure_source"] in {"fast", "full"}
