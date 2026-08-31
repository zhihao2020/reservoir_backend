from scripts.lab_v1_gate import run_lab_gate


def test_lab_gate_is_distinct_and_records_workflow_fields() -> None:
    rec = run_lab_gate(dev=True, t_end=0.5, threads=1)
    for key in (
        "gate",
        "wall_s",
        "physical_time_advanced_s",
        "accepted_steps",
        "rejected_steps",
        "newton_iterations",
        "linear_iterations",
        "flash_calls",
        "mass_error",
        "max_dp",
        "max_dS",
        "peak_memory",
        "wall_per_physical_s",
        "n_inlet_cells",
        "n_sensors",
        "linear_backend",
    ):
        assert key in rec
    assert rec["gate"] == "lab_v1_gate"
    assert rec["gate"] != "dpdp_scale_gate"
    assert rec["n_inlet_cells"] == rec["grid"][1] * rec["grid"][2]
    assert rec["ok"] is True
