from scripts.dpdp_scale_gate import run_standard_step


def test_scale_gate_records_required_fields() -> None:
    rec = run_standard_step(1, t_end=0.05, threads=1)
    for key in (
        "gate",
        "commit",
        "cpu",
        "threads",
        "python",
        "numpy",
        "scipy",
        "n_accept",
        "n_reject",
        "newton_its",
        "mass_rel",
        "jac_s",
        "solve_s",
        "flash_s",
        "t_end_s",
        "max_steps",
        "flash_backend",
    ):
        assert key in rec
    assert rec["gate"] == "dpdp_scale_gate"
    assert rec["max_steps"] == 1
    assert rec["ok"] is True
