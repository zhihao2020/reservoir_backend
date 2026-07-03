"""Configuration-driven case runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from examples.run_full_pipeline_demo import run_demo
from reservoir_backend.io.config_loader import load_case_config, normalize_units, validate_case_config


def run_case_from_config(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    case_id: str | None = None,
    mode: str | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, object]:
    """Run or validate a configured case."""
    config = load_case_config(config_path)
    if output_dir is not None:
        config["case"]["output_dir"] = str(output_dir)
    if case_id is not None:
        config["case"]["case_id"] = case_id
    if mode is not None:
        if mode not in {"archie_only", "multisignal", "three_phase"}:
            raise ValueError("--mode must be archie_only, multisignal, or three_phase")
        config["case"]["mode"] = mode
        if mode == "three_phase":
            config.setdefault("three_phase", {})["enabled"] = True
    validate_case_config(config)
    config = normalize_units(config)

    if dry_run:
        three_phase_enabled = bool(config.get("three_phase", {}).get("enabled", False))
        summary = {
            "case_id": config["case"]["case_id"],
            "output_dir": config["case"]["output_dir"],
            "mode": config["case"]["mode"],
            "grid": config["grid"],
            "permeability_m2": config["rock"]["permeability_m2"],
            "left_pressure_pa": config["pressure"]["left_pressure_pa"],
            "right_pressure_pa": config["pressure"]["right_pressure_pa"],
            "capillary_enabled": bool(config["capillary_pressure"]["enabled"]),
            "capillary_model": config["capillary_pressure"]["model"],
            "gravity_enabled": bool(config["gravity"]["enabled"]),
            "combined_transport_enabled": bool(config["capillary_pressure"]["enabled"]) and bool(config["gravity"]["enabled"]),
            "three_phase_enabled": three_phase_enabled,
            "three_phase_model": config.get("three_phase", {}).get("model", "none"),
            "three_phase_transport_enabled": three_phase_enabled,
            "black_oil_enabled": False,
            "rho_w": config["gravity"]["rho_w"],
            "rho_o": config["gravity"]["rho_o"],
            "density_difference": config["gravity"]["rho_w"] - config["gravity"]["rho_o"],
            "initial_saturation_type": config.get("initial_saturation", {}).get("type", "constant"),
            "dry_run": True,
            "success": True,
        }
        if verbose:
            print(json.dumps(summary, indent=2))
        return summary

    result = run_demo(
        case_id=config["case"]["case_id"],
        results_root=config["case"]["output_dir"],
        use_multisignal=config["case"]["mode"] == "multisignal",
        case_config=config,
    )
    response = {
        "case_path": str(result["case_dir"]),
        "case_summary": str(Path(result["case_dir"]) / "case_summary.json"),
        "success": True,
    }
    if verbose:
        print(json.dumps(response, indent=2))
    return response


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run reservoir_backend case from YAML config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--case-id")
    parser.add_argument("--mode", choices=["archie_only", "multisignal", "three_phase"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_case_from_config(
            args.config,
            output_dir=args.output_dir,
            case_id=args.case_id,
            mode=args.mode,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        if not args.verbose:
            print(json.dumps(result))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
