# CLI Usage

## Basic Command

```bash
python scripts/run_case.py --config config/demo_case.yaml
```

## Module Entry Point

```bash
python -m reservoir_backend.cli.run_case --config config/demo_case.yaml
```

## Dry Run

```bash
python scripts/run_case.py --config config/combined_case.yaml --dry-run
```

Dry-run validates and normalizes the config, prints core parameters, and does
not write simulation results.

## Override Case ID

```bash
python scripts/run_case.py --config config/demo_case.yaml --case-id my_case
```

## Override Output Directory

```bash
python scripts/run_case.py --config config/demo_case.yaml --output-dir results_test
```

## Supported Arguments

- `--config`: path to YAML case config. Required.
- `--output-dir`: override `case.output_dir`.
- `--case-id`: override `case.case_id`.
- `--mode`: override `case.mode`; valid values are `archie_only` and
  `multisignal`.
- `--dry-run`: validate and print normalized core settings without running the
  calculation.
- `--verbose`: print formatted JSON response.

## Common Commands

```bash
python scripts/run_case.py --config config/demo_case.yaml
python scripts/run_case.py --config config/multisignal_case.yaml
python scripts/run_case.py --config config/capillary_case.yaml
python scripts/run_case.py --config config/capillary_gradient_case.yaml
python scripts/run_case.py --config config/gravity_case.yaml
python scripts/run_case.py --config config/combined_case.yaml
```
