# Release Checklist

## Regression Checklist

- [ ] `pytest -q` passes
- [ ] `python scripts/run_case.py --config config/demo_case.yaml` runs
- [ ] `python scripts/run_case.py --config config/multisignal_case.yaml` runs
- [ ] `python scripts/run_case.py --config config/capillary_case.yaml` runs
- [ ] `python scripts/run_case.py --config config/capillary_gradient_case.yaml` runs
- [ ] `python scripts/run_case.py --config config/gravity_case.yaml` runs
- [ ] `python scripts/run_case.py --config config/combined_case.yaml` runs
- [ ] `python harness/run_validation.py` runs
- [ ] `python scripts/validate_combined_pipeline.py` runs
- [ ] `python scripts/profile_full_pipeline.py` runs
- [ ] `python scripts/profile_capillary_pipeline.py` runs
- [ ] `python scripts/profile_combined_pipeline.py` runs

## Documentation Checklist

- [ ] README updated
- [ ] docs updated
- [ ] specs updated
- [ ] module matrix updated
- [ ] case configuration documented
- [ ] CLI usage documented
- [ ] limitations and roadmap documented

## Release Guardrails

- [ ] no core solver modified during release documentation stage
- [ ] no UDP added
- [ ] no C++ added
- [ ] no three-phase flow added
- [ ] no black-oil model added
- [ ] tag created
