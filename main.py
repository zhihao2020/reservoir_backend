"""Entry point for the sensor four-field backend."""

from __future__ import annotations


def main() -> None:
    print("Four-field pipeline ready.")
    print("  python -m reservoir_backend.pipeline.run --config config/sensor_case.yaml")
    print("  pytest tests/test_pipeline_mesh.py tests/test_pipeline_fields.py tests/test_pipeline_e2e_cli.py -q")


if __name__ == "__main__":
    main()
