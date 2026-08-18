"""Extract lightweight fixtures from downloaded open-source reference files."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
FIXTURES = ROOT / "fixtures"


def extract_reference_cases() -> dict:
    """Extract metadata and small arrays from reference files."""
    opm_water = _extract_opm_water_1ph()
    opm_spe1 = _extract_opm_spe1()
    seq_tpfa = _extract_seq_tpfa()
    seq_bl = _extract_seq_buckley_leverett()
    summary = {
        "fixture_name": "open_source_adapted_reference_cases",
        "success": True,
        "sources": [
            opm_water["source"],
            opm_spe1["source"],
            seq_tpfa["source"],
            seq_bl["source"],
        ],
        "cases": [_case_without_arrays(opm_water), _case_without_arrays(opm_spe1), _case_without_arrays(seq_tpfa), _case_without_arrays(seq_bl)],
        "policy": {
            "runtime_dependency": False,
            "full_spe10_reproduction": False,
            "opm_flow_equivalence": False,
            "seq_runtime_integration": False,
            "commercial_simulator_equivalence": False,
        },
    }
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "open_source_adapted_cases.json").write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    np.savez(
        FIXTURES / "open_source_adapted_arrays.npz",
        spe1_permx_md=opm_spe1["arrays"]["permx_md"],
        spe1_dz_ft=opm_spe1["arrays"]["dz_ft"],
        seq_bl_grid=np.asarray(seq_bl["grid"], dtype=int),
        seq_bl_perm_md=np.asarray([seq_bl["permeability_md"]], dtype=float),
        seq_bl_porosity=np.asarray([seq_bl["porosity"]], dtype=float),
    )
    return summary


def _require_upstream(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing upstream file {path}; run: git submodule update --init --depth 1"
        )
    return path


def _extract_opm_water_1ph() -> dict:
    # WATER2F uses EQUALS blocks and later multiplier lines that would confuse a
    # last-keyword-wins scan. Keep property reads on the EQUALS-aware helpers.
    path = _require_upstream(UPSTREAM / "opm-tests" / "water-1ph" / "WATER2F.DATA")
    text = path.read_text(encoding="utf-8")
    dims = _first_ints_after_keyword(text, "SPECGRID", count=3)
    if dims is None:
        dims = _first_ints_after_keyword(text, "DIMENS", count=3)
    poro = _equals_value(text, "PORO")
    permx = _equals_value(text, "PERMX")
    permz = _equals_value(text, "PERMZ")
    return {
        "case_name": "opm_water_1ph_single_cell",
        "source": {
            "project": "OPM/opm-tests",
            "path": "water-1ph/WATER2F.DATA",
            "url": "https://github.com/OPM/opm-tests/blob/master/water-1ph/WATER2F.DATA",
        },
        "grid": dims,
        "porosity": poro,
        "permeability_md": {"kx": permx, "ky": permx, "kz": permz},
        "adapted_benchmark_use": "single-cell property parsing and pressure sanity reference",
    }


def _extract_opm_spe1() -> dict:
    path = _require_upstream(UPSTREAM / "opm-tests" / "spe1" / "SPE1CASE1.DATA")
    from reservoir_backend.io.structured_deck import load_structured_deck

    bundle = load_structured_deck(path)
    nx, ny, nz = bundle.grid.nx, bundle.grid.ny, bundle.grid.nz
    poro = bundle.porosity_field
    permx = bundle.permeability_x_md
    if poro is None or permx is None:
        raise ValueError("SPE1 adapted extract requires PORO and PERMX")
    dz_vec = bundle.grid.spacing_k
    dz = np.broadcast_to(dz_vec[:, None, None], (nz, ny, nx)).copy()
    arrays = {
        "permx_md": np.asarray(permx, dtype=float),
        "dz_ft": np.asarray(dz, dtype=float),
    }
    return {
        "case_name": "opm_spe1_case1_layered_subset",
        "source": {
            "project": "OPM/opm-tests",
            "path": "spe1/SPE1CASE1.DATA",
            "url": "https://github.com/OPM/opm-tests/blob/master/spe1/SPE1CASE1.DATA",
        },
        "grid": [nx, ny, nz],
        "porosity_min": float(np.min(poro)),
        "porosity_max": float(np.max(poro)),
        "permeability_min_md": float(np.min(permx)),
        "permeability_max_md": float(np.max(permx)),
        "permeability_contrast": float(np.max(permx) / np.min(permx)),
        "adapted_benchmark_use": "layered heterogeneous Cartesian pressure benchmark precursor",
        "arrays": arrays,
    }


def _extract_seq_tpfa() -> dict:
    path = _require_upstream(
        UPSTREAM / "mrst" / "modules" / "book" / "examples" / "1phase" / "src" / "simpleIncompTPFA.m"
    )
    text = path.read_text(encoding="utf-8")
    return {
        "case_name": "seq_simple_incomp_tpfa_reference",
        "source": {
            "project": "book-examples",
            "path": "modules/book/examples/1phase/src/simpleIncompTPFA.m",
        },
        "mentions_tpfa": "two-point flux approximation" in text.lower() or "TPFA" in text,
        "mentions_boundary_conditions": "Boundary condition" in text or "bc" in text,
        "mentions_sources": "source" in text.lower(),
        "adapted_benchmark_use": "TPFA diagnostic and report schema inspiration; not executed",
    }


def _extract_seq_buckley_leverett() -> dict:
    path = _require_upstream(
        UPSTREAM / "mrst" / "modules" / "book" / "examples" / "in2ph" / "buckleyLeverett1D.m"
    )
    text = path.read_text(encoding="utf-8")
    grid_match = re.search(r"cartGrid\(\[(\d+)\s*,\s*(\d+)\]\)", text)
    rock_match = re.search(r"makeRock\(G,\s*([\d.]+)\*milli\*darcy,\s*([\d.]+)\)", text)
    if grid_match is None or rock_match is None:
        raise ValueError("failed to parse sequential Buckley-Leverett reference")
    grid = [int(grid_match.group(1)), int(grid_match.group(2))]
    perm = float(rock_match.group(1))
    porosity = float(rock_match.group(2))
    return {
        "case_name": "seq_buckley_leverett_1d_reference",
        "source": {
            "project": "book-examples",
            "path": "modules/book/examples/in2ph/buckleyLeverett1D.m",
        },
        "grid": grid,
        "permeability_md": perm,
        "porosity": porosity,
        "mentions_explicit_transport": "explicitTransport" in text,
        "mentions_implicit_transport": "implicitTransport" in text,
        "adapted_benchmark_use": "1D waterflood front movement and timestep sensitivity reference",
    }


def _first_ints_after_keyword(text: str, keyword: str, count: int) -> list[int]:
    block = _keyword_block(text, keyword)
    values = []
    for token in block.replace("/", " ").split():
        try:
            values.append(int(token))
        except ValueError:
            continue
        if len(values) == count:
            return values
    raise ValueError(f"could not parse {count} integer values after {keyword}")


def _equals_value(text: str, keyword: str) -> float:
    pattern = rf"\b{keyword}\s+([\d.Ee+-]+)"
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"could not parse EQUALS value for {keyword}")
    return float(match.group(1))


def _expanded_values_after_keyword(text: str, keyword: str, expected: int) -> np.ndarray:
    block = _keyword_block(text, keyword)
    values: list[float] = []
    for raw in block.replace("/", " ").split():
        if raw.startswith("--"):
            continue
        if "*" in raw:
            left, right = raw.split("*", 1)
            repeat = int(left) if left else 1
            value = float(right)
            values.extend([value] * repeat)
        else:
            try:
                values.append(float(raw))
            except ValueError:
                continue
        if len(values) >= expected:
            return np.asarray(values[:expected], dtype=float)
    raise ValueError(f"could not parse {expected} values after {keyword}")


def _case_without_arrays(case: dict) -> dict:
    return {key: value for key, value in case.items() if key != "arrays"}


def _keyword_block(text: str, keyword: str) -> str:
    pattern = rf"(?ms)^\s*{re.escape(keyword)}\s*\n(.*?)/"
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"could not find keyword block {keyword}")
    lines = []
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        lines.append(stripped.split("--", 1)[0])
    return "\n".join(lines)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


if __name__ == "__main__":
    print(json.dumps(extract_reference_cases(), indent=2, default=_json_default))
