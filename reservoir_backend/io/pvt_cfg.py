"""Case-level oil PVT factory. Presets plus optional user tables.

PVT (including mu) is known fluid for the experiment. theta is rock (log K) only.
Supported presets: incompressible | slightly_compressible | cmg_seawater.
Mapping form: scalar mu / ct / pb overrides, plus SI table arrays that stamp
BlackOilPVT. Optional YAML/JSON sidecar via file / pvto (relative to the case
YAML). CMG *PVTO / *PVTW / *PVDG / *PVT text is parsed into those arrays
(field default: psi, scf/stb, cP -- same as cmg_seawater).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.io.pvto_load import looks_like_cmg_pvt, parse_pvto
from reservoir_backend.physics.pvt import BlackOilPVT

_CMG_ALIASES = frozenset({"cmg", "cmg_seawater", "black_oil"})
_INCOMP_ALIASES = frozenset({"incompressible", "none", "0", ""})
_SCALAR_KEYS = frozenset({"preset", "name", "type", "mu_w", "mu_o", "mu_g", "ct", "compressibility", "pb"})
_FILE_KEYS = frozenset({"file", "pvto", "table"})
_TABLE_ALIASES: dict[str, str] = {
    "p": "p_tab",
    "p_tab": "p_tab",
    "rs": "rs_tab",
    "rs_tab": "rs_tab",
    "bo": "bo_tab",
    "bo_tab": "bo_tab",
    "eg": "eg_tab",
    "eg_tab": "eg_tab",
    "bg": "eg_tab",
    "bg_tab": "eg_tab",
    "muo": "muo_tab",
    "mu_o_tab": "muo_tab",
    "muo_tab": "muo_tab",
    "mug": "mug_tab",
    "mu_g_tab": "mug_tab",
    "mug_tab": "mug_tab",
    "p_w": "p_w_tab",
    "p_w_tab": "p_w_tab",
    "bw": "bw_tab",
    "bw_tab": "bw_tab",
    "muw": "muw_tab",
    "mu_w_tab": "muw_tab",
    "muw_tab": "muw_tab",
}
_BG_KEYS = frozenset({"bg", "bg_tab"})
_KNOWN_MAP_KEYS = _SCALAR_KEYS | _FILE_KEYS | frozenset(_TABLE_ALIASES)


def _as_f64(name: str, values: Any) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("physics.pvt " + name + " is empty")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _pick_table(raw: dict[str, Any], names: tuple[str, ...]) -> tuple[str | None, NDArray[np.float64] | None]:
    for name in names:
        if name in raw and raw[name] is not None:
            return name, _as_f64(name, raw[name])
    return None, None


def _tables_from_mapping(raw: dict[str, Any]) -> dict[str, NDArray[np.float64]]:
    out: dict[str, NDArray[np.float64]] = {}
    _, val = _pick_table(raw, ("p_tab", "p"))
    if val is not None:
        out["p_tab"] = val
    _, val = _pick_table(raw, ("rs_tab", "rs"))
    if val is not None:
        out["rs_tab"] = val
    _, val = _pick_table(raw, ("bo_tab", "bo"))
    if val is not None:
        out["bo_tab"] = val
    _, val = _pick_table(raw, ("eg_tab", "eg"))
    if val is not None:
        out["eg_tab"] = val
    else:
        _, val = _pick_table(raw, ("bg_tab", "bg"))
        if val is not None:
            out["eg_tab"] = 1.0 / np.maximum(val, 1.0e-30)
    _, val = _pick_table(raw, ("muo_tab", "muo", "mu_o_tab"))
    if val is not None:
        out["muo_tab"] = val
    _, val = _pick_table(raw, ("mug_tab", "mug", "mu_g_tab"))
    if val is not None:
        out["mug_tab"] = val
    _, val = _pick_table(raw, ("p_w_tab", "p_w"))
    if val is not None:
        out["p_w_tab"] = val
    _, val = _pick_table(raw, ("bw_tab", "bw"))
    if val is not None:
        out["bw_tab"] = val
    _, val = _pick_table(raw, ("muw_tab", "muw", "mu_w_tab"))
    if val is not None:
        out["muw_tab"] = val
    return out


def _load_sidecar(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if looks_like_cmg_pvt(text):
        return parse_pvto(text)
    suffix = path.suffix.lower()
    data: Any
    if suffix == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("PyYAML is required to read PVT sidecar " + str(path)) from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("PVT sidecar must be a mapping: " + str(path))
    inner = data.get("pvt")
    if isinstance(inner, dict) and not any(k in data for k in _TABLE_ALIASES):
        data = inner
    return data


def _sidecar_maps(raw: dict[str, Any], cfg_dir: Path | None) -> list[dict[str, Any]]:
    maps: list[dict[str, Any]] = []
    for key in ("file", "pvto", "table"):
        if key not in raw or raw[key] is None:
            continue
        val = raw[key]
        if isinstance(val, dict):
            maps.append(val)
            continue
        path = Path(str(val))
        if not path.is_absolute():
            base = Path(cfg_dir) if cfg_dir is not None else Path(".")
            path = base / path
        if not path.is_file():
            raise ValueError("physics.pvt " + key + " not found: " + str(path))
        maps.append(_load_sidecar(path))
    return maps


def _check_table_lengths(tabs: dict[str, NDArray[np.float64]], *, p_axis: NDArray[np.float64] | None) -> None:
    n = None if p_axis is None else int(p_axis.size)
    for name in ("rs_tab", "bo_tab", "eg_tab", "muo_tab", "mug_tab"):
        if name not in tabs:
            continue
        if n is None:
            raise ValueError("physics.pvt " + name + " needs p / p_tab")
        if int(tabs[name].size) != n:
            raise ValueError(
                "physics.pvt " + name + " length " + str(int(tabs[name].size))
                + " != p_tab length " + str(n)
            )
    p_w = tabs.get("p_w_tab", p_axis)
    n_w = None if p_w is None else int(np.asarray(p_w).size)
    for name in ("bw_tab", "muw_tab"):
        if name not in tabs:
            continue
        if n_w is None:
            raise ValueError("physics.pvt " + name + " needs p_w / p_tab")
        if int(tabs[name].size) != n_w:
            raise ValueError(
                "physics.pvt " + name + " length " + str(int(tabs[name].size))
                + " != water p length " + str(n_w)
            )


def _overlay_tables(pvt: BlackOilPVT, tabs: dict[str, NDArray[np.float64]]) -> BlackOilPVT:
    if not tabs:
        return pvt
    p_axis = tabs.get("p_tab", pvt.p_tab)
    _check_table_lengths(tabs, p_axis=None if p_axis is None else np.asarray(p_axis, dtype=float))
    merged = dict(tabs)
    if "p_tab" in tabs:
        n = int(tabs["p_tab"].size)
        for name in ("rs_tab", "bo_tab", "eg_tab", "muo_tab", "mug_tab"):
            if name in merged:
                continue
            old = getattr(pvt, name)
            if old is not None and int(np.asarray(old).size) != n:
                merged[name] = None
    return replace(pvt, **merged)


def _scalars_from_mapping(raw: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("bw_ref", "cw", "pref_w", "mu_w", "co", "pb", "bo_ref", "pref_o"):
        if key in raw and raw[key] is not None:
            out[key] = float(raw[key])
    return out


def _collect_overlays(
    raw: dict[str, Any], cfg_dir: Path | None
) -> tuple[dict[str, NDArray[np.float64]], dict[str, float]]:
    tabs: dict[str, NDArray[np.float64]] = {}
    scalars: dict[str, float] = {}
    for extra in _sidecar_maps(raw, cfg_dir):
        tabs.update(_tables_from_mapping(extra))
        scalars.update(_scalars_from_mapping(extra))
    tabs.update(_tables_from_mapping(raw))
    return tabs, scalars



def _preset_and_overrides(phys_cfg: dict[str, Any], model: str) -> tuple[str, dict[str, float]]:
    """Resolve preset name and optional scalar mu / ct overrides from physics cfg."""
    raw = phys_cfg.get("pvt", None)
    overrides: dict[str, float] = {}
    preset = ""

    if isinstance(raw, dict):
        unknown = set(raw.keys()) - _KNOWN_MAP_KEYS
        if unknown:
            warnings.warn(
                "physics.pvt mapping ignores unknown keys: " + str(sorted(unknown)),
                stacklevel=3,
            )
        preset = str(raw.get("preset") or raw.get("name") or raw.get("type") or "").strip().lower()
        for key in ("mu_w", "mu_o", "mu_g"):
            if key in raw and raw[key] is not None:
                overrides[key] = float(raw[key])
        if "ct" in raw and raw["ct"] is not None:
            overrides["ct"] = float(raw["ct"])
        elif "compressibility" in raw and raw["compressibility"] is not None:
            c = raw["compressibility"]
            if str(c) not in _INCOMP_ALIASES:
                overrides["ct"] = float(c)
        if "pb" in raw and raw["pb"] is not None:
            overrides["pb"] = float(raw["pb"])
    elif raw is not None:
        preset = str(raw).strip().lower()

    if not preset:
        if model in {"black_oil", "d"}:
            preset = "cmg_seawater"
        else:
            comp = phys_cfg.get("compressibility", "incompressible")
            if str(comp) in _INCOMP_ALIASES:
                preset = "incompressible"
            else:
                preset = "slightly_compressible"
                if "ct" not in overrides:
                    overrides["ct"] = float(comp) if not isinstance(comp, str) else 1.5e-9

    if preset in _CMG_ALIASES:
        preset = "cmg_seawater"
    elif preset in _INCOMP_ALIASES:
        preset = "incompressible"
    elif preset in {"slightly_compressible", "compressible", "slight"}:
        preset = "slightly_compressible"
        if "ct" not in overrides:
            comp = phys_cfg.get("compressibility", 1.5e-9)
            if str(comp) in _INCOMP_ALIASES:
                overrides["ct"] = 1.5e-9
            else:
                overrides["ct"] = float(comp) if not isinstance(comp, str) else 1.5e-9
    else:
        raise ValueError(
            "unknown physics.pvt preset " + repr(preset) + "; "
            "use incompressible | slightly_compressible | cmg_seawater"
        )
    return preset, overrides


def pvt_from_cfg(
    phys_cfg: dict[str, Any] | None,
    *,
    p_init: float,
    model: str = "two_phase_immiscible",
    cfg_dir: str | Path | None = None,
) -> BlackOilPVT:
    """Build BlackOilPVT from case physics block.

    Parameters
    ----------
    phys_cfg:
        cfg["physics"] mapping (may be empty).
    p_init:
        Initial / reference pressure [Pa] for compressible and CMG presets.
    model:
        Phase model name; black_oil / d selects cmg_seawater when pvt is omitted.
    cfg_dir:
        Directory of the case YAML; used to resolve physics.pvt file / pvto paths.
    """
    phys = dict(phys_cfg or {})
    preset, ov = _preset_and_overrides(phys, str(model).lower())

    if preset == "cmg_seawater":
        pb = ov.get("pb")
        if pb is not None:
            pvt = BlackOilPVT.cmg_seawater(p_init=float(p_init), pb=float(pb))
        else:
            pvt = BlackOilPVT.cmg_seawater(p_init=float(p_init))
    else:
        mu_kw: dict[str, float] = {}
        if "mu_w" in ov:
            mu_kw["mu_w"] = float(ov["mu_w"])
        if "mu_o" in ov:
            mu_kw["mu_o"] = float(ov["mu_o"])
        if "mu_g" in ov:
            mu_kw["mu_g"] = float(ov["mu_g"])

        if preset == "incompressible":
            pvt = BlackOilPVT.incompressible(**mu_kw)
        else:
            ct = float(ov.get("ct", 1.5e-9))
            sc_kw = {k: v for k, v in mu_kw.items() if k in {"mu_w", "mu_o"}}
            pvt = BlackOilPVT.slightly_compressible(ct, pref=float(p_init), **sc_kw)
            if "mu_g" in mu_kw:
                pvt = replace(pvt, mu_g=float(mu_kw["mu_g"]))

    raw = phys.get("pvt", None)
    if isinstance(raw, dict):
        base = Path(cfg_dir) if cfg_dir is not None else None
        tabs, scalars = _collect_overlays(raw, base)
        pvt = _overlay_tables(pvt, tabs)
        if scalars:
            pvt = replace(pvt, **scalars)

    # Live-oil tables own mu(p); scalar overrides only stamp constant fields
    # used when tables are not queried (dead-oil / reporting).
    stamp = {k: float(ov[k]) for k in ("mu_w", "mu_o", "mu_g") if k in ov}
    if stamp:
        pvt = replace(pvt, **stamp)
    if "pb" in ov and isinstance(raw, dict):
        pvt = replace(pvt, pb=float(ov["pb"]))
    return pvt


def pvt_preset_name(phys_cfg: dict[str, Any] | None, *, model: str = "two_phase_immiscible") -> str:
    """Resolved preset string for reports (no fluid construction)."""
    preset, _ = _preset_and_overrides(dict(phys_cfg or {}), str(model).lower())
    return preset
