"""YAML ports and optional CMG/IMEX well include -> existing FlowPort."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from reservoir_backend.exceptions import InvalidControl
from reservoir_backend.ports.flow import FlowPort

_RATE_CTRL = frozenset(
    {"STW", "STO", "STG", "STL", "BHW", "BHO", "BHG", "BHL", "RATE", "Q"}
)
_BHP_CTRL = frozenset({"BHP", "PRESSURE", "PRES", "P"})
_SKIP_KW = frozenset(
    {
        "TIME",
        "DATE",
        "WPRN",
        "OUTPRN",
        "WSRF",
        "OUTSRF",
        "INCOMP",
        "MONITOR",
        "SHUTIN",
        "OPEN",
        "LAYER",
        "QUALIZER",
        "SCLTBL-WELL",
        "SCLTBL",
        "WELLHYD",
        "GROUP",
        "GCONPROD",
        "GCONINJE",
        "VFP",
        "VFPPROD",
        "VFPINJ",
        "WORKOVER",
        "WLIFT",
        "WELLLIST",
        "BRANCH",
        "MLWELL",
        "ONTIME",
        "STATUS",
        "SHUTIN",
        "TRIGGER",
        "ALTER",
    }
)


def ports_from_cfg(cfg: dict[str, Any], grid: Any, *, cfg_dir: str | Path = ".") -> list[FlowPort]:
    """Build FlowPort list from YAML ``ports`` / ``wells`` and optional ``wells.file``."""
    ports = [_port_from_yaml(grid, row) for row in _yaml_well_rows(cfg)]
    path = _well_file_path(cfg, cfg_dir)
    if path is not None:
        extra = ports_from_well_file(path, grid)
        _reject_duplicate_names(ports, extra)
        ports.extend(extra)
    return ports


def ports_from_well_file(path: str | Path, grid: Any) -> list[FlowPort]:
    text = Path(path).read_text(encoding="utf-8")
    return parse_well_deck(text, grid, source=str(path))


def parse_well_deck(text: str, grid: Any, *, source: str = "well deck") -> list[FlowPort]:
    """Parse a CMG-ish *WELL / *PERF snippet onto FlowPort. Numbers stay as written."""
    drafts = _parse_drafts(text, source=source)
    ports: list[FlowPort] = []
    for draft in drafts:
        if not draft["perfs"]:
            raise InvalidControl("well " + draft["name"] + " has no perforations in " + source)
        if draft["role"] is None:
            raise InvalidControl("well " + draft["name"] + " missing INJECTOR/PRODUCER")
        cells = _cells_from_ijk_triples(grid, draft["perfs"], one_based=True)
        ports.append(
            FlowPort(
                name=str(draft["name"]),
                role=str(draft["role"]),
                control=str(draft["control"] or "pressure"),
                cell_ids=np.asarray(cells, dtype=np.int64),
                sw_inj=float(draft["sw_inj"]),
                use_productivity=bool(draft["use_productivity"]),
                rw_m=float(draft["rw_m"]),
                skin=float(draft["skin"]),
                geofac=float(draft["geofac"]),
                wi_multiplier=float(draft["wi_multiplier"]),
            )
        )
    return ports


def _yaml_well_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ports = cfg.get("ports") or []
    if isinstance(ports, list):
        rows.extend(ports)
    wells = cfg.get("wells")
    if isinstance(wells, list):
        rows.extend(wells)
    elif isinstance(wells, dict):
        extra = wells.get("ports") or wells.get("wells") or []
        if isinstance(extra, list):
            rows.extend(extra)
    return rows


def _well_file_path(cfg: dict[str, Any], cfg_dir: str | Path) -> Path | None:
    wells = cfg.get("wells")
    if isinstance(wells, str) and wells.strip():
        return Path(cfg_dir) / wells
    if isinstance(wells, dict) and wells.get("file"):
        return Path(cfg_dir) / str(wells["file"])
    return None


def _reject_duplicate_names(existing: list[FlowPort], extra: list[FlowPort]) -> None:
    have = {p.name for p in existing}
    clash = sorted({p.name for p in extra} & have)
    if clash:
        raise InvalidControl("wells.file repeats YAML well names: " + ", ".join(clash))


def _port_from_yaml(grid: Any, p: dict[str, Any]) -> FlowPort:
    name = str(p["name"])
    role = str(p.get("role", "injector"))
    control = str(p.get("control", "rate"))
    sw_inj = float(p.get("sw_inj", 1.0))
    use_wi = bool(p.get("use_productivity", False))
    rw_m = float(p.get("rw_m", 0.0))
    skin = float(p.get("skin", 0.0))
    geofac = float(p.get("geofac", 0.0))
    cells = _yaml_cells(grid, p)
    if cells is not None:
        port = FlowPort(
            name=name,
            role=role,
            control=control,
            cell_ids=np.asarray(cells, dtype=np.int64),
            sw_inj=sw_inj,
            use_productivity=use_wi,
            rw_m=rw_m,
            skin=skin,
            geofac=geofac,
        )
    else:
        perforation = str(p.get("perforation", "point")).lower()
        if perforation in {"column", "full_column", "z"}:
            port = FlowPort.column(
                grid,
                name,
                role,
                control,
                float(p["x"]),
                float(p["y"]),
                sw_inj=sw_inj,
                use_productivity=use_wi,
                rw_m=rw_m,
                skin=skin,
                geofac=geofac,
            )
        else:
            xyz = (float(p["x"]), float(p["y"]), float(p.get("z", 0.0)))
            port = FlowPort.at_point(
                grid,
                name,
                role,
                control,
                xyz,
                radius_m=float(p.get("radius_m", 0.0)),
                sw_inj=sw_inj,
                use_productivity=use_wi,
                rw_m=rw_m,
                skin=skin,
                geofac=geofac,
            )
    if p.get("axis") is not None:
        port.axis = str(p["axis"]).strip().lower()[:1] or "k"
    if p.get("wi_multiplier") is not None:
        port.wi_multiplier = float(p["wi_multiplier"])
    return port


def _yaml_cells(grid: Any, p: dict[str, Any]) -> list[int] | None:
    if p.get("ijk") is not None:
        return _cells_from_ijk_triples(grid, p["ijk"], one_based=True)
    if p.get("perforations") is not None:
        return _cells_from_ijk_triples(grid, p["perforations"], one_based=True)
    if all(k in p for k in ("I", "J", "K")):
        return _cells_from_ijk_triples(grid, [(p["I"], p["J"], p["K"])], one_based=True)
    if all(k in p for k in ("i", "j", "k")):
        return _cells_from_ijk_triples(grid, [(p["i"], p["j"], p["k"])], one_based=True)
    return None


def _cells_from_ijk_triples(grid: Any, raw: Any, *, one_based: bool) -> list[int]:
    triples = _as_triples(raw)
    if not triples:
        raise InvalidControl("well ijk / perforations is empty")
    cells: list[int] = []
    for i_raw, j_raw, k_raw in triples:
        i, j, k = int(i_raw), int(j_raw), int(k_raw)
        if one_based:
            i, j, k = _to_zero_based(i, j, k)
        cells.append(int(grid.index(i, j, k)))
    return cells


def _as_triples(raw: Any) -> list[tuple[Any, Any, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        if all(k in raw for k in ("I", "J", "K")):
            return [(raw["I"], raw["J"], raw["K"])]
        if all(k in raw for k in ("i", "j", "k")):
            return [(raw["i"], raw["j"], raw["k"])]
        raise InvalidControl("perforation mapping needs I/J/K or i/j/k")
    if isinstance(raw, (list, tuple)):
        if len(raw) == 3 and not isinstance(raw[0], (list, tuple, dict)):
            return [(raw[0], raw[1], raw[2])]
        out: list[tuple[Any, Any, Any]] = []
        for item in raw:
            out.extend(_as_triples(item))
        return out
    raise InvalidControl("ijk must be [I,J,K] or a list of those")


def _to_zero_based(i: int, j: int, k: int) -> tuple[int, int, int]:
    """CMG IJK is 1-based. A 0 in any index is treated as already 0-based."""
    if min(i, j, k) < 0:
        raise InvalidControl("ijk indices must be non-negative")
    if min(i, j, k) == 0:
        return i, j, k
    return i - 1, j - 1, k - 1


def _new_draft(well_id: str, name: str) -> dict[str, Any]:
    return {
        "id": str(well_id),
        "name": str(name),
        "role": None,
        "control": None,
        "perfs": [],
        "sw_inj": 1.0,
        "use_productivity": False,
        "rw_m": 0.0,
        "skin": 0.0,
        "geofac": 0.0,
        "wi_multiplier": 1.0,
        "wfrac": 1.0,
    }


def _parse_drafts(text: str, *, source: str) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    current: dict[str, Any] | None = None
    mode: str | None = None

    def _ensure(wid: str, name: str | None = None) -> dict[str, Any]:
        key = str(wid)
        if key not in by_id:
            draft = _new_draft(key, name or key)
            by_id[key] = draft
            order.append(key)
        elif name and by_id[key]["name"] == key:
            by_id[key]["name"] = name
        return by_id[key]

    for raw_line in text.splitlines():
        tokens = _tokens(raw_line)
        if not tokens:
            continue
        head = _keyword(tokens[0])
        if head is None:
            if mode == "perf" and current is not None:
                _add_perf_tokens(current, tokens)
            continue
        mode = None
        if head == "WELL":
            wid, wname = _well_id_name(tokens[1:])
            current = _ensure(wid, wname)
            continue
        if head in {"INJECTOR", "INJ", "INJECTION"}:
            wid = tokens[1] if len(tokens) > 1 else (current["id"] if current else None)
            if wid is None:
                raise InvalidControl("INJECTOR without well id in " + source)
            current = _ensure(_bare(wid))
            current["role"] = "injector"
            continue
        if head in {"PRODUCER", "PROD", "PRODUCTION"}:
            wid = tokens[1] if len(tokens) > 1 else (current["id"] if current else None)
            if wid is None:
                raise InvalidControl("PRODUCER without well id in " + source)
            current = _ensure(_bare(wid))
            current["role"] = "producer"
            continue
        if head == "OPERATE":
            if current is None:
                raise InvalidControl("OPERATE before WELL in " + source)
            _apply_operate(current, tokens[1:])
            continue
        if head == "GEOMETRY":
            if current is None:
                raise InvalidControl("GEOMETRY before WELL in " + source)
            _apply_geometry(current, tokens[1:])
            continue
        if head == "PERF":
            wid = _perf_well_id(tokens[1:], current)
            if wid is None:
                raise InvalidControl("PERF without well id in " + source)
            current = _ensure(wid)
            if "GEO" in {_keyword(t) or _bare(t).upper() for t in tokens[1:]}:
                current["use_productivity"] = True
            mode = "perf"
            continue
        if head == "INCOMP":
            if current is not None:
                rest = {_keyword(t) or _bare(t).upper() for t in tokens[1:]}
                if "OIL" in rest or "GAS" in rest:
                    current["sw_inj"] = 0.0
                elif "WATER" in rest:
                    current["sw_inj"] = 1.0
            continue
        if head in _SKIP_KW or head.startswith("OUT") or head.startswith("WPRN"):
            continue
        # Unknown keyword: ignore the record (subset parser).
        continue

    return [by_id[k] for k in order if by_id[k]["role"] is not None or by_id[k]["perfs"]]


def _well_id_name(args: list[str]) -> tuple[str, str]:
    if not args:
        raise InvalidControl("WELL needs an id or name")
    if len(args) == 1:
        return _bare(args[0]), _bare(args[0])
    return _bare(args[0]), _bare(args[1])


def _perf_well_id(args: list[str], current: dict[str, Any] | None) -> str | None:
    for tok in args:
        kw = _keyword(tok)
        if kw in {None}:
            return _bare(tok)
        if kw in {"GEO", "IJK", "K", "I", "J", "GEOLOGY"}:
            continue
        if kw.isdigit() or _is_name(tok):
            return _bare(tok)
    if args:
        last = args[-1]
        if _keyword(last) not in {"GEO", "IJK", "K", "I", "J"}:
            return _bare(last)
    return None if current is None else str(current["id"])


def _is_name(tok: str) -> bool:
    t = _bare(tok)
    return bool(t) and not t.replace(".", "", 1).replace("-", "", 1).isdigit()


def _add_perf_tokens(draft: dict[str, Any], tokens: list[str]) -> None:
    nums: list[float] = []
    for tok in tokens:
        if _keyword(tok) in {"OPEN", "SHUT", "CLOSE", "ON", "OFF"}:
            continue
        try:
            nums.append(float(tok))
        except ValueError:
            return
    if len(nums) < 3:
        return
    i, j, k = int(nums[0]), int(nums[1]), int(nums[2])
    draft["perfs"].append((i, j, k))
    if len(nums) >= 4:
        ff = float(nums[3])
        if draft["perfs"] and len(draft["perfs"]) == 1:
            draft["wi_multiplier"] = ff * float(draft.get("wfrac") or 1.0)
        # later connections keep the first ff; FlowPort has one multiplier


def _apply_operate(draft: dict[str, Any], args: list[str]) -> None:
    kinds = [_keyword(t) or _bare(t).upper() for t in args]
    chosen: str | None = None
    for kind in kinds:
        if kind in _BHP_CTRL:
            chosen = "pressure"
            break
        if kind in _RATE_CTRL:
            chosen = "rate"
            break
    if chosen is None:
        return
    if draft["control"] is None:
        draft["control"] = chosen


def _apply_geometry(draft: dict[str, Any], args: list[str]) -> None:
    nums: list[float] = []
    for tok in args:
        kw = _keyword(tok)
        if kw in {"K", "I", "J", "H", "GEO"}:
            continue
        try:
            nums.append(float(tok))
        except ValueError:
            continue
    draft["use_productivity"] = True
    if len(nums) >= 1:
        draft["rw_m"] = float(nums[0])
    if len(nums) >= 2:
        draft["geofac"] = float(nums[1])
    if len(nums) >= 3:
        draft["wfrac"] = float(nums[2])
        draft["wi_multiplier"] = float(draft["wi_multiplier"]) * float(nums[2])
    if len(nums) >= 4:
        draft["skin"] = float(nums[3])


def _tokens(line: str) -> list[str]:
    s = _strip_comment(line)
    if not s:
        return []
    out: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    for ch in s:
        if quote is not None:
            if ch == quote:
                out.append("".join(buf))
                buf = []
                quote = None
            else:
                buf.append(ch)
            continue
        if ch in {"'", '"'}:
            if buf:
                out.append("".join(buf))
                buf = []
            quote = ch
            continue
        if ch.isspace() or ch == ",":
            if buf:
                out.append("".join(buf))
                buf = []
            continue
        if ch == "/":
            if buf:
                out.append("".join(buf))
                buf = []
            break
        buf.append(ch)
    if buf:
        out.append("".join(buf))
    return [t for t in out if t]


def _strip_comment(line: str) -> str:
    cut = line
    for sep in ("**", "!", "--"):
        idx = cut.find(sep)
        if idx >= 0:
            cut = cut[:idx]
    return cut.strip()


def _keyword(tok: str) -> str | None:
    t = tok.strip()
    if not t:
        return None
    if t.startswith("*"):
        t = t.lstrip("*")
        return t.upper() if t else None
    up = t.upper()
    if up in {
        "WELL",
        "INJECTOR",
        "INJ",
        "INJECTION",
        "PRODUCER",
        "PROD",
        "PRODUCTION",
        "OPERATE",
        "GEOMETRY",
        "PERF",
        "INCOMP",
        "MAX",
        "MIN",
        "BHP",
        "STW",
        "STO",
        "STG",
        "STL",
        "BHW",
        "BHO",
        "BHG",
        "GEO",
        "IJK",
        "WATER",
        "OIL",
        "GAS",
        "TIME",
        "DATE",
        "OPEN",
        "SHUT",
        "LAYER",
        "CONT",
        "REPEAT",
        "K",
        "I",
        "J",
        "H",
    } or up in _SKIP_KW:
        return up
    return None


def _bare(tok: str) -> str:
    t = tok.strip()
    if t.startswith("*"):
        t = t.lstrip("*")
    if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
        t = t[1:-1]
    return t
