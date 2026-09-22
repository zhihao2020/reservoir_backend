"""Single-command laboratory reconstruction (protocol v2)."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
from pathlib import Path

import numpy as np

from ..core.lab_case import load_lab_case, summarize_lab_case
from ..exceptions import CaseSchemaError, InvalidObservation
from ..programs.input_tcp import StepDecodeError, records_to_arrays, serve
from ..programs.pipeline import ProgramFields, field_scale_view, run_pipeline, write_output
from ..programs.protocol import (
    FIELD_IDS,
    FIELD_UNITS,
    KIND_NAMES,
    MAGIC,
    PROTOCOL_VERSION,
    QUANTITY_NAMES,
    QUANTITY_UNITS,
    NAK_BAD_DATA,
    NAK_BAD_TIME,
    NAK_BAD_VERSION,
    NAK_NOT_READY,
    NAK_TRUNCATED,
    TCP_ACK,
    TCP_HELLO,
    TCP_NAK,
    TCP_READY,
    TCP_STEP,
    UDP_RESEND,
    _MAX_DATAGRAM,
    _UDP_HEAD,
    json_frame,
    loads,
    unpack_step,
)
from ..programs.results import summarize_results
from ..programs.session import InversionSession
from ..programs.similarity import (
    MODEL_HEIGHT_M,
    MODEL_LENGTH_M,
    MODEL_VOLUME_M3,
    MODEL_WIDTH_M,
    similarity_ratios,
)
from ..programs.udp import publish_fields
from ..version import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reservoir",
        description="Probe and well series to full-grid pressure, saturation, porosity and permeability",
    )
    parser.add_argument("-V", "--version", action="version", version=f"reservoir-backend {__version__}")
    parser.add_argument("case", type=Path, help="path to case.yaml")
    parser.add_argument("-o", "--output", type=Path, default=None, help="optional directory for lab/field files")
    parser.add_argument("--tcp-port", type=int, default=None, metavar="PORT", help="listen for probe/well steps (persistent)")
    parser.add_argument("--ip", default="127.0.0.1", help="UDP destination IP for inverted fields")
    parser.add_argument("--lab-port", type=int, default=None, metavar="PORT", help="UDP port for laboratory-scale fields")
    parser.add_argument("--field-port", type=int, default=None, metavar="PORT", help="UDP port for field-scale fields")
    parser.add_argument("--control-port", type=int, default=None, metavar="PORT", help="UDP port to answer RESEND requests")
    parser.add_argument("--model", choices=("none", "black_oil", "compositional"), default=None, help="forward saturation model (overrides forward.model in the case)")
    args = parser.parse_args(argv)
    online = args.tcp_port is not None
    case = load_lab_case(args.case, require_series=not online)
    if args.model is not None:
        case.forward_model = args.model
    if online:
        session = InversionSession.open(case)
        print(
            json.dumps(
                {
                    "mode": "tcp",
                    "protocol_version": PROTOCOL_VERSION,
                    "tcp_port": int(args.tcp_port),
                    "ip": args.ip,
                    "lab_port": args.lab_port,
                    "field_port": args.field_port,
                    "control_port": args.control_port,
                    "window": session.window,
                    "n_probes": len(case.probes),
                    "n_wells": len(case.wells),
                    **summarize_lab_case(case),
                },
                indent=2,
            ),
            flush=True,
        )
        last = _Last()
        handler = _StepHandler(args, session, last)
        if args.control_port is not None:
            threading.Thread(
                target=_serve_control,
                args=(args, session, last),
                daemon=True,
            ).start()
        try:
            serve(int(args.tcp_port), handler)
        except KeyboardInterrupt:
            return 0
        return 0
    fields = run_pipeline(case)
    _emit(args, case, fields)
    return 0


class _Last:
    """Mutable holder for the most recently published reconstruction."""

    def __init__(self) -> None:
        self.fields: ProgramFields | None = None


class _StepHandler:
    def __init__(self, args: argparse.Namespace, session: InversionSession, last: _Last) -> None:
        self.args = args
        self.session = session
        self.last = last
        self.ready = False
        self.hello = _hello(session.case, session.mesh)

    def on_connect(self, host: str, port: int) -> list[bytes]:
        self.ready = False
        print(json.dumps({"tcp": "connected", "peer": f"{host}:{port}"}), flush=True)
        return [json_frame(TCP_HELLO, self.hello)]

    def on_message(self, version: int, msg_type: int, payload: bytes) -> list[bytes]:
        if version != PROTOCOL_VERSION:
            return [json_frame(TCP_NAK, {"seq": None, "code": NAK_BAD_VERSION, "message": f"unsupported protocol version {version}"})]
        if msg_type == TCP_READY:
            try:
                obj = loads(payload)
            except Exception:
                return []
            self.ready = True
            print(json.dumps({"tcp": "ready", "client_version": obj.get("version"), "ok": obj.get("ok")}), flush=True)
            return []
        if msg_type == TCP_STEP:
            if not self.ready:
                return [json_frame(TCP_NAK, {"seq": None, "code": NAK_NOT_READY, "message": "STEP before READY"})]
            return self._handle_step(payload)
        return []

    def _handle_step(self, payload: bytes) -> list[bytes]:
        try:
            step = unpack_step(payload)
        except ValueError as exc:
            return [json_frame(TCP_NAK, {"seq": None, "code": NAK_TRUNCATED, "message": str(exc)})]
        try:
            arrays = records_to_arrays(step, len(self.session.case.probes), len(self.session.case.wells))
            fields = self.session.step(
                arrays.time_s, arrays.pressure, arrays.sw, arrays.so, arrays.sg,
                arrays.well_pw, arrays.well_q, arrays.well_qw, arrays.well_qo, arrays.well_qg,
            )
        except StepDecodeError as exc:
            return [json_frame(TCP_NAK, {"seq": step.seq, "code": exc.code, "message": str(exc)})]
        except (InvalidObservation, CaseSchemaError) as exc:
            code = NAK_BAD_TIME if "earlier" in str(exc) else NAK_BAD_DATA
            return [json_frame(TCP_NAK, {"seq": step.seq, "code": code, "message": str(exc)})]
        self.last.fields = fields
        _emit(self.args, self.session.case, fields)
        return [json_frame(TCP_ACK, {"seq": step.seq, "time_s": step.time_s, "accepted": True})]

    def on_disconnect(self) -> None:
        print(json.dumps({"tcp": "disconnected"}), flush=True)


def _ratios(case) -> dict:
    model_flow_path = case.model_flow_path_m if case.model_flow_path_m is not None else MODEL_LENGTH_M
    return similarity_ratios(
        case.field_length_m,
        case.field_width_m,
        case.field_height_m,
        case.field_flow_path_m,
        model_flow_path_m=model_flow_path,
    )


def _hello(case, mesh) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "library": "reservoir-backend",
        "library_version": __version__,
        "n_probes": len(case.probes),
        "n_wells": len(case.wells),
        "probe_ids": case.probe_ids,
        "probe_xyz": [[float(v) for v in xyz] for xyz in case.probe_xyz],
        "well_ids": case.well_ids,
        "well_xyz": [[float(v) for v in xyz] for xyz in case.well_xyz],
        "grid": {
            "nx": int(mesh.grid.nx),
            "ny": int(mesh.grid.ny),
            "nz": int(mesh.grid.nz),
            "origin": [float(v) for v in mesh.grid.origin],
            "extent": [float(v) for v in mesh.grid.size_m()],
            "n_cells": int(mesh.grid.n_cells),
            "cell_order": "k*ny*nx + j*nx + i",
            "shape_ijk": list(mesh.grid.shape_ijk),
        },
        "similarity": _similarity_block(case, _ratios(case)),
        "quantity_enum": QUANTITY_NAMES,
        "quantity_units": QUANTITY_UNITS,
        "well_kind_enum": KIND_NAMES,
    }


def _similarity_block(case, ratios: dict) -> dict:
    return {
        **ratios,
        "model": {
            "length_m": MODEL_LENGTH_M,
            "width_m": MODEL_WIDTH_M,
            "height_m": MODEL_HEIGHT_M,
            "volume_m3": MODEL_VOLUME_M3,
        },
        "volume_basis": case.volume_basis,
    }


def _udp_manifest(grid, times, scale: str, ratios: dict, case) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "library": "reservoir-backend",
        "library_version": __version__,
        "scale": scale,
        "nx": int(grid.nx),
        "ny": int(grid.ny),
        "nz": int(grid.nz),
        "origin": [float(v) for v in grid.origin],
        "extent": [float(v) for v in grid.size_m()],
        "n_cells": int(grid.n_cells),
        "cell_order": "k*ny*nx + j*nx + i",
        "shape_ijk": list(grid.shape_ijk),
        "times": [float(v) for v in np.asarray(times, dtype=float)],
        "fields": [{"field_id": fid, "name": name, "unit": FIELD_UNITS[name]} for name, fid in FIELD_IDS.items()],
        "similarity": _similarity_block(case, ratios),
        "probes": [{"id": p.id, "x_m": float(p.x), "y_m": float(p.y), "z_m": float(p.z)} for p in case.probes],
        "wells": [
            {"id": w.id, "kind": w.kind, "x_m": float(w.x), "y_m": float(w.y), "z_m": float(w.z),
             "x2_m": None if w.x2 is None else float(w.x2),
             "y2_m": None if w.y2 is None else float(w.y2),
             "z2_m": None if w.z2 is None else float(w.z2)}
            for w in case.wells
        ],
    }


_publish_lock = threading.Lock()


def _publish_udp(args: argparse.Namespace, case, fields: ProgramFields) -> None:
    ratios = _ratios(case)
    with _publish_lock:
        if args.lab_port is not None:
            manifest = _udp_manifest(fields.mesh.grid, fields.times, "lab", ratios, case)
            results = summarize_results(case, fields.mesh, fields)
            publish_fields(args.ip, args.lab_port, manifest, fields.times, fields.p, fields.sw, fields.so, fields.sg, fields.phi, fields.k, results)
        if args.field_port is not None:
            field_grid, field_times, _ = field_scale_view(case, fields)
            manifest = _udp_manifest(field_grid, field_times, "field", ratios, case)
            results = summarize_results(case, fields.mesh, fields, times=field_times)
            publish_fields(args.ip, args.field_port, manifest, field_times, fields.p, fields.sw, fields.so, fields.sg, fields.phi, fields.k, results)


def _emit(args: argparse.Namespace, case, fields: ProgramFields) -> None:
    summary: dict = {}
    if args.output is not None:
        summary = write_output(args.output, case, fields)
        summary["output"] = str(args.output)
    else:
        summary["output"] = None
    _publish_udp(args, case, fields)
    summary.update(summarize_lab_case(case))
    print(json.dumps(summary, indent=2), flush=True)


def _serve_control(args: argparse.Namespace, session: InversionSession, last: _Last) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", int(args.control_port)))
    try:
        while True:
            data, _addr = sock.recvfrom(_MAX_DATAGRAM)
            if len(data) < _UDP_HEAD.size or data[:2] != MAGIC:
                continue
            _magic, _ver, typ, _sid, _seq = _UDP_HEAD.unpack_from(data, 0)
            if typ == UDP_RESEND and last.fields is not None:
                try:
                    _publish_udp(args, session.case, last.fields)
                except Exception:
                    pass  # a step may be mid-mutation; the next RESEND will succeed
    finally:
        sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
