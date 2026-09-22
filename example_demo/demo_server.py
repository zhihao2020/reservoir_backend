"""Showcase backend: frozen protocol v2, linear-interpolation core.

For pre-contract demos only. Speaks the exact wire protocol in
``docs/接口协议.md`` but replaces the real field reconstruction (kriging + rock
inversion) with a trivial 3-D linear interpolation, so the end-to-end data flow
can be demonstrated without shipping the core source code.

This directory is **self-contained**: it only imports its own ``protocol.py``
(a copy of the frozen, public wire framing). It does not import, and does not
need, any core inversion source.

Run from the repo root::

    python example_demo/demo_server.py examples/small/case.yaml \
        --tcp-port 9000 --ip 127.0.0.1 --lab-port 9001 --field-port 9002

Then drive it with the reference collector (it speaks the same frozen protocol)::

    python examples/online/send_steps.py --port 9000 \
        --probes examples/small/probes.csv --wells examples/small/wells.csv \
        --observations examples/small/observations.csv --series examples/small/series.csv

The dashboard that consumes the UDP field stream is provided by another team
(per ``docs/接口协议.md``); the reference receiver is ``src.programs.udp.receive_fields``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import griddata

from protocol import (  # local, self-contained copy of the frozen wire framing
    FIELD_IDS,
    FIELD_UNITS,
    KIND_NAMES,
    NAK_BAD_VERSION,
    NAK_NOT_READY,
    NAK_TRUNCATED,
    PROBE_Q_P,
    PROBE_Q_SG,
    PROBE_Q_SO,
    PROBE_Q_SW,
    PROTOCOL_VERSION,
    QUANTITY_NAMES,
    QUANTITY_UNITS,
    TCP_ACK,
    TCP_BYE,
    TCP_HELLO,
    TCP_NAK,
    TCP_READY,
    TCP_STEP,
    UDP_END,
    UDP_FIELD,
    UDP_MANIFEST,
    UDP_RESULTS,
    WELL_K_PW,
    _VALUES_PER_DATAGRAM,
    crc32,
    dumps,
    json_frame,
    loads,
    pack_end,
    pack_field_chunk,
    read_frame,
    udp_header,
    unpack_step,
)

MD_TO_M2 = 9.86923266716013e-16

MODEL_LENGTH_M = 0.30
MODEL_WIDTH_M = 0.30
MODEL_HEIGHT_M = 0.30
MODEL_VOLUME_M3 = MODEL_LENGTH_M * MODEL_WIDTH_M * MODEL_HEIGHT_M

# 滑动窗口：只看上下相邻时刻（当前 + 上一拍），只保留并重发这 2 拍。
_WINDOW = 2


# --------------------------------------------------------------------------
# minimal grid geometry (no dependency on the core cartesian module)
# --------------------------------------------------------------------------
def _cell_centers(nx, ny, nz, extent, origin):
    dx, dy, dz = extent[0] / nx, extent[1] / ny, extent[2] / nz
    cx = origin[0] + (np.arange(nx) + 0.5) * dx
    cy = origin[1] + (np.arange(ny) + 0.5) * dy
    cz = origin[2] + (np.arange(nz) + 0.5) * dz
    zz, yy, xx = np.meshgrid(cz, cy, cx, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)


def _locate_cell(xyz, nx, ny, nz, extent, origin):
    """Map an (n,3) coordinate array to flat cell indices ``k*ny*nx + j*nx + i``."""
    d = np.array(extent) / np.array([nx, ny, nz])
    ijk = (xyz - np.array(origin)) / d
    ijk = np.floor(ijk).astype(int)
    i = np.clip(ijk[:, 0], 0, nx - 1)
    j = np.clip(ijk[:, 1], 0, ny - 1)
    k = np.clip(ijk[:, 2], 0, nz - 1)
    return k * ny * nx + j * nx + i


# --------------------------------------------------------------------------
# case loading (grid + probes + wells; ignores inversion config)
# --------------------------------------------------------------------------
def _read_points(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [str(r["id"]) for r in rows]
    xyz = np.array([[float(r["x_m"]), float(r["y_m"]), float(r["z_m"])] for r in rows], dtype=float)
    return ids, xyz


def _read_wells(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [str(r["id"]) for r in rows]
    kinds = [str(r["kind"]) for r in rows]
    xyz = np.array([[float(r["x_m"]), float(r["y_m"]), float(r["z_m"])] for r in rows], dtype=float)
    return ids, kinds, xyz


def _load_case(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent
    geo = raw["geometry"]
    rock = raw.get("rock", {})
    sim = raw.get("similarity", {})
    probe_ids, probe_xyz = _read_points(base / raw["probes"]["file"])
    well_ids, well_kinds, well_xyz = _read_wells(base / raw["wells"]["file"])
    return {
        "origin": tuple(float(v) for v in geo.get("origin_m", [0.0, 0.0, 0.0])),
        "extent": tuple(float(v) for v in geo["extent_m"]),
        "nx": int(geo["nx"]), "ny": int(geo["ny"]), "nz": int(geo["nz"]),
        "phi0": float(rock.get("phi0", 0.05)),
        "k0": float(rock.get("k0_md", 0.03)) * MD_TO_M2,
        "probe_ids": probe_ids, "probe_xyz": probe_xyz,
        "well_ids": well_ids, "well_kinds": well_kinds, "well_xyz": well_xyz,
        "field_length_m": float(sim.get("field_length_m", 300.0)),
        "field_width_m": float(sim.get("field_width_m", 300.0)),
        "field_height_m": float(sim.get("field_height_m", 30.0)),
        "field_flow_path_m": float(sim.get("field_flow_path_m", 150.0)),
        "model_flow_path_m": float(sim.get("model_flow_path_m", 0.15)),
    }


def _ratios(cfg: dict) -> dict:
    lam_x = cfg["field_length_m"] / MODEL_LENGTH_M
    lam_y = cfg["field_width_m"] / MODEL_LENGTH_M
    lam_z = cfg["field_height_m"] / MODEL_LENGTH_M
    c_v = lam_x * lam_y * lam_z
    c_lc = cfg["field_flow_path_m"] / cfg["model_flow_path_m"]
    return {
        "lambda_x": lam_x, "lambda_y": lam_y, "lambda_z": lam_z,
        "c_v": c_v, "c_lc": c_lc, "c_t": c_lc, "c_qv": c_v / c_lc, "c_vinj": c_v,
    }


# --------------------------------------------------------------------------
# linear interpolation core
# --------------------------------------------------------------------------
def _linear_interp(points, values, targets):
    """3-D linear interpolation, nearest-neighbour for out-of-hull / degenerate."""
    values = np.asarray(values, dtype=float)
    fin = np.isfinite(values)
    n = int(fin.sum())
    if n == 0:
        return np.zeros(targets.shape[0], dtype=float)
    if n == 1:
        return np.full(targets.shape[0], float(values[fin][0]), dtype=float)
    pts, vals = points[fin], values[fin]
    nearest = griddata(pts, vals, targets, method="nearest")
    if n < 4:
        return nearest
    try:
        out = griddata(pts, vals, targets, method="linear")
    except Exception:
        return nearest
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = nearest[bad]
    return out


def _close(sw, so, sg):
    sw = np.clip(sw, 0.0, 1.0)
    so = np.clip(so, 0.0, 1.0)
    sg = np.clip(sg, 0.0, 1.0)
    total = sw + so + sg
    total = np.where(total > 0.0, total, 1.0)
    return sw / total, so / total, sg / total


# --------------------------------------------------------------------------
# demo session
# --------------------------------------------------------------------------
class LinearDemo:
    def __init__(self, case_path, ip, lab_port, field_port):
        self.cfg = _load_case(Path(case_path))
        self.ip = ip
        self.lab_port = lab_port
        self.field_port = field_port
        self.nx, self.ny, self.nz = self.cfg["nx"], self.cfg["ny"], self.cfg["nz"]
        self.n_cells = self.nx * self.ny * self.nz
        self.extent = self.cfg["extent"]
        self.origin = self.cfg["origin"]
        self.centers = _cell_centers(self.nx, self.ny, self.nz, self.extent, self.origin)
        self.probe_cells = _locate_cell(self.cfg["probe_xyz"], self.nx, self.ny, self.nz, self.extent, self.origin)
        self.well_cells = _locate_cell(self.cfg["well_xyz"], self.nx, self.ny, self.nz, self.extent, self.origin)
        self.n_probes = len(self.cfg["probe_ids"])
        self.n_wells = len(self.cfg["well_ids"])
        self.phi = np.full(self.n_cells, self.cfg["phi0"])
        self.k = np.full(self.n_cells, self.cfg["k0"])
        self.times: list[float] = []
        self.p: list[np.ndarray] = []
        self.sw: list[np.ndarray] = []
        self.so: list[np.ndarray] = []
        self.sg: list[np.ndarray] = []
        self.obs_p: list[np.ndarray] = []
        self.obs_sw: list[np.ndarray] = []
        self.obs_so: list[np.ndarray] = []
        self.obs_sg: list[np.ndarray] = []

    def _similarity(self):
        return {
            **_ratios(self.cfg),
            "model": {
                "length_m": MODEL_LENGTH_M, "width_m": MODEL_WIDTH_M,
                "height_m": MODEL_HEIGHT_M, "volume_m3": MODEL_VOLUME_M3,
            },
        }

    def hello(self):
        return {
            "protocol_version": PROTOCOL_VERSION,
            "library": "reservoir-demo",
            "library_version": "0.4.0-demo",
            "n_probes": self.n_probes,
            "n_wells": self.n_wells,
            "probe_ids": self.cfg["probe_ids"],
            "probe_xyz": [[float(v) for v in xyz] for xyz in self.cfg["probe_xyz"]],
            "well_ids": self.cfg["well_ids"],
            "well_xyz": [[float(v) for v in xyz] for xyz in self.cfg["well_xyz"]],
            "grid": {
                "nx": self.nx, "ny": self.ny, "nz": self.nz,
                "origin": list(self.origin), "extent": list(self.extent),
                "n_cells": self.n_cells,
                "cell_order": "k*ny*nx + j*nx + i",
                "shape_ijk": [self.nz, self.ny, self.nx],
            },
            "similarity": self._similarity(),
            "quantity_enum": QUANTITY_NAMES,
            "quantity_units": QUANTITY_UNITS,
            "well_kind_enum": KIND_NAMES,
        }

    def _manifest(self, grid, times, scale):
        return {
            "protocol_version": PROTOCOL_VERSION,
            "library": "reservoir-demo",
            "library_version": "0.4.0-demo",
            "scale": scale,
            "nx": grid["nx"], "ny": grid["ny"], "nz": grid["nz"],
            "origin": grid["origin"], "extent": grid["extent"], "n_cells": grid["n_cells"],
            "cell_order": "k*ny*nx + j*nx + i",
            "shape_ijk": grid["shape_ijk"],
            "times": [float(v) for v in times],
            "fields": [{"field_id": fid, "name": name, "unit": FIELD_UNITS[name]} for name, fid in FIELD_IDS.items()],
            "similarity": self._similarity(),
        }

    def _decode(self, step):
        pressure = np.full(self.n_probes, np.nan)
        sw = np.full(self.n_probes, np.nan)
        so = np.full(self.n_probes, np.nan)
        sg = np.full(self.n_probes, np.nan)
        well_pw = np.full(self.n_wells, np.nan)
        for idx, qty, val in step.probe_records:
            if not (0 <= idx < self.n_probes):
                continue
            if qty == PROBE_Q_P:
                pressure[idx] = val
            elif qty == PROBE_Q_SW:
                sw[idx] = val
            elif qty == PROBE_Q_SO:
                so[idx] = val
            elif qty == PROBE_Q_SG:
                sg[idx] = val
        for idx, kind, val in step.well_records:
            if 0 <= idx < self.n_wells and kind == WELL_K_PW:
                well_pw[idx] = val
        return pressure, sw, so, sg, well_pw

    def accept(self, step):
        pressure, sw, so, sg, well_pw = self._decode(step)
        p = _linear_interp(
            np.vstack([self.cfg["probe_xyz"], self.cfg["well_xyz"]]),
            np.concatenate([pressure, well_pw]),
            self.centers,
        )
        swf = _linear_interp(self.cfg["probe_xyz"], sw, self.centers)
        sof = _linear_interp(self.cfg["probe_xyz"], so, self.centers)
        sgf = _linear_interp(self.cfg["probe_xyz"], sg, self.centers)
        swf, sof, sgf = _close(swf, sof, sgf)
        self.times.append(float(step.time_s))
        self.p.append(p)
        self.sw.append(swf)
        self.so.append(sof)
        self.sg.append(sgf)
        self.obs_p.append(pressure)
        self.obs_sw.append(sw)
        self.obs_so.append(so)
        self.obs_sg.append(sg)
        # 滑动窗口：只保留最近 _WINDOW 拍，避免大网格/长会话无限累积
        if len(self.times) > _WINDOW:
            self.times = self.times[-_WINDOW:]
            self.p = self.p[-_WINDOW:]
            self.sw = self.sw[-_WINDOW:]
            self.so = self.so[-_WINDOW:]
            self.sg = self.sg[-_WINDOW:]
            self.obs_p = self.obs_p[-_WINDOW:]
            self.obs_sw = self.obs_sw[-_WINDOW:]
            self.obs_so = self.obs_so[-_WINDOW:]
            self.obs_sg = self.obs_sg[-_WINDOW:]
        self.publish()

    def _field_grid(self):
        r = _ratios(self.cfg)
        origin = tuple(o * s for o, s in zip(self.origin, (r["lambda_x"], r["lambda_y"], r["lambda_z"])))
        extent = tuple(e * s for e, s in zip(self.extent, (r["lambda_x"], r["lambda_y"], r["lambda_z"])))
        return {"nx": self.nx, "ny": self.ny, "nz": self.nz, "origin": list(origin),
                "extent": list(extent), "n_cells": self.n_cells, "shape_ijk": [self.nz, self.ny, self.nx]}

    def _results(self, times):
        n_t = len(self.times)
        p = np.array(self.p) if n_t else np.zeros((0, self.n_cells))
        sw = np.array(self.sw) if n_t else np.zeros((0, self.n_cells))
        so = np.array(self.so) if n_t else np.zeros((0, self.n_cells))
        sg = np.array(self.sg) if n_t else np.zeros((0, self.n_cells))
        probes = []
        for n, pid in enumerate(self.cfg["probe_ids"]):
            cell = int(self.probe_cells[n])
            rmse = {}
            for key, obs_list, rec in (
                ("pressure_pa", self.obs_p, p),
                ("sw", self.obs_sw, sw),
                ("so", self.obs_so, so),
                ("sg", self.obs_sg, sg),
            ):
                obs = np.array(obs_list)[:, n] if n_t else np.zeros(0)
                r = rec[:, cell] if n_t else np.zeros(0)
                m = np.isfinite(obs) & np.isfinite(r)
                rmse[key] = float(np.sqrt(np.mean((obs[m] - r[m]) ** 2))) if m.any() else float("nan")
            probes.append({"id": str(pid), "cell": cell, "rmse": rmse})
        wells = []
        for n, wid in enumerate(self.cfg["well_ids"]):
            cell = int(self.well_cells[n])
            bhp = p[:, cell] if n_t else np.zeros(n_t)
            wells.append({"id": str(wid), "kind": self.cfg["well_kinds"][n], "n_cells": 1,
                          "bhp_reconstructed_pa": [float(v) for v in bhp]})
        return {"n_times": n_t, "times_s": [float(v) for v in times], "probes": probes, "wells": wells,
                "diagnostics": {"demo": "linear interpolation"}}

    def publish(self):
        times = np.array(self.times, dtype=float)
        n_t = len(self.times)
        shape = (n_t, self.n_cells)
        p = np.array(self.p) if n_t else np.zeros(shape)
        sw = np.array(self.sw) if n_t else np.zeros(shape)
        so = np.array(self.so) if n_t else np.zeros(shape)
        sg = np.array(self.sg) if n_t else np.zeros(shape)
        phi = np.repeat(self.phi[None, :], n_t, axis=0) if n_t else np.zeros(shape)
        k = np.repeat(self.k[None, :], n_t, axis=0) if n_t else np.zeros(shape)
        stream_id = os.getpid() & 0xFFFFFFFF
        if self.lab_port:
            _publish(self.ip, self.lab_port, self._manifest(self._lab_grid(), times, "lab"),
                     times, p, sw, so, sg, phi, k, self._results(times), stream_id)
        if self.field_port:
            fgrid = self._field_grid()
            ftimes = times * float(_ratios(self.cfg)["c_t"])
            _publish(self.ip, self.field_port, self._manifest(fgrid, ftimes, "field"),
                     ftimes, p, sw, so, sg, phi, k, self._results(ftimes), stream_id)

    def _lab_grid(self):
        return {"nx": self.nx, "ny": self.ny, "nz": self.nz, "origin": list(self.origin),
                "extent": list(self.extent), "n_cells": self.n_cells, "shape_ijk": [self.nz, self.ny, self.nx]}


def _publish(host, port, manifest, times, p, sw, so, sg, phi, k, results, stream_id):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = 0
    total = 0
    running = 0

    def send(datagram):
        nonlocal seq, total
        sock.sendto(datagram, (host, port))
        seq += 1
        total += 1

    send(udp_header(UDP_MANIFEST, stream_id, seq) + dumps(manifest))
    n_t = int(times.size)
    for t in range(n_t):
        for fid, arr in ((0, p), (1, sw), (2, so), (3, sg), (4, phi), (5, k)):
            vals = np.asarray(arr[t], dtype=np.float64).ravel()
            for start in range(0, vals.size, _VALUES_PER_DATAGRAM):
                chunk = vals[start:start + _VALUES_PER_DATAGRAM]
                raw = chunk.tobytes()
                running = crc32(raw, running)
                send(udp_header(UDP_FIELD, stream_id, seq) + pack_field_chunk(t, fid, start, chunk))
    if results is not None:
        send(udp_header(UDP_RESULTS, stream_id, seq) + dumps(results))
    send(udp_header(UDP_END, stream_id, seq) + pack_end(total + 1, running))
    sock.close()


# --------------------------------------------------------------------------
# TCP server
# --------------------------------------------------------------------------
class DemoHandler:
    def __init__(self, demo: LinearDemo):
        self.demo = demo
        self.ready = False

    def on_connect(self, host, port):
        self.ready = False
        print(json.dumps({"tcp": "connected", "peer": f"{host}:{port}"}), flush=True)
        return [json_frame(TCP_HELLO, self.demo.hello())]

    def on_message(self, version, msg_type, payload):
        if version != PROTOCOL_VERSION:
            return [json_frame(TCP_NAK, {"seq": None, "code": NAK_BAD_VERSION, "message": f"unsupported version {version}"})]
        if msg_type == TCP_READY:
            self.ready = True
            return []
        if msg_type == TCP_STEP:
            if not self.ready:
                return [json_frame(TCP_NAK, {"seq": None, "code": NAK_NOT_READY, "message": "STEP before READY"})]
            return self._step(payload)
        return []

    def _step(self, payload):
        try:
            step = unpack_step(payload)
        except ValueError as exc:
            return [json_frame(TCP_NAK, {"seq": None, "code": NAK_TRUNCATED, "message": str(exc)})]
        self.demo.accept(step)
        return [json_frame(TCP_ACK, {"seq": step.seq, "time_s": step.time_s, "accepted": True})]

    def on_disconnect(self):
        print(json.dumps({"tcp": "disconnected"}), flush=True)


def _serve(port, handler):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", int(port)))
    server.listen(1)
    while True:
        conn, addr = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            for frame in handler.on_connect(str(addr[0]), int(addr[1])) or ():
                conn.sendall(frame)
            while True:
                fr = read_frame(conn)
                if fr is None:
                    break
                version, msg_type, payload = fr
                if msg_type == TCP_BYE:
                    break
                for reply in handler.on_message(version, msg_type, payload) or ():
                    conn.sendall(reply)
        finally:
            conn.close()
            handler.on_disconnect()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Demo backend: frozen protocol v2 + linear interpolation")
    parser.add_argument("case", type=Path, help="path to case.yaml (grid/probes/wells only)")
    parser.add_argument("--tcp-port", type=int, required=True, metavar="PORT")
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--lab-port", type=int, default=None, metavar="PORT")
    parser.add_argument("--field-port", type=int, default=None, metavar="PORT")
    args = parser.parse_args(argv)

    demo = LinearDemo(args.case, args.ip, args.lab_port, args.field_port)
    print(json.dumps({
        "mode": "tcp", "protocol_version": PROTOCOL_VERSION, "tcp_port": args.tcp_port,
        "demo": True, "core": "linear interpolation",
    }), flush=True)
    try:
        _serve(int(args.tcp_port), DemoHandler(demo))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
