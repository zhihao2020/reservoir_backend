"""TCP collector (protocol v2): HELLO → READY → STEP/ACK/NAK loop.

Reads ``observations.csv`` / ``series.csv`` and sends one length-prefixed STEP
per time, waiting for an ACK/NAK after each. The HELLO manifest from the backend
is printed so the operator can confirm probe/well ordering and counts before any
data flows.

Self-contained: only imports the local ``protocol.py`` (a copy of the frozen
wire framing).

    python send_steps.py --host 127.0.0.1 --port 9000
"""

from __future__ import annotations

import argparse
import csv
import socket
import sys
from collections import defaultdict
from pathlib import Path

from protocol import (
    PROBE_Q_P,
    PROBE_Q_SG,
    PROBE_Q_SO,
    PROBE_Q_SW,
    PROTOCOL_VERSION,
    TCP_ACK,
    TCP_BYE,
    TCP_HELLO,
    TCP_NAK,
    TCP_READY,
    TCP_STEP,
    WELL_K_INJ,
    WELL_K_PROD_G,
    WELL_K_PROD_O,
    WELL_K_PROD_W,
    WELL_K_PW,
    frame_tcp,
    json_frame,
    loads,
    pack_step,
    read_frame,
)

HERE = Path(__file__).resolve().parent

_QMAP = {
    "pressure": PROBE_Q_P,
    "p": PROBE_Q_P,
    "sw": PROBE_Q_SW,
    "swtest": PROBE_Q_SW,
    "so": PROBE_Q_SO,
    "sotest": PROBE_Q_SO,
    "sg": PROBE_Q_SG,
    "sgtest": PROBE_Q_SG,
}
_RATE_EPS = 1.0e-18


def _read_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [str(r.get("id") or r.get("name")) for r in rows]


def _load_probe_steps(path: Path, probe_ids: list[str]) -> dict[float, list[tuple[int, int, float]]]:
    index = {name: n for n, name in enumerate(probe_ids)}
    out: dict[float, list[tuple[int, int, float]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            t = float(row["time_s"] if "time_s" in row else row["time"])
            name = str(row.get("probe") or row.get("id"))
            qty = str(row.get("quantity") or row.get("kind") or "").strip().lower()
            if name not in index or qty not in _QMAP:
                continue
            out[t].append((index[name], _QMAP[qty], float(row["value"])))
    return dict(out)


def _load_well_steps(path: Path, well_ids: list[str]) -> dict[float, list[tuple[int, int, float]]]:
    index = {name: n for n, name in enumerate(well_ids)}
    out: dict[float, list[tuple[int, int, float]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            t = float(row["time_s"] if "time_s" in row else row["time"])
            name = str(row.get("well") or row.get("id"))
            if name not in index:
                continue
            j = index[name]
            pw = row.get("pw_pa") or row.get("pw")
            if pw not in (None, ""):
                out[t].append((j, WELL_K_PW, float(pw)))
            q = float(row["q_m3s"] or 0.0) if row.get("q_m3s") not in (None, "") else 0.0
            qw = float(row["qw_m3s"] or 0.0) if row.get("qw_m3s") not in (None, "") else 0.0
            qo = float(row["qo_m3s"] or 0.0) if row.get("qo_m3s") not in (None, "") else 0.0
            qg = float(row["qg_m3s"] or 0.0) if row.get("qg_m3s") not in (None, "") else 0.0
            if q > _RATE_EPS:
                out[t].append((j, WELL_K_INJ, q))
            if qw < -_RATE_EPS:
                out[t].append((j, WELL_K_PROD_W, -qw))
            if qo < -_RATE_EPS:
                out[t].append((j, WELL_K_PROD_O, -qo))
            if qg < -_RATE_EPS:
                out[t].append((j, WELL_K_PROD_G, -qg))
    return dict(out)


def steps(
    probes_csv: Path,
    wells_csv: Path,
    observations_csv: Path,
    series_csv: Path,
) -> list[tuple[float, list[tuple[int, int, float]], list[tuple[int, int, float]]]]:
    probe_ids = _read_ids(probes_csv)
    well_ids = _read_ids(wells_csv)
    probes = _load_probe_steps(observations_csv, probe_ids)
    wells = _load_well_steps(series_csv, well_ids)
    times = sorted(set(probes) | set(wells))
    return [(t, probes.get(t, []), wells.get(t, [])) for t in times]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send lab probe/well steps over TCP (protocol v2)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--probes", type=Path, default=HERE / "probes.csv")
    parser.add_argument("--wells", type=Path, default=HERE / "wells.csv")
    parser.add_argument("--observations", type=Path, default=HERE / "observations.csv")
    parser.add_argument("--series", type=Path, default=HERE / "series.csv")
    args = parser.parse_args(argv)
    packed = steps(args.probes, args.wells, args.observations, args.series)
    if not packed:
        print("no observation/series times", file=sys.stderr)
        return 1

    sock = socket.create_connection((args.host, int(args.port)))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        version, msg_type, payload = read_frame(sock)
        if msg_type != TCP_HELLO:
            print(f"expected HELLO, got type {msg_type}", file=sys.stderr)
            return 1
        manifest = loads(payload)
        print(
            f"HELLO v{manifest.get('protocol_version')}: "
            f"n_probes={manifest.get('n_probes')} n_wells={manifest.get('n_wells')} "
            f"grid={manifest.get('grid', {}).get('nx')}x"
            f"{manifest.get('grid', {}).get('ny')}x{manifest.get('grid', {}).get('nz')}",
            flush=True,
        )
        sock.sendall(json_frame(TCP_READY, {"version": PROTOCOL_VERSION, "ok": True}))
        seq = 0
        for time_s, probes, wells in packed:
            sock.sendall(frame_tcp(TCP_STEP, pack_step(seq, time_s, probes, wells)))
            _version, _msg_type, reply = read_frame(sock)
            if _msg_type == TCP_ACK:
                ack = loads(reply)
                print(f"ACK seq={ack.get('seq')} time_s={ack.get('time_s')}", flush=True)
            elif _msg_type == TCP_NAK:
                nak = loads(reply)
                print(f"NAK seq={nak.get('seq')} code={nak.get('code')} {nak.get('message')}", file=sys.stderr, flush=True)
            else:
                print(f"unexpected reply type {_msg_type}", file=sys.stderr, flush=True)
            seq += 1
        sock.sendall(frame_tcp(TCP_BYE))
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
