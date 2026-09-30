"""UDP receiver (protocol v2): bind lab/field ports and reassemble FIELD streams.

Start this before the backend so datagrams are not dropped.

    python receive_fields.py --lab-port 9001 --field-port 9002
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from src.programs.udp import IncompleteStreamError, receive_fields  # noqa: E402


def _bind(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind(("0.0.0.0", int(port)))
    return sock


def _summarize(scale: str, manifest: dict, fields: dict, results: dict | None) -> dict:
    n_cells = int(manifest.get("n_cells", 0))
    times = manifest.get("times") or []
    out = {
        "scale": scale,
        "stream_id": manifest.get("stream_id"),
        "nx": manifest.get("nx"),
        "ny": manifest.get("ny"),
        "nz": manifest.get("nz"),
        "n_cells": n_cells,
        "n_times": len(times),
        "fields": sorted(fields),
    }
    for name, arr in fields.items():
        finite = np.isfinite(arr)
        out[f"{name}_finite"] = int(finite.sum())
        if finite.any():
            out[f"{name}_mean"] = float(np.mean(arr[finite]))
    if results:
        out["results_n_times"] = results.get("n_times")
        out["n_probes"] = len(results.get("probes") or [])
        out["n_wells"] = len(results.get("wells") or [])
    return out


def _save(out_dir: Path, scale: str, seq: int, manifest: dict, fields: dict, results: dict | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{scale}_{seq:04d}.npz"
    payload = {name: np.asarray(arr) for name, arr in fields.items()}
    payload["_manifest"] = np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8)
    if results is not None:
        payload["_results"] = np.frombuffer(json.dumps(results).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, **payload)
    print(json.dumps({"saved": str(path)}), flush=True)


def _loop(scale: str, port: int, sock: socket.socket, timeout: float, out_dir: Path | None) -> None:
    seq = 0
    print(json.dumps({"udp": "listening", "scale": scale, "port": port}), flush=True)
    while True:
        try:
            manifest, fields, results = receive_fields(port, timeout=timeout, sock=sock)
        except TimeoutError:
            continue
        except IncompleteStreamError as exc:
            print(json.dumps({"scale": scale, "error": str(exc)}), file=sys.stderr, flush=True)
            continue
        except OSError:
            return
        seq += 1
        print(json.dumps(_summarize(scale, manifest, fields, results)), flush=True)
        if out_dir is not None:
            _save(out_dir, scale, seq, manifest, fields, results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Receive inverted lab/field UDP streams (protocol v2)")
    parser.add_argument("--lab-port", type=int, default=9001)
    parser.add_argument("--field-port", type=int, default=9002)
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds to wait for each stream")
    parser.add_argument("-o", "--output", type=Path, default=None, help="optional directory for *.npz")
    args = parser.parse_args(argv)

    lab_sock = _bind(int(args.lab_port))
    field_sock = _bind(int(args.field_port))
    threads = [
        threading.Thread(
            target=_loop,
            args=("lab", int(args.lab_port), lab_sock, float(args.timeout), args.output),
            daemon=True,
        ),
        threading.Thread(
            target=_loop,
            args=("field", int(args.field_port), field_sock, float(args.timeout), args.output),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        return 0
    finally:
        lab_sock.close()
        field_sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
