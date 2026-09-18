"""UDP receiver (protocol v2): bind one port, verify and print each stream.

Reassembles the versioned/sequenced/checksummed stream from
``src.programs.udp.receive_fields``, which raises ``IncompleteStreamError`` on
any missing or corrupt datagram (instead of silently producing a wrong field).
Prints the manifest, per-field shapes/means, a 3D reshape of the first field,
and the per-well / per-probe results block.

    python examples/online/recv_fields.py --port 9001 --label lab
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.programs.udp import IncompleteStreamError, receive_fields  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Receive inverted fields over UDP (protocol v2)")
    parser.add_argument("--port", type=int, required=True, help="lab-port or field-port")
    parser.add_argument("--label", default="", help="printed name, e.g. lab or field")
    args = parser.parse_args(argv)
    label = args.label or str(args.port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", int(args.port)))
    print(f"listening UDP {label} on {args.port}", flush=True)
    n = 0
    try:
        while True:
            try:
                manifest, fields, results = receive_fields(int(args.port), timeout=None, sock=sock)
            except IncompleteStreamError as exc:
                print(f"[{label} #{n}] INCOMPLETE: {exc}", flush=True)
                continue
            n += 1
            print(
                f"[{label} #{n}] scale={manifest.get('scale')} "
                f"nx={manifest['nx']} ny={manifest['ny']} nz={manifest['nz']} "
                f"extent_m={manifest['extent']} n_times={len(manifest['times'])}",
                flush=True,
            )
            print(
                f"  cell_order={manifest.get('cell_order')} "
                f"shape_ijk={manifest.get('shape_ijk')} times_s={manifest['times']}",
                flush=True,
            )
            for name, arr in fields.items():
                finite = arr[np.isfinite(arr)]
                mean = float(np.mean(finite)) if finite.size else float("nan")
                print(f"  {name}: shape={arr.shape} mean={mean:.6g}", flush=True)
            # Demonstrate the documented reshape: flat -> (nz, ny, nx).
            first = next(iter(fields))
            arr3 = fields[first].reshape(manifest["shape_ijk"])
            print(f"  reshape({first}) -> {arr3.shape}", flush=True)
            if results is not None:
                _print_results(results)
    except KeyboardInterrupt:
        return 0
    finally:
        sock.close()
    return 0


def _print_results(results: dict) -> None:
    print("  results:", flush=True)
    for w in results.get("wells", []):
        bhp = w.get("bhp_reconstructed_pa") or []
        last_bhp = bhp[-1] if bhp else float("nan")
        print(f"    well {w['id']} ({w['kind']}): n_cells={w['n_cells']} last_bhp={last_bhp:.3e}", flush=True)
    for p in results.get("probes", []):
        rmse = p.get("rmse", {})
        print(
            f"    probe {p['id']}: p_rmse={rmse.get('pressure_pa', float('nan')):.3e} "
            f"sg_rmse={rmse.get('sg', float('nan')):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
