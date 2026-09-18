"""UDP streaming of full-field time series and results, with loss detection.

Protocol v2 datagrams each carry a ``stream_id`` and a monotonic ``seq`` so a
receiver can detect loss/reordering. Field chunks carry a per-chunk CRC32, and
the stream END datagram carries a total-datagram count plus a stream-wide CRC
over the field value bytes, so a missing or corrupted datagram is reported
instead of silently producing a wrong field.

Field values are flattened in grid C-order (``k*ny*nx + j*nx + i``, z slowest);
the MANIFEST records this as ``cell_order`` / ``shape_ijk`` so the receiver can
reshape to ``(nz, ny, nx)`` without hard-coding the convention.
"""

from __future__ import annotations

import os
import socket

import numpy as np
from numpy.typing import NDArray

from .protocol import (
    FIELD_IDS,
    ID_FIELDS,
    MAGIC,
    UDP_END,
    UDP_FIELD,
    UDP_MANIFEST,
    UDP_RESEND,
    UDP_RESULTS,
    _FIELD_HEAD,
    _FIELD_TAIL,
    _MAX_DATAGRAM,
    _UDP_HEAD,
    _VALUES_PER_DATAGRAM,
    crc32,
    dumps,
    loads,
    pack_end,
    pack_field_chunk,
    udp_header,
    unpack_end,
    unpack_field_chunk,
)


class IncompleteStreamError(ValueError):
    """The received UDP stream is missing or corrupt; reassembly is unreliable."""

    def __init__(self, message: str, missing: list[str] | None = None) -> None:
        self.missing = missing or []
        super().__init__(message)


def parse_host_port(spec: str, default_host: str = "127.0.0.1", default_port: int = 9999) -> tuple[str, int]:
    """Parse ``"HOST:PORT"``, ``":PORT"`` or ``"PORT"`` into ``(host, port)``."""
    s = str(spec).strip()
    if ":" in s:
        host, _, port = s.rpartition(":")
        return (host or default_host), int(port)
    return default_host, (int(s) if s.isdigit() else default_port)


def _field_arrays(t, p, sw, so, sg, phi, k):
    return ((0, p[t]), (1, sw[t]), (2, so[t]), (3, sg[t]), (4, phi[t]), (5, k[t]))


class UdpPublisher:
    """Publishes a versioned, sequenced UDP stream (manifest / fields / results / end)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9999, stream_id: int | None = None):
        self.host = host
        self.port = int(port)
        self.stream_id = int(stream_id) if stream_id is not None else (os.getpid() & 0xFFFFFFFF)
        self._seq = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def close(self) -> None:
        self._sock.close()

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    def _send(self, data: bytes) -> None:
        self._sock.sendto(data, (self.host, self.port))

    def send_manifest(self, manifest: dict) -> None:
        self._send(udp_header(UDP_MANIFEST, self.stream_id, self._next_seq()) + dumps(manifest))

    def send_results(self, results: dict) -> None:
        self._send(udp_header(UDP_RESULTS, self.stream_id, self._next_seq()) + dumps(results))

    def send_field_chunk(self, t: int, field_id: int, offset: int, values: NDArray[np.float64]) -> bytes:
        """Send one FIELD datagram; returns its raw value bytes (for stream CRC)."""
        arr = np.asarray(values, dtype=np.float64).ravel()
        raw = arr.tobytes()
        body = _FIELD_HEAD.pack(int(t), int(field_id), int(offset), int(arr.size)) + raw + _FIELD_TAIL.pack(crc32(raw))
        self._send(udp_header(UDP_FIELD, self.stream_id, self._next_seq()) + body)
        return raw

    def send_end(self, total_datagrams: int, stream_crc: int) -> None:
        self._send(udp_header(UDP_END, self.stream_id, self._next_seq()) + pack_end(total_datagrams, stream_crc))

    def send_all(
        self,
        manifest: dict,
        times: NDArray[np.float64],
        p: NDArray[np.float64],
        sw: NDArray[np.float64],
        so: NDArray[np.float64],
        sg: NDArray[np.float64],
        phi: NDArray[np.float64],
        k: NDArray[np.float64],
        results: dict | None = None,
    ) -> None:
        """Send MANIFEST + all FIELD chunks + optional RESULTS + END.

        The stream CRC is accumulated over the field value bytes in the exact
        send order ``(t, field_id, offset)``, which is also the order the
        receiver reassembles in.
        """
        n_t = int(np.asarray(times, dtype=float).size)
        total = 0
        running = 0
        self.send_manifest(manifest)
        total += 1
        for t in range(n_t):
            for field_id, arr in _field_arrays(t, p, sw, so, sg, phi, k):
                vals = np.asarray(arr, dtype=np.float64).ravel()
                for start in range(0, vals.size, _VALUES_PER_DATAGRAM):
                    raw = self.send_field_chunk(t, field_id, start, vals[start : start + _VALUES_PER_DATAGRAM])
                    running = crc32(raw, running)
                    total += 1
        if results is not None:
            self.send_results(results)
            total += 1
        # total counts every datagram in the stream, END included.
        self.send_end(total + 1, running)


def publish_fields(
    host: str,
    port: int,
    manifest: dict,
    times: NDArray[np.float64],
    p: NDArray[np.float64],
    sw: NDArray[np.float64],
    so: NDArray[np.float64],
    sg: NDArray[np.float64],
    phi: NDArray[np.float64],
    k: NDArray[np.float64],
    results: dict | None = None,
) -> None:
    """One-shot helper: publish manifest + fields + results, then close."""
    pub = UdpPublisher(host, port)
    try:
        pub.send_all(manifest, times, p, sw, so, sg, phi, k, results)
    finally:
        pub.close()


def receive_fields(port: int, timeout: float | None = 5.0, sock: socket.socket | None = None):
    """Receive and verify one published stream → ``(manifest, fields, results)``.

    Raises :class:`IncompleteStreamError` if any datagram is missing (by
    ``seq``/count), any field chunk fails its CRC, or a field does not cover the
    full grid contiguously. ``fields`` maps name → ``(n_times, n_cells)`` float
    array (still flat; reshape with the manifest's ``shape_ijk``).
    """
    owned = sock is None
    if sock is None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", int(port)))
    sock.settimeout(timeout)
    manifest = None
    stream_id = None
    seen: set[int] = set()
    chunks: dict[tuple[int, int], dict[int, tuple[bytes, bool]]] = {}
    results = None
    total_datagrams = None
    expected_crc = None
    try:
        while True:
            data, _addr = sock.recvfrom(_MAX_DATAGRAM)
            if len(data) < _UDP_HEAD.size or data[:2] != MAGIC:
                continue
            _magic, _ver, typ, sid, seq = _UDP_HEAD.unpack_from(data, 0)
            body = data[_UDP_HEAD.size :]
            if typ == UDP_MANIFEST:
                manifest = loads(body)
                stream_id = int(sid)
                seen = {int(seq)}
                chunks = {}
                continue
            if stream_id is not None and int(sid) != stream_id:
                continue
            seen.add(int(seq))
            if typ == UDP_FIELD:
                t, fid, offset, raw, ok = unpack_field_chunk(body)
                if fid not in ID_FIELDS:
                    continue
                chunks.setdefault((int(t), int(fid)), {})[int(offset)] = (raw, ok)
            elif typ == UDP_RESULTS:
                results = loads(body)
            elif typ == UDP_END:
                total_datagrams, expected_crc = unpack_end(body)
                break
    finally:
        if owned:
            sock.close()
    if manifest is None:
        raise TimeoutError("UDP stream ended before a manifest packet")

    n_times = int(manifest.get("n_times", len(manifest.get("times", []))))
    n_cells = int(manifest["n_cells"])
    field_names = [f["name"] for f in manifest.get("fields", [])]
    if not field_names:
        field_names = list(FIELD_IDS)

    problems: list[str] = []
    if total_datagrams is None:
        problems.append("no END datagram")
    elif len(seen) != total_datagrams:
        problems.append(f"expected {total_datagrams} datagrams, received {len(seen)}")
    else:
        expect_seq = set(range(total_datagrams))
        if seen != expect_seq:
            missing = sorted(expect_seq - seen)
            problems.append(f"missing seq numbers: {missing[:10]}")

    fields: dict[str, NDArray[np.float64]] = {}
    running = 0
    for t in range(n_times):
        for fid in range(6):
            name = ID_FIELDS.get(fid)
            if name is None or name not in field_names:
                continue
            parts = chunks.get((t, fid), {})
            covered = 0
            for offset in sorted(parts):
                raw, ok = parts[offset]
                if not ok:
                    problems.append(f"crc mismatch at t={t} field={name} offset={offset}")
                    continue
                running = crc32(raw, running)
                arr = np.frombuffer(raw, dtype=np.float64)
                if arr.size > 0 and offset != covered:
                    problems.append(f"gap at t={t} field={name}: expected offset {covered}, got {offset}")
                covered = offset + arr.size
                if name not in fields:
                    fields[name] = np.full((n_times, n_cells), np.nan)
                fields[name][t, offset : offset + arr.size] = arr
            if covered != n_cells:
                problems.append(f"field {name} t={t} covers {covered}/{n_cells} cells")

    if expected_crc is not None and running != expected_crc:
        problems.append(f"stream crc mismatch: computed {running}, expected {expected_crc}")

    if problems:
        raise IncompleteStreamError("incomplete UDP stream: " + "; ".join(problems[:6]), missing=problems)
    return manifest, fields, results
