"""Protocol v2: versioned, sequenced, checksummed framing shared by TCP and UDP.

This module is the single source of truth for the wire format. Little-endian
throughout. It defines the magic bytes, message-type enums, the probe/well
record encoding, the TCP length-prefixed frame, and the UDP datagram header
(with per-process ``stream_id`` and monotonic ``seq`` so a receiver can detect
loss or reordering).

Message payloads are either JSON (UTF-8) — used for low-frequency control and
metadata (HELLO / READY / ACK / NAK, MANIFEST, RESULTS) — or packed binary —
used for the high-volume STEP records and FIELD chunks. JSON makes the
metadata self-describing and easy to parse from any language; binary keeps the
bulk transfer compact.

See ``docs/接口协议.md`` for the byte-level reference that matches this module.
"""

from __future__ import annotations

import json
import socket
import struct
import zlib
from dataclasses import dataclass

PROTOCOL_VERSION = 2

#: Datagram magic — retained from v1 so a v2 receiver can still recognise the
#: stream family (the version byte immediately follows and disambiguates).
MAGIC = b"RB"

# --- TCP message types (length-prefixed, bidirectional) -------------------
TCP_HELLO = 0x01  # backend -> collector: JSON manifest (grid/ids/enums/units)
TCP_READY = 0x02  # collector -> backend: JSON {version, ok, warnings}
TCP_STEP = 0x03   # collector -> backend: binary [seq][time][records]
TCP_ACK = 0x04    # backend -> collector: JSON {seq, time_s, accepted}
TCP_NAK = 0x05    # backend -> collector: JSON {seq, code, message}
TCP_BYE = 0x06    # either: empty payload, graceful close

# --- UDP message types (unidirectional, each datagram self-contained) ------
UDP_MANIFEST = 0x01  # backend -> dashboard: JSON metadata (grid/times/fields)
UDP_FIELD = 0x02     # backend -> dashboard: binary field chunk + crc32
UDP_RESULTS = 0x03   # backend -> dashboard: JSON per-well/per-probe results
UDP_END = 0x04       # backend -> dashboard: binary total-datagram count + crc32
UDP_RESEND = 0x05    # dashboard -> backend: request re-send of latest stream

# --- probe quantity enum (a record's payload is one of these) -------------
PROBE_Q_P = 1
PROBE_Q_SW = 2
PROBE_Q_SO = 3
PROBE_Q_SG = 4

QUANTITY_NAMES = {PROBE_Q_P: "pressure", PROBE_Q_SW: "sw", PROBE_Q_SO: "so", PROBE_Q_SG: "sg"}
QUANTITY_UNITS = {PROBE_Q_P: "Pa", PROBE_Q_SW: "", PROBE_Q_SO: "", PROBE_Q_SG: ""}

# --- well kind enum --------------------------------------------------------
WELL_K_PW = 1       # bottom-hole pressure, Pa
WELL_K_INJ = 2      # injection (CO2), m3/s — carried as gas
WELL_K_PROD_W = 3   # produced water, m3/s
WELL_K_PROD_O = 4   # produced oil, m3/s
WELL_K_PROD_G = 5   # produced gas, m3/s

KIND_NAMES = {
    WELL_K_PW: "bhp",
    WELL_K_INJ: "inject_co2",
    WELL_K_PROD_W: "prod_water",
    WELL_K_PROD_O: "prod_oil",
    WELL_K_PROD_G: "prod_gas",
}

# --- UDP field_id enum ------------------------------------------------------
FIELD_IDS = {"p": 0, "sw": 1, "so": 2, "sg": 3, "phi": 4, "k": 5}
ID_FIELDS = {v: k for k, v in FIELD_IDS.items()}
FIELD_UNITS = {"p": "Pa", "sw": "", "so": "", "sg": "", "phi": "", "k": "m2"}

# --- NAK reason codes -------------------------------------------------------
NAK_BAD_VERSION = "bad_version"
NAK_NOT_READY = "not_ready"
NAK_BAD_SEQUENCE = "bad_sequence"
NAK_BAD_TIME = "bad_time"
NAK_BAD_INDEX = "bad_index"
NAK_BAD_QUANTITY = "bad_quantity"
NAK_BAD_KIND = "bad_kind"
NAK_BAD_DATA = "bad_data"
NAK_TRUNCATED = "truncated"

# --- binary layouts (all little-endian) ------------------------------------
_LEN = struct.Struct("<I")          # TCP frame length prefix (u32)
_REC = struct.Struct("<HBd")        # one probe/well record: index(u16) code(u8) value(f64)
_STEP_HEAD = struct.Struct("<IdHH")  # seq(u32) time_s(f64) n_probe(u16) n_well(u16)
_UDP_HEAD = struct.Struct("<2sBBII")  # magic(2s) version(u8) type(u8) stream_id(u32) seq(u32)
_FIELD_HEAD = struct.Struct("<iBii")  # t(i32) field_id(u8) offset(i32) n_values(i32)
_FIELD_TAIL = struct.Struct("<I")     # chunk crc32(u32)
_END_HEAD = struct.Struct("<II")      # total_datagrams(u32) stream crc32(u32)

_MAX_FRAME = 1_048_576             # 1 MiB TCP frame ceiling
_VALUES_PER_DATAGRAM = 7000        # 7000 f64 = 56 000 bytes of payload
_MAX_DATAGRAM = 60000

_JSON_INDENT = None


def crc32(data: bytes, value: int = 0) -> int:
    """Unsigned 32-bit CRC of ``data`` (zlib polynomial, the ``binascii.crc32`` one).

    Pass ``value`` to continue a running CRC over concatenated chunks.
    """
    return zlib.crc32(data, int(value)) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# TCP framing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StepFrame:
    """One decoded TCP STEP payload (probe/well records as raw tuples)."""

    seq: int
    time_s: float
    probe_records: tuple[tuple[int, int, float], ...]
    well_records: tuple[tuple[int, int, float], ...]


def pack_step(
    seq: int,
    time_s: float,
    probe_records: list[tuple[int, int, float]],
    well_records: list[tuple[int, int, float]],
) -> bytes:
    """Build one STEP payload (binary, without the frame header)."""
    body = _STEP_HEAD.pack(int(seq), float(time_s), len(probe_records), len(well_records))
    for index, quantity, value in probe_records:
        body += _REC.pack(int(index), int(quantity), float(value))
    for index, kind, value in well_records:
        body += _REC.pack(int(index), int(kind), float(value))
    return body


def unpack_step(payload: bytes) -> StepFrame:
    """Decode a STEP payload. Raises ``ValueError`` when truncated."""
    if len(payload) < _STEP_HEAD.size:
        raise ValueError("STEP payload shorter than header")
    seq, time_s, n_pr, n_wr = _STEP_HEAD.unpack_from(payload, 0)
    need = _STEP_HEAD.size + (n_pr + n_wr) * _REC.size
    if len(payload) < need:
        raise ValueError("STEP payload truncated")
    off = _STEP_HEAD.size
    probes: list[tuple[int, int, float]] = []
    wells: list[tuple[int, int, float]] = []
    for _ in range(n_pr):
        index, quantity, value = _REC.unpack_from(payload, off)
        off += _REC.size
        probes.append((index, quantity, value))
    for _ in range(n_wr):
        index, kind, value = _REC.unpack_from(payload, off)
        off += _REC.size
        wells.append((index, kind, value))
    return StepFrame(seq=int(seq), time_s=float(time_s), probe_records=tuple(probes), well_records=tuple(wells))


def frame_tcp(msg_type: int, payload: bytes = b"") -> bytes:
    """Wrap ``payload`` into a length-prefixed TCP frame: ``[u32 len][ver][type][payload]``.

    ``len`` counts ``version + type + payload`` bytes.
    """
    body = bytes([PROTOCOL_VERSION, int(msg_type)]) + payload
    return _LEN.pack(len(body)) + body


def json_frame(msg_type: int, obj) -> bytes:
    """Convenience: JSON-encode ``obj`` and frame it."""
    return frame_tcp(msg_type, dumps(obj))


def dumps(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def loads(payload: bytes):
    return json.loads(payload.decode("utf-8"))


def _recvall(conn: socket.socket, n: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < n:
        piece = conn.recv(n - len(chunks))
        if not piece:
            return None
        chunks.extend(piece)
    return bytes(chunks)


def read_frame(conn: socket.socket) -> tuple[int, int, bytes] | None:
    """Read one TCP frame → ``(version, msg_type, payload)``, or ``None`` on EOF."""
    header = _recvall(conn, _LEN.size)
    if header is None:
        return None
    (nbytes,) = _LEN.unpack(header)
    if nbytes < 2 or nbytes > _MAX_FRAME:
        raise ValueError(f"illegal TCP frame length {nbytes}")
    body = _recvall(conn, int(nbytes))
    if body is None:
        raise ValueError("TCP disconnect mid-frame")
    return int(body[0]), int(body[1]), body[2:]


def write_frame(conn: socket.socket, msg_type: int, payload: bytes = b"") -> None:
    conn.sendall(frame_tcp(msg_type, payload))


# ---------------------------------------------------------------------------
# UDP datagram helpers
# ---------------------------------------------------------------------------
def udp_header(msg_type: int, stream_id: int, seq: int) -> bytes:
    return _UDP_HEAD.pack(MAGIC, PROTOCOL_VERSION, int(msg_type), int(stream_id), int(seq))


def pack_field_chunk(t: int, field_id: int, offset: int, values) -> bytes:
    """Pack one FIELD datagram body (header + values + crc32 of the values)."""
    arr = _as_f64(values)
    head = _FIELD_HEAD.pack(int(t), int(field_id), int(offset), int(arr.size))
    raw = arr.tobytes()
    return head + raw + _FIELD_TAIL.pack(crc32(raw))


def unpack_field_chunk(payload: bytes) -> tuple[int, int, int, bytes, int]:
    """Decode a FIELD body → ``(t, field_id, offset, raw_values_bytes, crc_ok)``."""
    if len(payload) < _FIELD_HEAD.size + _FIELD_TAIL.size:
        raise ValueError("FIELD datagram truncated")
    t, field_id, offset, n = _FIELD_HEAD.unpack_from(payload, 0)
    raw = payload[_FIELD_HEAD.size : _FIELD_HEAD.size + n * 8]
    if len(raw) != n * 8:
        raise ValueError("FIELD datagram value bytes truncated")
    got = _FIELD_TAIL.unpack_from(payload, _FIELD_HEAD.size + n * 8)[0]
    ok = int(crc32(raw)) == int(got)
    return int(t), int(field_id), int(offset), raw, ok


def pack_end(total_datagrams: int, stream_crc: int) -> bytes:
    return _END_HEAD.pack(int(total_datagrams), int(stream_crc))


def unpack_end(payload: bytes) -> tuple[int, int]:
    if len(payload) < _END_HEAD.size:
        raise ValueError("END datagram truncated")
    total, crc = _END_HEAD.unpack_from(payload, 0)
    return int(total), int(crc)


def _as_f64(values):
    import numpy as np

    return np.asarray(values, dtype=np.float64).ravel()
