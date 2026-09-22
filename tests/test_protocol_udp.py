"""UDP protocol v2: field chunk CRC, stream reassembly, and loss detection."""

import socket

import numpy as np
import pytest

from src.programs.protocol import (
    FIELD_IDS,
    ID_FIELDS,
    crc32,
    pack_field_chunk,
    unpack_field_chunk,
)
from src.programs.udp import IncompleteStreamError, UdpPublisher, receive_fields


def _manifest(nx=2, ny=2, nz=2, times=(0.0, 1.0)):
    return {
        "protocol_version": 2,
        "scale": "lab",
        "nx": nx, "ny": ny, "nz": nz,
        "origin": [0.0, 0.0, 0.0],
        "extent": [0.3, 0.3, 0.3],
        "n_cells": nx * ny * nz,
        "cell_order": "k*ny*nx + j*nx + i",
        "shape_ijk": [nz, ny, nx],
        "times": list(times),
        "fields": [{"field_id": v, "name": k, "unit": ""} for k, v in FIELD_IDS.items()],
    }


def _fields(n_cells=8, n_times=2):
    rng = np.random.default_rng(0)
    base = rng.random((n_times, n_cells))
    return {name: base * (i + 1) for i, name in enumerate(ID_FIELDS.values())}


def test_field_chunk_roundtrip_and_crc():
    vals = np.arange(10, dtype=np.float64)
    body = pack_field_chunk(0, 0, 0, vals)
    t, fid, offset, raw, ok = unpack_field_chunk(body)
    assert (t, fid, offset) == (0, 0, 0)
    assert ok is True
    assert np.frombuffer(raw, dtype=np.float64).tolist() == vals.tolist()


def test_field_chunk_crc_detects_corruption():
    vals = np.arange(10, dtype=np.float64)
    body = bytearray(pack_field_chunk(0, 0, 0, vals))
    # corrupt one value byte (skip 13-byte field header)
    body[13 + 3] ^= 0xFF
    _t, _fid, _off, _raw, ok = unpack_field_chunk(bytes(body))
    assert ok is False


def test_receive_fields_end_to_end():
    n_cells, n_times = 8, 2
    manifest = _manifest(nx=2, ny=2, nz=2, times=(0.0, 1.0))
    fields = _fields(n_cells, n_times)
    results = {"n_times": n_times, "probes": [], "wells": []}

    recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv.bind(("127.0.0.1", 0))
    port = recv.getsockname()[1]

    pub = UdpPublisher("127.0.0.1", port, stream_id=12345)
    pub.send_all(manifest, np.array(manifest["times"]), fields["p"], fields["sw"], fields["so"], fields["sg"], fields["phi"], fields["k"], results)
    pub.close()

    got_manifest, got_fields, got_results = receive_fields(port, timeout=5.0, sock=recv)
    recv.close()
    assert got_manifest["n_cells"] == n_cells
    assert got_manifest["shape_ijk"] == [2, 2, 2]
    assert set(got_fields) == set(ID_FIELDS.values())
    assert got_fields["p"].shape == (n_times, n_cells)
    assert np.allclose(got_fields["p"], fields["p"])
    assert got_results["n_times"] == n_times


def test_receive_fields_reshape_single_slice():
    # Regression: each field is (n_times, n_cells), so the receiver must
    # reshape one time slice, not the whole array (which only works when
    # n_times == 1).
    n_cells, n_times = 8, 3
    manifest = _manifest(nx=2, ny=2, nz=2, times=(0.0, 1.0, 2.0))
    fields = _fields(n_cells, n_times)
    results = {"n_times": n_times, "probes": [], "wells": []}

    recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv.bind(("127.0.0.1", 0))
    port = recv.getsockname()[1]

    pub = UdpPublisher("127.0.0.1", port, stream_id=12345)
    pub.send_all(manifest, np.array(manifest["times"]), fields["p"], fields["sw"], fields["so"], fields["sg"], fields["phi"], fields["k"], results)
    pub.close()

    got_manifest, got_fields, _ = receive_fields(port, timeout=5.0, sock=recv)
    recv.close()
    shape_ijk = tuple(got_manifest["shape_ijk"])
    first = next(iter(got_fields))
    # The whole (n_times, n_cells) array must NOT be reshapeable to shape_ijk.
    assert got_fields[first].size != int(np.prod(shape_ijk))
    # One time slice must reshape cleanly to (nz, ny, nx).
    assert got_fields[first][0].reshape(shape_ijk).shape == shape_ijk


def test_receive_fields_detects_missing_datagram():
    # Publish a manifest + a partial field set, then an END claiming one more
    # datagram than was actually sent -> the receiver must refuse.
    n_cells = 8
    manifest = _manifest(nx=2, ny=2, nz=2, times=(0.0,))
    recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv.bind(("127.0.0.1", 0))
    port = recv.getsockname()[1]

    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    from src.programs.protocol import dumps, pack_end, udp_header, UDP_MANIFEST, UDP_FIELD, UDP_END, UDP_RESULTS

    sender.sendto(udp_header(UDP_MANIFEST, 1, 0) + dumps(manifest), ("127.0.0.1", port))
    vals = np.zeros(n_cells)
    sender.sendto(udp_header(UDP_FIELD, 1, 1) + pack_field_chunk(0, 0, 0, vals), ("127.0.0.1", port))
    sender.sendto(udp_header(UDP_RESULTS, 1, 2) + dumps({"probes": [], "wells": []}), ("127.0.0.1", port))
    # END claims 5 datagrams total (one more than the 4 actually sent).
    sender.sendto(udp_header(UDP_END, 1, 3) + pack_end(5, 0), ("127.0.0.1", port))
    sender.close()

    with pytest.raises(IncompleteStreamError):
        receive_fields(port, timeout=5.0, sock=recv)
    recv.close()


def test_receive_fields_detects_gap_in_field():
    # Send a manifest for 8 cells but only the second half of field 0 -> the
    # offset gap (0..4 missing) must be reported.
    n_cells = 8
    manifest = _manifest(nx=2, ny=2, nz=2, times=(0.0,))
    recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv.bind(("127.0.0.1", 0))
    port = recv.getsockname()[1]
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    from src.programs.protocol import dumps, pack_end, udp_header, UDP_MANIFEST, UDP_FIELD, UDP_END

    sender.sendto(udp_header(UDP_MANIFEST, 1, 0) + dumps(manifest), ("127.0.0.1", port))
    sender.sendto(udp_header(UDP_FIELD, 1, 1) + pack_field_chunk(0, 0, 4, np.zeros(4)), ("127.0.0.1", port))
    sender.sendto(udp_header(UDP_END, 1, 2) + pack_end(3, 0), ("127.0.0.1", port))
    sender.close()

    with pytest.raises(IncompleteStreamError):
        receive_fields(port, timeout=5.0, sock=recv)
    recv.close()
