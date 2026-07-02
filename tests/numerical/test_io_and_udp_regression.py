from __future__ import annotations

import json
import socket
from pathlib import Path

import numpy as np

from reservoir_backend.api.udp_server import UDPArchieServer
from reservoir_backend.io.result_manager import load_field_npz, save_field_npz

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "regression" / "references"


def test_field3d_npz_roundtrip(tmp_path: Path) -> None:
    reference = load_field_npz(REFERENCE_DIR / "field3d_roundtrip_reference.npz")
    output = tmp_path / "field_roundtrip.npz"

    save_field_npz(reference, output)
    loaded = load_field_npz(output)

    assert loaded.grid == reference.grid
    assert loaded.values.shape == reference.values.shape
    assert loaded.unit == reference.unit
    assert loaded.name == reference.name
    assert np.allclose(loaded.values, reference.values)
    assert loaded.confidence is not None
    assert np.allclose(loaded.confidence, reference.confidence)


def test_udp_archie_compute_roundtrip() -> None:
    with (REFERENCE_DIR / "udp_archie_compute_roundtrip.json").open("r", encoding="utf-8") as f:
        fixture = json.load(f)

    server = UDPArchieServer()
    server.start()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            client.settimeout(2.0)
            client.sendto(json.dumps(fixture["request"]).encode("utf-8"), server.address)
            raw, _ = client.recvfrom(8192)
        response = json.loads(raw.decode("utf-8"))
    finally:
        server.stop()

    assert response["status"] == "ok"
    assert response["unit"] == "fraction"
    assert np.allclose(response["sw"], fixture["expected"]["sw"], rtol=1e-12, atol=1e-12)
    assert np.allclose(
        response["confidence"],
        fixture["expected"]["confidence"],
        rtol=0.0,
        atol=0.0,
    )
    assert len(json.dumps(response).encode("utf-8")) < fixture["max_response_bytes"]
