# UDP Protocol

Status: Draft with minimal implemented prototype

The repository contains a minimal UDP JSON server in
`reservoir_backend/api/udp_server.py`. Older documentation claimed no UDP
server existed; that statement is stale. The current implementation is small
and regression-tested, but it is not a complete frontend/backend protocol.

## Communication Roles

- Backend server: `UDPArchieServer`
- Client: any UDP client that sends UTF-8 JSON datagrams
- Transport: UDP
- Encoding: JSON text encoded as UTF-8

## IP and Port Configuration

`UDPArchieServer` constructor:

```python
UDPArchieServer(host="127.0.0.1", port=0, max_packet_size=8192)
```

- `host`: defaults to loopback `127.0.0.1`
- `port`: defaults to `0`, meaning the OS chooses a free local port
- `max_packet_size`: defaults to `8192` bytes

The bound address is available from:

```python
server.address
```

No YAML or environment-driven UDP configuration exists yet.

## Implemented Request Format

The server expects one JSON object per datagram.

### `ping`

```json
{
  "command": "ping"
}
```

### `archie_compute`

```json
{
  "command": "archie_compute",
  "rt": [10.0, 20.0],
  "rw": 0.5,
  "phi": [0.25, 0.30],
  "a": 1.0,
  "m": 2.0,
  "n": 2.0,
  "swi": 0.2,
  "sor": 0.2,
  "invalid_policy": "raise"
}
```

Required fields for `archie_compute`:

- `command`
- `rt`
- `rw`
- `phi`

Optional fields:

- `a`
- `m`
- `n`
- `swi`
- `sor`
- `invalid_policy`

## Implemented Response Format

### Successful `ping`

```json
{
  "status": "ok",
  "message": "pong"
}
```

### Successful `archie_compute`

```json
{
  "status": "ok",
  "sw": [0.5, 0.4],
  "confidence": [1.0, 1.0],
  "unit": "fraction"
}
```

The actual `sw` and `confidence` values depend on Archie inputs. NumPy arrays
are serialized to JSON lists.

## Error Format

Unknown commands return:

```json
{
  "status": "error",
  "message": "unknown command: <name>"
}
```

Exceptions at the UDP boundary are converted to:

```json
{
  "status": "error",
  "message": "<exception message>"
}
```

There are no stable numeric error codes yet.

## Timeout Handling

Current server behavior:

- Internal socket timeout: `0.1` seconds to allow the background loop to stop.
- Client timeout/retry behavior is not defined by the protocol.
- The regression test client uses a `2.0` second timeout.

Current test:

- `tests/numerical/test_io_and_udp_regression.py`

Current reference fixture:

- `tests/regression/references/udp_archie_compute_roundtrip.json`

## Current Limitations

- No `request_id`
- No protocol version
- No explicit response schema version
- No standardized error code enum
- No run-case command
- No status-query command
- No result download or result path command
- No fragmentation/chunking for large arrays
- No retry/backoff policy
- No authentication or access control
- No frontend contract document beyond this draft

## Recommended Next Protocol Version

Introduce a versioned command envelope:

```json
{
  "protocol_version": "0.1",
  "request_id": "req-0001",
  "command": "run_case",
  "payload": {
    "config": "config/demo_case.yaml",
    "dry_run": false
  }
}
```

Recommended response envelope:

```json
{
  "protocol_version": "0.1",
  "request_id": "req-0001",
  "status": "ok",
  "payload": {
    "case_id": "demo_case",
    "result_dir": "results/demo_case",
    "summary_path": "results/demo_case/case_summary.json"
  }
}
```

Large arrays should be exchanged through result files or a separate binary/file
channel, not directly inside UDP datagrams.
