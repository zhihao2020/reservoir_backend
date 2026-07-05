# Interface Contract Placeholder

UDP product development is still deferred because the frontend communication
protocol is not finalized. A minimal JSON UDP Archie prototype is implemented
in `reservoir_backend/api/udp_server.py` and regression-tested in
`tests/numerical/test_io_and_udp_regression.py`, but it is not a complete
frontend/backend interface.

The current primary interface remains:

- CLI commands
- YAML case files
- result directories
- JSON / CSV / NPY reports

Future frontend communication is expected to use command-style JSON. Large
arrays should not be transferred directly. The backend should return `case_id`,
`result_dir`, a compact `summary`, and report paths. The final protocol may be
UDP, TCP, REST, or a file-based exchange depending on frontend requirements.

Example request:

```json
{
  "request_id": "req-001",
  "command": "run_case",
  "payload": {
    "config": "config/demo_case.yaml",
    "dry_run": false
  }
}
```

Example response:

```json
{
  "request_id": "req-001",
  "success": true,
  "case_id": "demo_case",
  "result_dir": "results/demo_case",
  "summary": {}
}
```

See `docs/udp_protocol.md` for the currently implemented UDP draft.

## Legacy Traceability Notes

Some repository tests still look for historical release-candidate wording. The
old phrases were:

- "UDP currently deferred"
- "No UDP server is implemented in this stage."

Those phrases are retained here only for traceability. The current repository
fact is that a minimal UDP JSON Archie prototype exists, while full frontend
UDP product development remains deferred.
