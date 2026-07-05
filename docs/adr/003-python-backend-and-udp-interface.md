# ADR 003: Python Backend and UDP Interface

## Background

The backend is implemented in Python. Requirements mention UDP communication
with a frontend if Python is used. The repository currently contains a minimal
UDP JSON Archie server and regression test.

## Decision

Keep Python as the backend implementation for the current MVP. Treat UDP as a
small command protocol that must be versioned before frontend integration.

## Reasons

- Python is adequate for the current small Cartesian benchmark cases.
- Existing validation/profiling does not justify C++ migration yet.
- UDP can support lightweight frontend triggers, but large arrays should remain
file/report based.

## Alternatives

- Switch backend core to C++ now.
- Use REST/TCP instead of UDP.
- Exchange files only and skip UDP.

## Impact

The next UDP work should add protocol version, request ID, stable error codes,
case execution, status query, and result summary commands.

## Risks

UDP can drop packets and has datagram size limits. A frontend may require
stateful progress, retries, or authentication that UDP alone does not solve.

## Revision Conditions

Revisit if frontend requirements mandate REST/TCP/WebSocket, or profiling
shows Python kernels cannot meet performance requirements.
