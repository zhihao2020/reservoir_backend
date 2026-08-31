"""In-process command queue. UDP only enqueues; Newton runs on drain."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeCommand:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)


class CommandQueue:
    def __init__(self) -> None:
        self._items: deque[RuntimeCommand] = deque()

    def push(self, name: str, payload: dict[str, Any] | None = None) -> RuntimeCommand:
        cmd = RuntimeCommand(name=str(name), payload=dict(payload or {}))
        self._items.append(cmd)
        return cmd

    def pop(self) -> RuntimeCommand | None:
        if not self._items:
            return None
        return self._items.popleft()

    def __len__(self) -> int:
        return len(self._items)
