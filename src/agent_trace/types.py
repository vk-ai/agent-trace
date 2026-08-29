from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Kind = Literal["llm", "tool", "retrieve", "chain"]


@dataclass
class Span:
    name: str
    kind: Kind
    started_ms: float
    ended_ms: float | None = None
    parent_id: str | None = None
    span_id: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None

    @property
    def latency_ms(self) -> float:
        if self.ended_ms is None:
            return 0.0
        return max(0.0, self.ended_ms - self.started_ms)


class Clock(Protocol):
    def now_ms(self) -> float: ...


class IdFactory(Protocol):
    def new_id(self) -> str: ...
