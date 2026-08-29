from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from typing import Any, Iterator

from .types import Clock, IdFactory, Kind, Span

_CURRENT: ContextVar[str | None] = ContextVar("agent_trace_span", default=None)


class _WallClock:
    def now_ms(self) -> float:
        return time.time() * 1000


class _UuidFactory:
    def new_id(self) -> str:
        return uuid.uuid4().hex[:12]


class Tracer:
    """Records nested spans for one agent run. Zero third-party deps.

    Inject Clock and IdFactory in tests. Production uses wall clock + uuid.
    """

    def __init__(
        self,
        run_id: str,
        model: str = "",
        usd_per_1k_in: float = 0.0,
        usd_per_1k_out: float = 0.0,
        clock: Clock | None = None,
        ids: IdFactory | None = None,
    ) -> None:
        self.run_id = run_id
        self.model = model
        self.usd_per_1k_in = usd_per_1k_in
        self.usd_per_1k_out = usd_per_1k_out
        self._clock = clock or _WallClock()
        self._ids = ids or _UuidFactory()
        self.spans: list[Span] = []

    @contextmanager
    def span(self, name: str, kind: Kind = "chain", **attrs: Any) -> Iterator[Span]:
        parent = _CURRENT.get()
        record = Span(
            name=name,
            kind=kind,
            started_ms=self._clock.now_ms(),
            parent_id=parent,
            span_id=self._ids.new_id(),
            attrs=dict(attrs),
        )
        token = _CURRENT.set(record.span_id)
        try:
            yield record
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.ended_ms = self._clock.now_ms()
            self.spans.append(record)
            _CURRENT.reset(token)

    def tokens(self, span: Span, prompt: int, completion: int) -> None:
        span.tokens_in += prompt
        span.tokens_out += completion

    def cost_usd(self) -> float:
        tin = sum(s.tokens_in for s in self.spans)
        tout = sum(s.tokens_out for s in self.spans)
        return (tin / 1000) * self.usd_per_1k_in + (tout / 1000) * self.usd_per_1k_out

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model": self.model,
            "spans": len(self.spans),
            "errors": sum(1 for s in self.spans if s.error),
            "tokens_in": sum(s.tokens_in for s in self.spans),
            "tokens_out": sum(s.tokens_out for s in self.spans),
            "cost_usd": round(self.cost_usd(), 6),
            "latency_ms": round(sum(s.latency_ms for s in self.spans if s.parent_id is None), 2),
        }

    def export(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model": self.model,
            "summary": self.summary(),
            "spans": [asdict(s) for s in self.spans],
        }
