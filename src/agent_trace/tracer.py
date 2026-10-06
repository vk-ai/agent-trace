from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from typing import Any, Iterator

from .budget import Budget, BudgetGate
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
        budget: Budget | None = None,
    ) -> None:
        self.run_id = run_id
        self.model = model
        self.usd_per_1k_in = usd_per_1k_in
        self.usd_per_1k_out = usd_per_1k_out
        self._clock = clock or _WallClock()
        self._ids = ids or _UuidFactory()
        self.spans: list[Span] = []
        self.gate: BudgetGate | None = (
            BudgetGate(
                budget,
                usd_per_1k_in=usd_per_1k_in,
                usd_per_1k_out=usd_per_1k_out,
            )
            if budget is not None
            else None
        )

    @contextmanager
    def span(self, name: str, kind: Kind = "chain", **attrs: Any) -> Iterator[Span]:
        """Open a span. Tool spans pass ``tool=`` and optionally ``args=``.

        With a budget, tool policy, loop checks and the step budget all run
        before the span starts; a refused span records nothing.
        """
        tool: str | None = None
        args = attrs.get("args")
        if self.gate is not None:
            # Policy checks before consuming a step slot so rejections are atomic.
            if kind == "tool" and attrs.get("tool") is not None:
                tool = str(attrs["tool"])
                self.gate.authorize_tool(tool)
                self.gate.check_call(tool, args)
            self.gate.authorize_step()
            if tool is not None:
                self.gate.record_call(tool, args)
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
            if self.gate is not None and tool is not None:
                self.gate.record_error(tool, args, exc)
            raise
        finally:
            record.ended_ms = self._clock.now_ms()
            self.spans.append(record)
            _CURRENT.reset(token)

    def reserve_tokens(self, prompt: int, completion: int = 0) -> None:
        """Reserve estimated tokens/cost on the attached gate before an LLM call."""
        if self.gate is None:
            raise RuntimeError("Tracer has no budget gate; pass budget=Budget(...)")
        self.gate.reserve_tokens(prompt, completion)

    def authorize_tool(self, tool: str) -> None:
        """Authorize a tool on the attached gate before invoking it."""
        if self.gate is None:
            raise RuntimeError("Tracer has no budget gate; pass budget=Budget(...)")
        self.gate.authorize_tool(tool)

    def observe_state(self, state: Any) -> bool:
        """Report agent state for the no-progress guard. True if it changed."""
        if self.gate is None:
            raise RuntimeError("Tracer has no budget gate; pass budget=Budget(...)")
        return self.gate.observe_state(state)

    def tokens(self, span: Span, prompt: int, completion: int) -> None:
        span.tokens_in += prompt
        span.tokens_out += completion

    def cost_usd(self) -> float:
        tin = sum(s.tokens_in for s in self.spans)
        tout = sum(s.tokens_out for s in self.spans)
        return (tin / 1000) * self.usd_per_1k_in + (tout / 1000) * self.usd_per_1k_out

    def summary(self) -> dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "model": self.model,
            "spans": len(self.spans),
            "errors": sum(1 for s in self.spans if s.error),
            "tokens_in": sum(s.tokens_in for s in self.spans),
            "tokens_out": sum(s.tokens_out for s in self.spans),
            "cost_usd": round(self.cost_usd(), 6),
            "latency_ms": round(sum(s.latency_ms for s in self.spans if s.parent_id is None), 2),
        }
        if self.gate is not None:
            payload["budget"] = self.gate.snapshot()
        return payload

    def export(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model": self.model,
            "summary": self.summary(),
            "spans": [asdict(s) for s in self.spans],
        }
