"""In-path budget and policy gates for agent runs.

Observability alone records overruns after the fact. A BudgetGate refuses the
next LLM/tool step *before* the side effect when tokens, estimated USD, step
count, or tool policy would be exceeded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

BudgetReason = Literal["tokens", "cost", "steps", "tool"]


@dataclass(frozen=True)
class Budget:
    """Hard limits for one agent run. ``None`` means unlimited for that field.

    ``allowed_tools``:
      - ``None`` — any tool name is allowed
      - empty frozenset — no tools allowed
      - non-empty — only listed tool names may be authorized
    """

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_steps: int | None = None
    allowed_tools: frozenset[str] | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_tokens", self.max_tokens),
            ("max_steps", self.max_steps),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0 or None")
        if self.max_cost_usd is not None and self.max_cost_usd < 0:
            raise ValueError("max_cost_usd must be >= 0 or None")


class BudgetExceeded(RuntimeError):
    """Raised when a reservation or authorization would exceed the Budget."""

    def __init__(
        self,
        reason: BudgetReason,
        *,
        limit: Any,
        observed: Any,
        requested: Any,
        message: str | None = None,
    ) -> None:
        self.reason = reason
        self.limit = limit
        self.observed = observed
        self.requested = requested
        text = message or (
            f"budget exceeded ({reason}): observed={observed!r} "
            f"requested={requested!r} limit={limit!r}"
        )
        super().__init__(text)


class BudgetGate:
    """Pre-action gate: reserve tokens/cost and authorize steps/tools.

    Rejected calls leave internal counters unchanged. Cost uses the same
    USD-per-1k rates as Tracer.
    """

    def __init__(
        self,
        budget: Budget,
        *,
        usd_per_1k_in: float = 0.0,
        usd_per_1k_out: float = 0.0,
    ) -> None:
        self.budget = budget
        self.usd_per_1k_in = usd_per_1k_in
        self.usd_per_1k_out = usd_per_1k_out
        self._tokens_in = 0
        self._tokens_out = 0
        self._steps = 0

    @property
    def tokens_in(self) -> int:
        return self._tokens_in

    @property
    def tokens_out(self) -> int:
        return self._tokens_out

    @property
    def tokens(self) -> int:
        return self._tokens_in + self._tokens_out

    @property
    def steps(self) -> int:
        return self._steps

    def cost_usd(self) -> float:
        return (self._tokens_in / 1000) * self.usd_per_1k_in + (
            self._tokens_out / 1000
        ) * self.usd_per_1k_out

    def _estimate_cost(self, prompt: int, completion: int) -> float:
        return (prompt / 1000) * self.usd_per_1k_in + (
            completion / 1000
        ) * self.usd_per_1k_out

    def authorize_step(self) -> None:
        """Authorize one more step/span. Call before starting side-effect work."""
        limit = self.budget.max_steps
        if limit is None:
            self._steps += 1
            return
        if self._steps + 1 > limit:
            raise BudgetExceeded(
                "steps",
                limit=limit,
                observed=self._steps,
                requested=1,
                message=f"step budget exceeded: {self._steps + 1} > {limit}",
            )
        self._steps += 1

    def authorize_tool(self, tool: str) -> None:
        """Authorize a named tool before invoking it."""
        if not tool:
            raise ValueError("tool name must be non-empty")
        allowed = self.budget.allowed_tools
        if allowed is None:
            return
        if tool not in allowed:
            raise BudgetExceeded(
                "tool",
                limit=sorted(allowed),
                observed=None,
                requested=tool,
                message=f"tool not allowed: {tool!r}",
            )

    def reserve_tokens(self, prompt: int, completion: int = 0) -> None:
        """Reserve estimated tokens (and implied USD) before an LLM call."""
        if prompt < 0 or completion < 0:
            raise ValueError("token reservation amounts must be >= 0")

        next_in = self._tokens_in + prompt
        next_out = self._tokens_out + completion
        next_total = next_in + next_out

        token_limit = self.budget.max_tokens
        if token_limit is not None and next_total > token_limit:
            raise BudgetExceeded(
                "tokens",
                limit=token_limit,
                observed=self.tokens,
                requested=prompt + completion,
                message=(
                    f"token budget exceeded: {next_total} > {token_limit} "
                    f"(observed={self.tokens}, requested={prompt + completion})"
                ),
            )

        cost_limit = self.budget.max_cost_usd
        if cost_limit is not None:
            next_cost = self.cost_usd() + self._estimate_cost(prompt, completion)
            if next_cost > cost_limit + 1e-12:
                raise BudgetExceeded(
                    "cost",
                    limit=cost_limit,
                    observed=round(self.cost_usd(), 6),
                    requested=round(self._estimate_cost(prompt, completion), 6),
                    message=(
                        f"cost budget exceeded: {next_cost:.6f} > {cost_limit} "
                        f"(observed={self.cost_usd():.6f})"
                    ),
                )

        self._tokens_in = next_in
        self._tokens_out = next_out

    def snapshot(self) -> dict[str, Any]:
        """Stable JSON-friendly view of limits and consumption."""
        b = self.budget
        return {
            "allowed_tools": (
                None if b.allowed_tools is None else sorted(b.allowed_tools)
            ),
            "cost_usd": round(self.cost_usd(), 6),
            "max_cost_usd": b.max_cost_usd,
            "max_steps": b.max_steps,
            "max_tokens": b.max_tokens,
            "steps": self._steps,
            "tokens": self.tokens,
            "tokens_in": self._tokens_in,
            "tokens_out": self._tokens_out,
        }
