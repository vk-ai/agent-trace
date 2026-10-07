"""In-path budget and policy gates for agent runs.

Observability alone records overruns after the fact. A BudgetGate refuses the
next LLM/tool step *before* the side effect when tokens, estimated USD, step
count, or tool policy would be exceeded, or when the run looks stuck:
the same tool call repeated, the same failure repeated, or no state change
over N steps.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

BudgetReason = Literal["tokens", "cost", "steps", "tool", "loop", "no_progress"]

_DIGITS = re.compile(r"\d+")
_HEX = re.compile(r"\b0x[0-9a-fA-F]+\b|\b[0-9a-fA-F]{8,}\b")
_SPACE = re.compile(r"\s+")


def _canonical(value: Any, volatile: frozenset[str]) -> Any:
    """JSON-friendly canonical form; drops ``volatile`` keys at any depth."""
    if isinstance(value, Mapping):
        return {
            str(k): _canonical(v, volatile)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            if str(k) not in volatile
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(v, volatile) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(v, volatile) for v in value), key=repr)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def fingerprint(value: Any, volatile_keys: frozenset[str] = frozenset()) -> str:
    """Stable short hash of ``value`` with ``volatile_keys`` removed."""
    blob = json.dumps(
        _canonical(value, volatile_keys), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def normalize_error(error: BaseException | str) -> str:
    """Collapse numbers, hex ids and whitespace so retries of one failure match."""
    text = (
        f"{type(error).__name__}: {error}"
        if isinstance(error, BaseException)
        else str(error)
    )
    text = _HEX.sub("<hex>", text)
    text = _DIGITS.sub("<n>", text)
    return _SPACE.sub(" ", text).strip()


@dataclass(frozen=True)
class Budget:
    """Hard limits for one agent run. ``None`` means unlimited for that field.

    ``allowed_tools``:
      - ``None`` — any tool name is allowed
      - empty frozenset — no tools allowed
      - non-empty — only listed tool names may be authorized

    Loop / no-progress guard (all off by default):
      - ``max_repeat`` — most consecutive identical tool calls allowed
        (same tool + same args after dropping ``volatile_keys``). The next
        identical call is refused with reason ``"loop"``. Non-tool steps in
        between (e.g. the LLM turn) do not reset the streak; a different
        tool call does.
      - ``max_repeat_errors`` — most times the same call may fail with the
        same normalized error. Once reached, that call is refused with
        reason ``"loop"``.
      - ``max_no_progress`` — most steps allowed since the observed state
        last changed (see ``BudgetGate.observe_state``). The next step is
        refused with reason ``"no_progress"``. Inactive until a state has
        been observed.
      - ``volatile_keys`` — arg keys ignored when fingerprinting
        (request ids, timestamps, nonces).
    """

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_steps: int | None = None
    allowed_tools: frozenset[str] | None = None
    max_repeat: int | None = None
    max_repeat_errors: int | None = None
    max_no_progress: int | None = None
    volatile_keys: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        for name, value in (
            ("max_tokens", self.max_tokens),
            ("max_steps", self.max_steps),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0 or None")
        for name, value in (
            ("max_repeat", self.max_repeat),
            ("max_repeat_errors", self.max_repeat_errors),
            ("max_no_progress", self.max_no_progress),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 or None")
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
        # Loop / no-progress guard state.
        self._last_call: str | None = None
        self._repeat_streak = 0
        self._error_counts: dict[str, int] = {}
        self._state_fp: str | None = None
        self._steps_since_progress = 0

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
        self.check_progress()
        limit = self.budget.max_steps
        if limit is not None and self._steps + 1 > limit:
            raise BudgetExceeded(
                "steps",
                limit=limit,
                observed=self._steps,
                requested=1,
                message=f"step budget exceeded: {self._steps + 1} > {limit}",
            )
        self._steps += 1
        if self._state_fp is not None:
            self._steps_since_progress += 1

    # -- loop / no-progress guard -------------------------------------------

    def call_fingerprint(self, tool: str, args: Any = None) -> str:
        """Fingerprint of (tool, args) with ``Budget.volatile_keys`` removed."""
        return fingerprint({"tool": tool, "args": args}, self.budget.volatile_keys)

    def _error_key(self, call_fp: str, error: BaseException | str) -> str:
        return f"{call_fp}:{fingerprint(normalize_error(error))}"

    def check_call(self, tool: str, args: Any = None) -> None:
        """Refuse a tool call that repeats a loop. Does not mutate the gate."""
        b = self.budget
        fp = self.call_fingerprint(tool, args)
        if b.max_repeat is not None and fp == self._last_call:
            if self._repeat_streak + 1 > b.max_repeat:
                raise BudgetExceeded(
                    "loop",
                    limit=b.max_repeat,
                    observed=self._repeat_streak,
                    requested={"tool": tool, "args": args},
                    message=(
                        f"loop detected: {tool!r} called with identical args "
                        f"{self._repeat_streak + 1} times in a row "
                        f"(max_repeat={b.max_repeat})"
                    ),
                )
        if b.max_repeat_errors is not None and self._error_counts:
            prefix = fp + ":"
            worst = max(
                (n for k, n in self._error_counts.items() if k.startswith(prefix)),
                default=0,
            )
            if worst >= b.max_repeat_errors:
                raise BudgetExceeded(
                    "loop",
                    limit=b.max_repeat_errors,
                    observed=worst,
                    requested={"tool": tool, "args": args},
                    message=(
                        f"loop detected: {tool!r} with these args already failed "
                        f"{worst} times with the same error "
                        f"(max_repeat_errors={b.max_repeat_errors})"
                    ),
                )

    def record_call(self, tool: str, args: Any = None) -> None:
        """Record a dispatched tool call for repeat detection."""
        fp = self.call_fingerprint(tool, args)
        if fp == self._last_call:
            self._repeat_streak += 1
        else:
            self._last_call = fp
            self._repeat_streak = 1

    def authorize_call(self, tool: str, args: Any = None) -> None:
        """Tool allowlist + loop checks, then record the call (atomic)."""
        self.authorize_tool(tool)
        self.check_call(tool, args)
        self.record_call(tool, args)

    def record_error(self, tool: str, args: Any, error: BaseException | str) -> None:
        """Record a failed tool call; identical failures feed ``max_repeat_errors``."""
        key = self._error_key(self.call_fingerprint(tool, args), error)
        self._error_counts[key] = self._error_counts.get(key, 0) + 1

    def observe_state(self, state: Any) -> bool:
        """Report the agent's current state. Returns True if it changed.

        Pass anything that captures progress (a plan, files touched, the
        scratchpad, a set of facts found). ``volatile_keys`` are ignored.
        """
        fp = fingerprint(state, self.budget.volatile_keys)
        if fp != self._state_fp:
            self._state_fp = fp
            self._steps_since_progress = 0
            return True
        return False

    def check_progress(self) -> None:
        """Refuse the next step when state has not changed for too long."""
        limit = self.budget.max_no_progress
        if limit is None or self._state_fp is None:
            return
        if self._steps_since_progress + 1 > limit:
            raise BudgetExceeded(
                "no_progress",
                limit=limit,
                observed=self._steps_since_progress,
                requested=1,
                message=(
                    f"no progress: state unchanged for {self._steps_since_progress} "
                    f"steps (max_no_progress={limit})"
                ),
            )

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
        payload: dict[str, Any] = {
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
        if (
            b.max_repeat is not None
            or b.max_repeat_errors is not None
            or b.max_no_progress is not None
        ):
            payload["loop"] = {
                "max_no_progress": b.max_no_progress,
                "max_repeat": b.max_repeat,
                "max_repeat_errors": b.max_repeat_errors,
                "repeat_streak": self._repeat_streak,
                "repeated_errors": max(self._error_counts.values(), default=0),
                "steps_since_progress": self._steps_since_progress,
            }
        return payload
