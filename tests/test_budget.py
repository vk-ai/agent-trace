from __future__ import annotations

import pytest

from agent_trace import Budget, BudgetExceeded, BudgetGate, Tracer


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def now_ms(self) -> float:
        self.t += 10
        return self.t


class SeqIds:
    def __init__(self) -> None:
        self.n = 0

    def new_id(self) -> str:
        self.n += 1
        return f"s{self.n}"


def test_reserve_tokens_exact_boundary() -> None:
    gate = BudgetGate(Budget(max_tokens=100), usd_per_1k_in=1.0, usd_per_1k_out=1.0)
    gate.reserve_tokens(60, 40)
    assert gate.tokens == 100
    with pytest.raises(BudgetExceeded) as exc:
        gate.reserve_tokens(1, 0)
    assert exc.value.reason == "tokens"
    assert gate.tokens == 100  # unchanged after reject


def test_reserve_tokens_rejects_without_mutating_cost() -> None:
    gate = BudgetGate(Budget(max_cost_usd=0.001), usd_per_1k_in=1.0, usd_per_1k_out=1.0)
    gate.reserve_tokens(1, 0)  # $0.001 exact
    assert gate.cost_usd() == pytest.approx(0.001)
    with pytest.raises(BudgetExceeded) as exc:
        gate.reserve_tokens(1, 0)
    assert exc.value.reason == "cost"
    assert gate.tokens == 1


def test_steps_and_tool_allowlist() -> None:
    gate = BudgetGate(Budget(max_steps=1, allowed_tools=frozenset({"web"})))
    gate.authorize_tool("web")
    gate.authorize_step()
    with pytest.raises(BudgetExceeded) as exc:
        gate.authorize_step()
    assert exc.value.reason == "steps"
    assert gate.steps == 1
    with pytest.raises(BudgetExceeded) as exc2:
        gate.authorize_tool("shell")
    assert exc2.value.reason == "tool"
    assert exc2.value.requested == "shell"


def test_unlimited_fields_and_negative_inputs() -> None:
    gate = BudgetGate(Budget())  # all unlimited
    gate.authorize_step()
    gate.authorize_tool("anything")
    gate.reserve_tokens(10, 5)
    assert gate.tokens == 15
    with pytest.raises(ValueError):
        gate.reserve_tokens(-1, 0)
    with pytest.raises(ValueError):
        Budget(max_tokens=-1)


def test_tracer_gates_before_side_effect() -> None:
    budget = Budget(max_steps=2, max_tokens=50, allowed_tools=frozenset({"web"}))
    t = Tracer(
        "run_b",
        model="test",
        usd_per_1k_in=1.0,
        usd_per_1k_out=1.0,
        clock=FakeClock(),
        ids=SeqIds(),
        budget=budget,
    )
    with t.span("answer", "llm") as sp:
        t.reserve_tokens(10, 5)
        t.tokens(sp, 10, 5)
        with t.span("search", "tool", tool="web"):
            pass
    assert t.gate is not None
    assert t.gate.steps == 2
    assert t.summary()["budget"]["tokens"] == 15

    with pytest.raises(BudgetExceeded):
        with t.span("again", "llm"):
            pass


def test_denied_tool_does_not_consume_step() -> None:
    t = Tracer(
        "run_tool",
        clock=FakeClock(),
        ids=SeqIds(),
        budget=Budget(max_steps=1, allowed_tools=frozenset({"web"})),
    )
    with pytest.raises(BudgetExceeded) as exc:
        with t.span("bad", "tool", tool="shell"):
            pass
    assert exc.value.reason == "tool"
    assert t.gate is not None
    assert t.gate.steps == 0
    assert t.spans == []


def test_existing_tracer_api_without_budget() -> None:
    t = Tracer("run_1", model="test", usd_per_1k_in=1.0, usd_per_1k_out=2.0, clock=FakeClock(), ids=SeqIds())
    with t.span("answer", "llm") as sp:
        t.tokens(sp, 1000, 500)
        with t.span("search", "tool", tool="web"):
            pass
    summary = t.summary()
    assert summary["spans"] == 2
    assert "budget" not in summary
