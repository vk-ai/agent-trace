from __future__ import annotations

import pytest

from agent_trace import Budget, BudgetExceeded, BudgetGate, Tracer, fingerprint, normalize_error


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


def _tracer(**budget_kw: object) -> Tracer:
    return Tracer("run_loop", clock=FakeClock(), ids=SeqIds(), budget=Budget(**budget_kw))  # type: ignore[arg-type]


def test_fingerprint_ignores_key_order_and_volatile_keys() -> None:
    a = {"q": "x", "page": 1, "request_id": "r1", "meta": {"ts": 1, "k": [1, 2]}}
    b = {"meta": {"k": [1, 2], "ts": 99}, "request_id": "r2", "page": 1, "q": "x"}
    vol = frozenset({"request_id", "ts"})
    assert fingerprint(a, vol) == fingerprint(b, vol)
    assert fingerprint(a) != fingerprint(b)
    assert fingerprint({"q": "x"}) != fingerprint({"q": "y"})


def test_normalize_error_collapses_ids_and_numbers() -> None:
    e1 = TimeoutError("timed out after 30s (req 0xdeadbeef, id 7f3a9c21d4)")
    e2 = TimeoutError("timed out after 31s  (req 0xcafe, id 99aa00bb11)")
    assert normalize_error(e1) == normalize_error(e2)
    assert normalize_error(e1) != normalize_error(ValueError("timed out after 30s"))


def test_max_repeat_blocks_next_identical_call_before_dispatch() -> None:
    gate = BudgetGate(Budget(max_repeat=3))
    for _ in range(3):
        gate.authorize_call("search", {"q": "weather"})
    with pytest.raises(BudgetExceeded) as exc:
        gate.authorize_call("search", {"q": "weather"})
    assert exc.value.reason == "loop"
    assert exc.value.limit == 3
    assert exc.value.observed == 3
    assert exc.value.requested == {"tool": "search", "args": {"q": "weather"}}
    assert "4 times in a row" in str(exc.value)
    # Refusal is atomic: the streak did not grow.
    assert gate.snapshot()["loop"]["repeat_streak"] == 3
    # A different call is fine and resets the streak.
    gate.authorize_call("search", {"q": "weather in Paris"})
    assert gate.snapshot()["loop"]["repeat_streak"] == 1
    gate.authorize_call("search", {"q": "weather"})


def test_volatile_keys_do_not_hide_a_loop() -> None:
    gate = BudgetGate(Budget(max_repeat=2, volatile_keys=frozenset({"request_id"})))
    gate.authorize_call("fetch", {"url": "u", "request_id": 1})
    gate.authorize_call("fetch", {"url": "u", "request_id": 2})
    with pytest.raises(BudgetExceeded) as exc:
        gate.authorize_call("fetch", {"url": "u", "request_id": 3})
    assert exc.value.reason == "loop"


def test_max_repeat_errors_counts_identical_failures_even_when_interleaved() -> None:
    gate = BudgetGate(Budget(max_repeat_errors=2))
    args = {"path": "/tmp/x"}
    gate.authorize_call("read", args)
    gate.record_error("read", args, FileNotFoundError("no such file: /tmp/x (errno 2)"))
    gate.authorize_call("list", {"dir": "/tmp"})  # interleaved different call
    gate.authorize_call("read", args)
    gate.record_error("read", args, FileNotFoundError("no such file: /tmp/x (errno 2)"))
    with pytest.raises(BudgetExceeded) as exc:
        gate.authorize_call("read", args)
    assert exc.value.reason == "loop"
    assert "failed 2 times" in str(exc.value)
    # Other args for the same tool are still allowed.
    gate.authorize_call("read", {"path": "/tmp/y"})


def test_different_errors_do_not_trip_repeat_errors() -> None:
    gate = BudgetGate(Budget(max_repeat_errors=2))
    gate.record_error("read", None, ValueError("bad"))
    gate.record_error("read", None, KeyError("other"))
    gate.authorize_call("read", None)  # each distinct failure seen once


def test_no_progress_blocks_after_n_steps_without_state_change() -> None:
    gate = BudgetGate(Budget(max_no_progress=2))
    # Inactive until a state is observed.
    for _ in range(5):
        gate.authorize_step()
    gate.observe_state({"facts": ["a"]})
    gate.authorize_step()
    assert gate.observe_state({"facts": ["a"]}) is False
    gate.authorize_step()
    with pytest.raises(BudgetExceeded) as exc:
        gate.authorize_step()
    assert exc.value.reason == "no_progress"
    steps_before = gate.steps
    # Progress resets the window; the refused step did not consume budget.
    assert gate.observe_state({"facts": ["a", "b"]}) is True
    gate.authorize_step()
    assert gate.steps == steps_before + 1


def test_tracer_tool_span_loop_is_refused_without_recording_span() -> None:
    t = _tracer(max_repeat=2, max_steps=10)
    for _ in range(2):
        with t.span("search", "tool", tool="web", args={"q": "same"}):
            pass
    with pytest.raises(BudgetExceeded) as exc:
        with t.span("search", "tool", tool="web", args={"q": "same"}):
            raise AssertionError("tool body must not run")
    assert exc.value.reason == "loop"
    assert len(t.spans) == 2
    assert t.gate is not None and t.gate.steps == 2
    summary = t.summary()["budget"]
    assert summary["loop"]["max_repeat"] == 2
    assert summary["loop"]["repeat_streak"] == 2


def test_llm_turns_between_identical_tool_calls_do_not_reset_streak() -> None:
    t = _tracer(max_repeat=2)
    for _ in range(2):
        with t.span("think", "llm"):
            pass
        with t.span("search", "tool", tool="web", args={"q": "same"}):
            pass
    with t.span("think", "llm"):
        pass
    with pytest.raises(BudgetExceeded):
        with t.span("search", "tool", tool="web", args={"q": "same"}):
            pass


def test_tracer_records_tool_failures_for_repeat_errors() -> None:
    t = _tracer(max_repeat_errors=2)
    for _ in range(2):
        with pytest.raises(ConnectionError):
            with t.span("fetch", "tool", tool="http", args={"url": "u"}):
                raise ConnectionError("connection refused (attempt 1)")
    with pytest.raises(BudgetExceeded) as exc:
        with t.span("fetch", "tool", tool="http", args={"url": "u"}):
            pass
    assert exc.value.reason == "loop"
    assert t.summary()["budget"]["loop"]["repeated_errors"] == 2


def test_tracer_observe_state_and_no_progress() -> None:
    t = _tracer(max_no_progress=1)
    t.observe_state("plan v1")
    with t.span("step", "chain"):
        pass
    with pytest.raises(BudgetExceeded) as exc:
        with t.span("step", "chain"):
            pass
    assert exc.value.reason == "no_progress"
    t.observe_state("plan v2")
    with t.span("step", "chain"):
        pass


def test_loop_snapshot_absent_when_guard_off_and_validation() -> None:
    assert "loop" not in BudgetGate(Budget(max_steps=1)).snapshot()
    with pytest.raises(ValueError):
        Budget(max_repeat=0)
    with pytest.raises(ValueError):
        Budget(max_no_progress=0)
    with pytest.raises(RuntimeError):
        Tracer("x").observe_state({})
