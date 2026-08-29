from agent_trace import Tracer


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


def test_nested_spans_and_cost() -> None:
    t = Tracer("run_1", model="test", usd_per_1k_in=1.0, usd_per_1k_out=2.0, clock=FakeClock(), ids=SeqIds())
    with t.span("answer", "llm") as sp:
        t.tokens(sp, 1000, 500)
        with t.span("search", "tool", tool="web"):
            pass
    summary = t.summary()
    assert summary["spans"] == 2
    assert summary["tokens_in"] == 1000
    assert summary["cost_usd"] == 2.0
    names = [s.name for s in t.spans]
    assert names == ["search", "answer"]
    child = next(s for s in t.spans if s.name == "search")
    parent = next(s for s in t.spans if s.name == "answer")
    assert child.parent_id == parent.span_id


def test_error_is_recorded_and_reraised() -> None:
    t = Tracer("run_err", clock=FakeClock(), ids=SeqIds())
    try:
        with t.span("boom", "tool"):
            raise ValueError("hung")
    except ValueError:
        pass
    assert t.spans[0].error == "ValueError: hung"
    assert t.summary()["errors"] == 1
