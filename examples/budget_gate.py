"""Refuse the next tool/LLM call when a hard budget would be exceeded."""

from agent_trace import Budget, BudgetExceeded, Tracer


def main() -> None:
    t = Tracer(
        "run_budget",
        model="grok",
        usd_per_1k_in=0.003,
        usd_per_1k_out=0.015,
        budget=Budget(max_tokens=30, max_steps=2, allowed_tools=frozenset({"web"})),
    )
    with t.span("answer", "llm") as span:
        t.reserve_tokens(12, 8)
        t.tokens(span, 12, 8)
        with t.span("search", "tool", tool="web"):
            pass
    print("ok", t.summary()["budget"])

    try:
        t.reserve_tokens(20, 0)
    except BudgetExceeded as exc:
        print("blocked", exc.reason, "limit", exc.limit)


if __name__ == "__main__":
    main()
