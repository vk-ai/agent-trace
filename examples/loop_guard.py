"""Stop an agent that keeps calling the same tool with the same args."""

from agent_trace import Budget, BudgetExceeded, Tracer


def flaky_search(q: str) -> str:
    raise TimeoutError(f"search timed out after 30s (q={q!r})")


def main() -> None:
    t = Tracer(
        "run_loop",
        budget=Budget(
            max_steps=20,
            max_repeat=3,
            max_repeat_errors=2,
            volatile_keys=frozenset({"request_id"}),
        ),
    )
    for attempt in range(10):
        try:
            with t.span("search", "tool", tool="web", args={"q": "weather", "request_id": attempt}):
                flaky_search("weather")
        except TimeoutError as exc:
            print("tool failed:", exc)
        except BudgetExceeded as exc:
            print("blocked:", exc.reason, "-", exc)
            break
    print("budget:", t.summary()["budget"]["loop"])


if __name__ == "__main__":
    main()
