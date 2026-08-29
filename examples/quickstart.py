"""Drop into a new project or an existing agent loop."""

from agent_trace import Tracer


def call_model(prompt: str) -> dict:
    return {"text": "ok", "usage": {"prompt": 12, "completion": 8}}


def main() -> None:
    t = Tracer("run_01", model="grok", usd_per_1k_in=0.003, usd_per_1k_out=0.015)
    with t.span("answer", "llm") as span:
        res = call_model("hello")
        t.tokens(span, res["usage"]["prompt"], res["usage"]["completion"])
    print(t.summary())


if __name__ == "__main__":
    main()
