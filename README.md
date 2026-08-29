# agent-trace

Zero-dependency tracer for LLM agents. Nested spans, tool names, tokens, estimated USD.

**Live:** [github.com/vk-ai/agent-trace](https://github.com/vk-ai/agent-trace)

![What a run looks like](docs/looks.svg)

## Why this exists

LangSmith, Phoenix, and OpenTelemetry SDKs are products. This is a **120-line contract** you can read in one sitting. Inject a clock in tests. Export JSON. No vendor, no daemon.

| | agent-trace | Typical stacks |
|---|---|---|
| Dependencies | 0 | OTel + exporter + vendor SDK |
| Testable | `Clock` + `IdFactory` protocols | Real time, flaky |
| Shape | One `Tracer` per run | Global processors |
| Cost | Tokens × your rates | Dashboard, later |

Design: **dependency injection** (clock, ids), **context manager** spans, **ContextVar** for parent/child without passing ids, frozen-friendly `Span` dataclass.

## Install — new project

![Install](docs/install.svg)

```bash
git clone https://github.com/vk-ai/agent-trace.git
cd agent-trace
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python examples/quickstart.py
```

## Install — existing project

```bash
pip install agent-trace
```

```python
from agent_trace import Tracer

t = Tracer("run_01", model="grok", usd_per_1k_in=0.003, usd_per_1k_out=0.015)
with t.span("answer", "llm") as span:
    t.tokens(span, 120, 40)
    with t.span("search", "tool", tool="web"):
        pass
print(t.summary())
```

## What it looks like

`t.summary()` prints:

```
{'run_id': 'run_01', 'model': 'grok', 'spans': 2, 'errors': 0,
 'tokens_in': 120, 'tokens_out': 40, 'cost_usd': 0.00096, 'latency_ms': 40.0}
```

Errors are recorded on the span and re-raised. Tests never sleep — they inject a fake clock.

MIT. Python 3.10+. CI on 3.10 and 3.12.
