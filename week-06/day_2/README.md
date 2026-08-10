# 🧠 Research Crew — a minimal multi-agent system

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

Four small agents — a **planner**, a **researcher**, a **writer** and a **critic** — pass work
down a pipeline until the critic signs off. No frameworks, no graph DSL, no vector store.
Roughly 250 lines of plain Python you can read in one sitting.

Built to answer one question: *what is the smallest honest version of a multi-agent system?*

---

## Why this exists

Most multi-agent demos hide the interesting part behind a framework. This repo keeps the
loop visible: each agent is a class with a system prompt and a `run()` method, and the
orchestrator is a `for` loop with a break condition. If you understand this, you understand
what CrewAI and LangGraph are doing underneath.

## How it works

```
  topic
    │
    ▼
┌──────────┐   3-5 sub-questions   ┌────────────┐   dense notes   ┌────────┐
│ Planner  │──────────────────────▶│ Researcher │────────────────▶│ Writer │
└──────────┘                       └────────────┘                 └────┬───┘
                                                                       │ draft
                                                    ┌────────┐         ▼
                                          approved ─│ Critic │◀─────────
                                             │      └────────┘
                                             │           │ up to 2 revisions
                                             ▼           └──────▶ back to Writer
                                        briefing.md
```

| Agent | Job | Fails loudly when |
| --- | --- | --- |
| `Planner` | Turns a vague topic into 3–5 answerable questions | The topic is too broad to decompose |
| `Researcher` | Answers each question as dense bullets, tagging `[uncertain]` | It has to guess |
| `Writer` | Composes a markdown briefing **from the notes only** | It would need to invent a fact |
| `Critic` | Replies `APPROVED`, or at most 3 numbered fixes | The draft drifts from the notes |

The critic is the whole trick. Without it you have a chain; with it you have a loop that
can catch its own mistakes.

## Quickstart

```bash
git clone https://github.com/<your-username>/multi-agent-research-crew.git
cd multi-agent-research-crew
pip install -r requirements.txt
```

Run it with no API key at all — every agent returns a stub, so you can watch the
control flow before spending a token:

```bash
python main.py "how do vector databases work" --mock
```

Then run it for real:

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # Windows: set ANTHROPIC_API_KEY=sk-ant-...
python main.py "how do vector databases work" --out briefing.md
```

```
> 1/4 planning
  [planner] working on: Topic: how do vector databases work
> 2/4 researching
  [researcher] working on: Topic: how do vector databases work
> 3/4 writing (attempt 1)
> 4/4 reviewing
  [critic] working on: Notes: ...
> critic requested changes, revising
> 3/4 writing (attempt 2)
> critic approved the draft
```

### Options

| Flag | Default | What it does |
| --- | --- | --- |
| `--mock` | off | Runs the full pipeline with stubbed agents, no API key needed |
| `--out PATH` | — | Writes the final briefing to a file |
| `--max-revisions N` | `2` | How many times the critic may send the draft back |
| `--quiet` | off | Suppresses the step-by-step log |

Set `MODEL` in your environment to switch models.

## Layout

```
.
├── main.py            # CLI: parses args, builds the client, prints the result
├── orchestrator.py    # Crew: the pipeline and the revision loop
├── agents/
│   ├── base.py        # Agent: system prompt + run() + mock fallback
│   ├── planner.py
│   ├── researcher.py
│   ├── writer.py
│   └── critic.py      # owns the APPROVED sentinel
└── tests/test_crew.py # runs in mock mode — no key, no network
```

## Tests

```bash
pip install pytest && pytest -q
```

They run entirely in mock mode, so CI never needs a secret.

## Extending it

- **Add an agent** — subclass `Agent`, give it a `name`, `role` and `system_prompt`, then
  slot it into `Crew.run()`. That's the whole contract.
- **Give the researcher real sources** — swap its `run()` for a call that passes the
  `web_search` tool, and the rest of the pipeline is unchanged.
- **Run agents in parallel** — the researcher answers questions independently, so a
  `ThreadPoolExecutor` over the planner's list is an easy win.
- **Swap the model** — `agents/base.py` is the only file that touches the SDK.

## Known limits

Deliberately not solved here, so the code stays readable: no persistent memory between
runs, no token budgeting, no retry on API errors, and the critic is the same model as the
writer (so it shares its blind spots). Each is a good first PR.

## License

MIT — see [LICENSE](LICENSE).
