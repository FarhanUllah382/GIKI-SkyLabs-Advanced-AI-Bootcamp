# Support Ticket Router (LangGraph Subgraph Demo)

A small LangGraph project that demonstrates composing **independent subgraphs**
into a single parent graph. A support query flows through three stages —
intake, resolution, and review — each implemented as its own self-contained
graph with its own state schema, then wired together by a parent orchestrator.

## How it works

The parent graph (`ParentState`) does **not** share a state schema with any
of the subgraphs. Each subgraph is fully independent and has no idea a
parent even exists. The parent's node functions are responsible for:

1. Translating relevant fields from `ParentState` into the subgraph's own
   input state.
2. Calling `subgraph.invoke(...)`.
3. Translating the subgraph's output back into a `ParentState` update.

```
START -> intake -> resolution -> review -> END
```

| Stage | File | Subgraph state | What it does |
|---|---|---|---|
| Intake | `intake_graph.py` | `IntakeState` | Strips whitespace from the raw query, then classifies intent as `billing`, `technical`, or `general` via keyword matching. |
| Resolution | `resolution_graph.py` | `ResolutionState` | Routes to a `billing`, `technical`, or `general` handler based on the detected intent and produces a draft response. |
| Review | `review_graph.py` | `ReviewState` | Validates the draft (must be longer than 10 characters). If invalid, loops through a `revise` node up to `MAX_RETRIES` (2) times, then force-approves. |

## Project structure

```
.
├── intake_graph.py       # IntakeState + normalize/classify nodes + intake_subgraph
├── resolution_graph.py   # ResolutionState + billing/technical/general handlers + resolution_subgraph
├── review_graph.py       # ReviewState + validate/revise loop + review_subgraph
├── parent_graph.py       # ParentState + wrapper nodes that invoke each subgraph + app
├── main.py                # Runs sample queries through `app`, optional graph PNG export
└── README.md
```

## Requirements

- Python 3.9+
- [`langgraph`](https://pypi.org/project/langgraph/)

```bash
pip install langgraph
```

To render the graph diagram (`render_graph()` in `main.py`) you'll also need
the optional Mermaid rendering dependencies pulled in by
`draw_mermaid_png()`.

## Usage

Run the sample queries through the full pipeline:

```bash
python main.py
```

This prints step-by-step logs from each subgraph and the final `ParentState`
for three example queries (a billing question, a technical issue, and a
generic greeting).

To use the compiled graph in your own code:

```python
from parent_graph import app

result = app.invoke({
    "query": "Why was I charged twice on my last invoice?",
    "intent": None,
    "draft_response": None,
    "approved": False,
    "retry_count": 0,
})

print(result)
```

## Notes

- Each subgraph can be developed, tested, and reused independently of the
  parent graph — swap out `resolution_graph.py`'s handlers, for example,
  without touching intake or review.
- State translation between parent and child happens explicitly inside the
  parent's node functions (`intake_node`, `resolution_node`, `review_node`
  in `parent_graph.py`), which keeps the subgraphs decoupled from the
  parent's schema.
