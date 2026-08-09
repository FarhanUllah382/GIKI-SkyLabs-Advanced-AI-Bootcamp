# Customer Support Triage — Multi-Agent System (LangGraph)

A multi-agent customer support triage pipeline built with [LangGraph](https://langchain-ai.github.io/langgraph/) and Claude. It classifies an incoming ticket, routes it to a specialized agent (Billing, Technical, Sales, or General), and has a reviewer agent produce the final response — deciding whether the ticket needs to escalate to a human.

Everything runs on **mock tickets and mock data** (`support_triage/mock_data.py`) — no Zendesk, Stripe, or other real integrations. Swapping in real data sources later just means changing what feeds `TicketState` and what the specialist nodes read from.

## Architecture

```
                     ┌──────────────┐
                     │  classify    │  → category (billing/technical/sales/general)
                     └──────┬───────┘    + urgency (low/medium/high)
                            │
              ┌─────────────┼─────────────┬─────────────┐
              ▼             ▼             ▼             ▼
        billing_agent  technical_agent sales_agent  general_agent
              │             │             │             │
              └─────────────┴─────────────┴─────────────┘
                            │
                            ▼
                       ┌─────────┐
                       │ review  │  → final_response + escalate_to_human
                       └─────────┘
```

- **`classify`** — Claude reads the ticket and returns a structured `{category, urgency, reasoning}`.
- **Specialist agents** (`billing_agent` / `technical_agent` / `sales_agent` / `general_agent`) — each pulls relevant mock context (customer billing record, matched knowledge-base articles, plan sheet) and drafts a resolution plus internal notes. They share one implementation (`nodes/specialists.py: make_specialist_node`) parameterized by role/prompt/context.
- **`review`** — reads the specialist's draft + notes, polishes the customer-facing response, and flags `escalate_to_human` when the fix requires an action the agent can't actually perform (refund, account unlock, etc.) or the specialist was uncertain.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then put your real key in .env
```

## Run

```bash
python -m support_triage.main            # run all mock tickets
python -m support_triage.main --ticket TCK-1002   # run one
```

Each run prints the classification, which agent handled it, the final customer-facing response, and whether it was flagged for human escalation.

## Files

| File | Purpose |
|---|---|
| `support_triage/state.py` | Shared `TicketState` passed between graph nodes |
| `support_triage/mock_data.py` | Mock tickets, customer records, KB articles, pricing plans |
| `support_triage/llm.py` | Single place the Claude model (`claude-opus-5`) is constructed |
| `support_triage/nodes/classifier.py` | Classification node + routing function |
| `support_triage/nodes/specialists.py` | Billing/Technical/Sales/General agent nodes |
| `support_triage/nodes/reviewer.py` | Final review + escalation decision |
| `support_triage/graph.py` | Wires the nodes into a `StateGraph` |
| `support_triage/main.py` | CLI runner over the mock ticket set |

## Extending toward real integrations

- Replace `mock_data.CUSTOMERS` with a Stripe customer lookup.
- Replace `mock_data.KNOWLEDGE_BASE` keyword search with a real vector-store retriever.
- Swap `MOCK_TICKETS` for a Zendesk webhook/poller that constructs `TicketState` from real tickets.
- The graph shape (classify → route → specialist → review) doesn't need to change for any of this — only the data sources feeding each node.
