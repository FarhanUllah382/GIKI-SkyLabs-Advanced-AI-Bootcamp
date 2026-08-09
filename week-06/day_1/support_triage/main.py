"""CLI entry point: runs mock tickets through the triage graph and prints
the result of each stage.

Usage (run from the repo root, one level above support_triage/):
    python -m support_triage.main
    python -m support_triage.main --ticket TCK-1002
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from support_triage.graph import build_graph
from support_triage.mock_data import MOCK_TICKETS

SEPARATOR = "=" * 72


def _print_result(state: dict) -> None:
    print(SEPARATOR)
    print(f"Ticket {state['ticket_id']} — {state['subject']}")
    print(f"Customer: {state['customer_id']}")
    print(SEPARATOR)
    print(f"Category:  {state.get('category')}  |  Urgency: {state.get('urgency')}")
    print(f"Reasoning: {state.get('classification_reasoning')}")
    print(f"\nHandled by: {state.get('specialist')} agent")
    print(f"\n--- Final response to customer ---\n{state.get('final_response')}")
    print(f"\nEscalate to human: {state.get('escalate_to_human')}")
    print(f"Review notes: {state.get('review_notes')}")
    print()


def main() -> int:
    load_dotenv()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
            "your key, or export it in your shell.",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description="Run mock tickets through the triage graph.")
    parser.add_argument(
        "--ticket",
        help="Run a single ticket by ID (e.g. TCK-1002) instead of the full set.",
    )
    args = parser.parse_args()

    tickets = MOCK_TICKETS
    if args.ticket:
        tickets = [t for t in MOCK_TICKETS if t["ticket_id"] == args.ticket]
        if not tickets:
            print(f"No mock ticket with ID {args.ticket!r}.", file=sys.stderr)
            return 1

    graph = build_graph()

    for ticket in tickets:
        result = graph.invoke(dict(ticket))
        _print_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
