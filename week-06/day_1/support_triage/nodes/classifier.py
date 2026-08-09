"""Classifies an incoming ticket into a category + urgency before routing."""

from __future__ import annotations

from pydantic import BaseModel, Field

from support_triage.llm import get_model
from support_triage.state import TicketState

CLASSIFIER_SYSTEM_PROMPT = """\
You are a triage classifier for a customer support inbox. Read the ticket \
and decide which team should handle it and how urgently.

Categories:
- billing: payments, invoices, refunds, subscription/plan changes, being \
locked out due to billing status
- technical: bugs, errors, API issues, integrations, outages, data/export \
problems
- sales: new business, upgrades/expansion, pricing questions from \
prospects or customers not currently blocked by a problem
- general: anything that doesn't clearly fit the above

Urgency:
- high: customer is blocked/locked out, losing money, or facing an \
imminent deadline
- medium: real problem but no immediate deadline or blocker
- low: informational question, no urgency

Pick exactly one category and one urgency level, and give a one-sentence \
reason.\
"""


class Classification(BaseModel):
    category: str = Field(description="One of: billing, technical, sales, general")
    urgency: str = Field(description="One of: low, medium, high")
    reasoning: str = Field(description="One sentence explaining the classification")


def classify_ticket(state: TicketState) -> dict:
    model = get_model().with_structured_output(Classification)

    result: Classification = model.invoke(
        [
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Subject: {state['subject']}\n\n"
                    f"Message:\n{state['message']}"
                ),
            },
        ]
    )

    category = result.category.strip().lower()
    if category not in ("billing", "technical", "sales", "general"):
        category = "general"

    urgency = result.urgency.strip().lower()
    if urgency not in ("low", "medium", "high"):
        urgency = "medium"

    return {
        "category": category,
        "urgency": urgency,
        "classification_reasoning": result.reasoning,
    }


def route_by_category(state: TicketState) -> str:
    """Conditional-edge function: picks which specialist node runs next."""
    return state.get("category", "general")
