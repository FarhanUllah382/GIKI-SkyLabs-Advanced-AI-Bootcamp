"""Final review step: checks the specialist's draft and produces the
customer-facing response, deciding whether a human needs to step in."""

from __future__ import annotations

from pydantic import BaseModel, Field

from support_triage.llm import get_model
from support_triage.state import TicketState

REVIEWER_SYSTEM_PROMPT = """\
You are the quality-review step in a customer support pipeline. You \
receive a specialist's draft response plus their internal notes, and you \
decide what actually gets sent to the customer.

Rules:
- Tighten the draft for tone and clarity, but don't invent facts, \
promises, dollar amounts, or dates that aren't already in the draft or \
ticket.
- Set escalate_to_human=true if: the ticket is high urgency AND requires \
an action the specialist can't actually perform (refunds, account \
unlocks, contract/pricing exceptions), OR the specialist's notes express \
real uncertainty, OR the issue implies a broader outage affecting more \
than one customer.
- Otherwise escalate_to_human=false — the draft is good enough to send \
as-is (after your polish).
- review_notes is a one-sentence internal explanation of your decision, \
not shown to the customer.\
"""


class Review(BaseModel):
    final_response: str = Field(description="The polished, customer-facing response")
    escalate_to_human: bool = Field(
        description="Whether a human must act before/instead of sending this"
    )
    review_notes: str = Field(description="One-sentence internal rationale")


def review_and_finalize(state: TicketState) -> dict:
    model = get_model().with_structured_output(Review)

    user_prompt = (
        f"Ticket {state['ticket_id']} — category: {state.get('category')}, "
        f"urgency: {state.get('urgency')}\n\n"
        f"Original message:\n{state['message']}\n\n"
        f"Specialist ({state.get('specialist')}) draft response:\n"
        f"{state.get('specialist_response')}\n\n"
        f"Specialist internal notes:\n{state.get('specialist_notes')}"
    )

    result: Review = model.invoke(
        [
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    return {
        "final_response": result.final_response,
        "escalate_to_human": result.escalate_to_human,
        "review_notes": result.review_notes,
    }
