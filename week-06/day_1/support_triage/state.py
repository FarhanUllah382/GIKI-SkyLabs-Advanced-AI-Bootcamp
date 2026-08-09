"""Shared state schema passed between LangGraph nodes."""

from typing import Literal, TypedDict

Category = Literal["billing", "technical", "sales", "general"]
Urgency = Literal["low", "medium", "high"]


class TicketState(TypedDict, total=False):
    # Input
    ticket_id: str
    customer_id: str
    subject: str
    message: str

    # Set by the classifier node
    category: Category
    urgency: Urgency
    classification_reasoning: str

    # Set by whichever specialist node handles the ticket
    specialist: str
    specialist_response: str
    specialist_notes: str

    # Set by the reviewer node
    final_response: str
    escalate_to_human: bool
    review_notes: str
