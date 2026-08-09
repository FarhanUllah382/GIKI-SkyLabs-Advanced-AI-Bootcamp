"""Specialized agent nodes: billing, technical, sales, general.

Each specialist is built from the same small factory (`make_specialist_node`)
since they all do the same job — pull relevant mock context, draft a
resolution, and hand off internal notes — and differ only in role, prompt,
and which mock data they consult.
"""

from __future__ import annotations

from collections.abc import Callable

from support_triage.llm import get_model
from support_triage.mock_data import CUSTOMERS, KNOWLEDGE_BASE, PLANS
from support_triage.state import TicketState

SPECIALIST_USER_TEMPLATE = """\
Ticket {ticket_id} (urgency: {urgency})
Subject: {subject}

Customer message:
{message}

Relevant account/context data:
{context}

Draft a resolution for this customer. Then, separately, write one or two \
internal notes for whoever reviews this before it's sent (e.g. anything \
uncertain, anything that needs a human to actually execute like issuing a \
refund).

Respond in this exact format:
RESPONSE:
<the customer-facing draft response>

NOTES:
<internal notes>\
"""


def _format_customer_context(customer_id: str) -> str:
    customer = CUSTOMERS.get(customer_id)
    if not customer:
        return "No account record found for this customer ID."
    return (
        f"Name: {customer['name']}\n"
        f"Plan: {customer['plan']} (${customer['mrr_usd']}/mo)\n"
        f"Account age: {customer['account_age_days']} days\n"
        f"Billing status: {customer['billing_status']}\n"
        f"Open invoices: {customer['open_invoices']}"
    )


def _format_plans() -> str:
    lines = []
    for plan in PLANS:
        highlights = ", ".join(plan["highlights"])
        lines.append(
            f"- {plan['name']}: ${plan['price_usd_per_month']}/mo, "
            f"{plan['seats_included']} seats included — {highlights}"
        )
    return "\n".join(lines)


def _search_kb(text: str, max_results: int = 2) -> str:
    text_lower = text.lower()
    scored = []
    for article in KNOWLEDGE_BASE:
        hits = sum(1 for kw in article["keywords"] if kw in text_lower)
        if hits:
            scored.append((hits, article))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    if not scored:
        return "No matching knowledge base articles found."

    chunks = []
    for _, article in scored[:max_results]:
        chunks.append(f"[{article['title']}]\n{article['content']}")
    return "\n\n".join(chunks)


def _parse_draft(text: str) -> tuple[str, str]:
    """Split the model's RESPONSE:/NOTES: reply into (response, notes)."""
    response, _, notes = text.partition("NOTES:")
    response = response.replace("RESPONSE:", "", 1).strip()
    notes = notes.strip() or "No additional notes."
    return response, notes


def make_specialist_node(
    role: str,
    system_prompt: str,
    context_builder: Callable[[TicketState], str],
) -> Callable[[TicketState], dict]:
    def node(state: TicketState) -> dict:
        model = get_model()
        context = context_builder(state)

        user_prompt = SPECIALIST_USER_TEMPLATE.format(
            ticket_id=state["ticket_id"],
            urgency=state.get("urgency", "medium"),
            subject=state["subject"],
            message=state["message"],
            context=context,
        )

        result = model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        response_text, notes = _parse_draft(result.content)

        return {
            "specialist": role,
            "specialist_response": response_text,
            "specialist_notes": notes,
        }

    return node


BILLING_SYSTEM_PROMPT = """\
You are a billing support specialist. You help with payments, invoices, \
refunds, duplicate charges, and account lockouts caused by billing status. \
You can see the customer's plan and billing status but you cannot actually \
issue refunds or change billing state yourself — flag anything that \
requires a human to execute in your notes. Be empathetic and concrete: name \
the specific charge/amount/date if the customer mentioned one, and give a \
clear next step and timeframe.\
"""

TECHNICAL_SYSTEM_PROMPT = """\
You are a technical support specialist for a developer-facing API product. \
You help with bugs, errors, integration issues, and outages. Ground your \
answer in the knowledge base excerpts provided — don't invent behavior \
that isn't in them. If the KB doesn't cover the issue, say so plainly and \
note in your internal notes that it needs engineering escalation rather \
than guessing.\
"""

SALES_SYSTEM_PROMPT = """\
You are a sales specialist. You help prospects and existing customers with \
pricing questions, plan upgrades, seat expansion, and feature availability \
per plan. Use the plan sheet provided — don't invent pricing or features. \
Be direct about what's included at each tier and suggest a clear next \
step (e.g. connecting with an account exec for custom/Enterprise needs).\
"""

GENERAL_SYSTEM_PROMPT = """\
You are a general support agent handling a ticket that didn't clearly fit \
billing, technical, or sales. Give a helpful, honest response, and if the \
ticket actually seems like it belongs to one of those teams, say so in \
your internal notes so a human can re-route it.\
"""


billing_agent = make_specialist_node(
    role="billing",
    system_prompt=BILLING_SYSTEM_PROMPT,
    context_builder=lambda state: _format_customer_context(state["customer_id"]),
)

technical_agent = make_specialist_node(
    role="technical",
    system_prompt=TECHNICAL_SYSTEM_PROMPT,
    context_builder=lambda state: _search_kb(f"{state['subject']} {state['message']}"),
)

sales_agent = make_specialist_node(
    role="sales",
    system_prompt=SALES_SYSTEM_PROMPT,
    context_builder=lambda state: (
        _format_customer_context(state["customer_id"]) + "\n\nPlans:\n" + _format_plans()
    ),
)

general_agent = make_specialist_node(
    role="general",
    system_prompt=GENERAL_SYSTEM_PROMPT,
    context_builder=lambda state: _format_customer_context(state["customer_id"]),
)
