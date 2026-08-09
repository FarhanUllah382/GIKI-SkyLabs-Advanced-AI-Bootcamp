"""Builds the LangGraph StateGraph: classify -> route -> specialist -> review."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from support_triage.nodes.classifier import classify_ticket, route_by_category
from support_triage.nodes.reviewer import review_and_finalize
from support_triage.nodes.specialists import (
    billing_agent,
    general_agent,
    sales_agent,
    technical_agent,
)
from support_triage.state import TicketState

SPECIALIST_NODES = {
    "billing": "billing_agent",
    "technical": "technical_agent",
    "sales": "sales_agent",
    "general": "general_agent",
}


def build_graph():
    builder = StateGraph(TicketState)

    builder.add_node("classify", classify_ticket)
    builder.add_node("billing_agent", billing_agent)
    builder.add_node("technical_agent", technical_agent)
    builder.add_node("sales_agent", sales_agent)
    builder.add_node("general_agent", general_agent)
    builder.add_node("review", review_and_finalize)

    builder.set_entry_point("classify")

    builder.add_conditional_edges("classify", route_by_category, SPECIALIST_NODES)

    for node_name in SPECIALIST_NODES.values():
        builder.add_edge(node_name, "review")

    builder.add_edge("review", END)

    return builder.compile()
