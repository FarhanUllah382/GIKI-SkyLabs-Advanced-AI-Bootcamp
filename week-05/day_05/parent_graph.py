"""
Parent graph — orchestrates the intake, resolution, and review
subgraphs. Each subgraph is invoked from inside a parent node with
its own independent state; results are translated back into
ParentState manually.
"""
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from intake_graph import IntakeState, intake_subgraph
from resolution_graph import ResolutionState, resolution_subgraph
from review_graph import ReviewState, review_subgraph


class ParentState(TypedDict):
    query: str
    intent: Optional[str]
    draft_response: Optional[str]
    approved: bool
    retry_count: int


def intake_node(state: ParentState) -> ParentState:
    sub_input: IntakeState = {
        "raw_text": state["query"],
        "cleaned_text": None,
        "detected_intent": None,
    }
    sub_output = intake_subgraph.invoke(sub_input)
    return {
        "query": sub_output["cleaned_text"],
        "intent": sub_output["detected_intent"],
    }


def resolution_node(state: ParentState) -> ParentState:
    sub_input: ResolutionState = {
        "intent": state["intent"],
        "handler_input": state["query"],
        "handler_output": None,
    }
    sub_output = resolution_subgraph.invoke(sub_input)
    return {"draft_response": sub_output["handler_output"]}


def review_node(state: ParentState) -> ParentState:
    sub_input: ReviewState = {
        "text_to_check": state["draft_response"],
        "is_valid": False,
        "revision_count": 0,
    }
    sub_output = review_subgraph.invoke(sub_input)
    return {
        "draft_response": sub_output["text_to_check"],
        "approved": sub_output["is_valid"],
        "retry_count": sub_output["revision_count"],
    }


def build_parent_graph():
    builder = StateGraph(ParentState)

    builder.add_node("intake", intake_node)
    builder.add_node("resolution", resolution_node)
    builder.add_node("review", review_node)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "resolution")
    builder.add_edge("resolution", "review")
    builder.add_edge("review", END)

    return builder.compile()


app = build_parent_graph()
