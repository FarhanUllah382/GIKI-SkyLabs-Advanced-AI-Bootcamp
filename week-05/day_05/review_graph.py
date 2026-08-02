"""
Review subgraph — validates text and loops through revisions until
it's valid or a retry limit is hit.
Fully independent: has no knowledge of the parent graph's state.
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

MAX_RETRIES = 2


class ReviewState(TypedDict):
    text_to_check: str
    is_valid: bool
    revision_count: int


def validate_node(state: ReviewState) -> ReviewState:
    valid = len(state["text_to_check"].strip()) > 10
    print(f"[Review] Validating (valid={valid}): '{state['text_to_check']}'")
    return {"is_valid": valid}


def revise_node(state: ReviewState) -> ReviewState:
    new_count = state["revision_count"] + 1
    print(f"[Review] Revision attempt #{new_count}")
    return {
        "text_to_check": state["text_to_check"] + " [revised for clarity]",
        "revision_count": new_count,
    }


def review_router(state: ReviewState) -> str:
    if state["is_valid"]:
        return "approved"
    if state["revision_count"] >= MAX_RETRIES:
        print("[Review] Max retries reached, forcing approval")
        return "approved"
    return "revise"


def build_review_subgraph():
    builder = StateGraph(ReviewState)
    builder.add_node("validate", validate_node)
    builder.add_node("revise", revise_node)
    builder.add_edge(START, "validate")
    builder.add_conditional_edges(
        "validate", review_router, {"approved": END, "revise": "revise"}
    )
    builder.add_edge("revise", "validate")
    return builder.compile()


review_subgraph = build_review_subgraph()
