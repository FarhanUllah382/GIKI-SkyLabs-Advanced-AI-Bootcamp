"""
Intake subgraph — normalizes raw text and classifies intent.
Fully independent: has no knowledge of the parent graph's state.
"""
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END


class IntakeState(TypedDict):
    raw_text: str
    cleaned_text: Optional[str]
    detected_intent: Optional[str]


def normalize_node(state: IntakeState) -> IntakeState:
    cleaned = state["raw_text"].strip()
    print(f"[Intake] Normalized: '{cleaned}'")
    return {"cleaned_text": cleaned}


def classify_node(state: IntakeState) -> IntakeState:
    text = state["cleaned_text"].lower()
    if "bill" in text or "invoice" in text or "charge" in text:
        intent = "billing"
    elif "error" in text or "bug" in text or "crash" in text:
        intent = "technical"
    else:
        intent = "general"
    print(f"[Intake] Classified: {intent}")
    return {"detected_intent": intent}


def build_intake_subgraph():
    builder = StateGraph(IntakeState)
    builder.add_node("normalize", normalize_node)
    builder.add_node("classify", classify_node)
    builder.add_edge(START, "normalize")
    builder.add_edge("normalize", "classify")
    builder.add_edge("classify", END)
    return builder.compile()


intake_subgraph = build_intake_subgraph()
