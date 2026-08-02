"""
Resolution subgraph — routes to a handler based on detected intent.
Fully independent: has no knowledge of the parent graph's state.
"""
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END


class ResolutionState(TypedDict):
    intent: str
    handler_input: str
    handler_output: Optional[str]


def billing_handler(state: ResolutionState) -> ResolutionState:
    print("[Resolution] Billing handler")
    return {"handler_output": f"Billing team response to: '{state['handler_input']}'"}


def technical_handler(state: ResolutionState) -> ResolutionState:
    print("[Resolution] Technical handler")
    return {"handler_output": f"Tech support response to: '{state['handler_input']}'"}


def general_handler(state: ResolutionState) -> ResolutionState:
    print("[Resolution] General handler")
    return {"handler_output": f"General support response to: '{state['handler_input']}'"}


def route_by_intent(state: ResolutionState) -> str:
    return state["intent"]


def build_resolution_subgraph():
    builder = StateGraph(ResolutionState)
    builder.add_node("billing", billing_handler)
    builder.add_node("technical", technical_handler)
    builder.add_node("general", general_handler)

    builder.add_conditional_edges(
        START, route_by_intent,
        {"billing": "billing", "technical": "technical", "general": "general"}
    )
    builder.add_edge("billing", END)
    builder.add_edge("technical", END)
    builder.add_edge("general", END)
    return builder.compile()


resolution_subgraph = build_resolution_subgraph()
