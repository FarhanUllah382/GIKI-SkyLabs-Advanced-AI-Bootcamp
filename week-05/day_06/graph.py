"""
graph.py
--------
Builds and compiles the LangGraph StateGraph wiring together the
supervisor, researcher, and writer nodes.
"""

from langgraph.graph import END, StateGraph

from nodes import make_researcher_node, make_writer_node, supervisor_node
from state import AgentState


def build_graph(llm):
    """Construct and compile the multi-agent workflow graph.

    Args:
        llm: A LangChain chat model instance shared by the researcher and
             writer nodes.

    Returns:
        A compiled LangGraph graph ready to `.stream()` or `.invoke()`.
    """
    researcher_node = make_researcher_node(llm)
    writer_node = make_writer_node(llm)

    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {"researcher": "researcher", "writer": "writer", "end": END},
    )
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("writer", "supervisor")

    return workflow.compile()
