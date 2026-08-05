"""
Assembles the CRAG (Corrective RAG) StateGraph.

Flow:
    retrieve -> grade_documents -> [generate | transform_query]
    transform_query -> web_search -> generate
    generate -> [generate (retry) | transform_query (not useful) | END (useful)]
"""

from langgraph.graph import StateGraph, END

from src.edges import decide_to_generate, grade_generation
from src.nodes import generate, grade_documents, retrieve, transform_query, web_search
from src.state import GraphState


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"},
    )

    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")

    workflow.add_conditional_edges(
        "generate",
        grade_generation,
        {"generate": "generate", "not useful": "transform_query", "useful": END},
    )

    return workflow.compile()


app = build_graph()
