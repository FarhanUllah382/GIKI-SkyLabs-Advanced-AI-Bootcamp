"""
Conditional edge functions. These inspect GraphState after a node runs
and return a string key that `graph.py` maps to the next node.
"""

from src.config import MAX_RETRIES
from src.graders import answer_grader, hallucination_grader
from src.state import GraphState


def decide_to_generate(state: GraphState) -> str:
    """After grading docs: go straight to generate, or fall back to web search."""
    return "transform_query" if state["web_search_needed"] == "Yes" else "generate"


def grade_generation(state: GraphState) -> str:
    """
    After generating: check the answer is grounded in the documents, then
    check it actually resolves the question. Bails out to "useful" once
    MAX_RETRIES is hit so the graph can't loop forever.
    """
    if state.get("retries", 0) >= MAX_RETRIES:
        return "useful"

    grounded = hallucination_grader.invoke(
        [
            (
                "human",
                f"Documents: {state['documents']}\n\n"
                f"Answer: {state['generation']}\n\n"
                "Is this answer grounded in / supported by the documents? Answer yes or no.",
            )
        ]
    )
    if grounded.binary_score.lower() != "yes":
        return "generate"  # retry generation with the same docs

    useful = answer_grader.invoke(
        [
            (
                "human",
                f"Question: {state['question']}\n\n"
                f"Answer: {state['generation']}\n\n"
                "Does this answer resolve the question? Answer yes or no.",
            )
        ]
    )
    return "useful" if useful.binary_score.lower() == "yes" else "not useful"
