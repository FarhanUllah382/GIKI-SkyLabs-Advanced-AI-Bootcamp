"""
Shared state object passed between every node in the LangGraph pipeline.
"""

from typing import List
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Attributes:
        question:          The current user question. Gets rewritten by
                            `transform_query` if retrieved documents are
                            graded as irrelevant.
        generation:         The LLM's current answer.
        documents:          Documents currently held, sourced from the
                            vectorstore and/or a web search fallback.
        web_search_needed:  "Yes" / "No" flag set by `grade_documents`,
                             read by `decide_to_generate`.
        retries:            Number of times `generate` has been re-run
                             because the previous answer failed a grading
                             check. Prevents infinite loops.
    """

    question: str
    generation: str
    documents: List[str]
    web_search_needed: str
    retries: int
