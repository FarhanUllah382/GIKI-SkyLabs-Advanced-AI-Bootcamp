"""
LLM-backed graders used throughout the CRAG pipeline.

Each grader wraps the shared `llm` with `with_structured_output`, forcing
the model to return a small Pydantic object instead of free text, which
keeps the conditional-edge logic in `edges.py` simple and reliable.
"""

from pydantic import BaseModel, Field

from src.config import llm


class GradeDocuments(BaseModel):
    """Binary relevance check for a single retrieved document."""

    binary_score: str = Field(
        description="Document is relevant to the question, 'yes' or 'no'"
    )


class GradeHallucination(BaseModel):
    """Binary check that a generation is grounded in the supplied documents."""

    binary_score: str = Field(
        description="Answer is grounded in / supported by the facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary check that a generation actually answers the question asked."""

    binary_score: str = Field(
        description="Answer resolves the question, 'yes' or 'no'"
    )


doc_grader = llm.with_structured_output(GradeDocuments)
hallucination_grader = llm.with_structured_output(GradeHallucination)
answer_grader = llm.with_structured_output(GradeAnswer)
