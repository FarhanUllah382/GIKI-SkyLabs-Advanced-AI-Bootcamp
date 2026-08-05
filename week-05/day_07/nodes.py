"""
Node functions for the CRAG graph. Each node takes the current
GraphState and returns a partial state update (LangGraph merges it in).
"""

from langchain_community.tools.tavily_search import TavilySearchResults

from src.config import llm, WEB_SEARCH_RESULTS
from src.graders import doc_grader
from src.retriever import retriever
from src.state import GraphState


def retrieve(state: GraphState) -> GraphState:
    """Pull candidate documents from the vectorstore for the question."""
    docs = retriever.invoke(state["question"])
    return {"documents": [d.page_content for d in docs]}


def grade_documents(state: GraphState) -> GraphState:
    """
    Filter out irrelevant documents. If ANY document is graded irrelevant,
    flag that a web search fallback should run (tune this threshold to
    your needs — e.g. require a majority instead of a single bad doc).
    """
    filtered = []
    web_search_needed = "No"

    for doc in state["documents"]:
        score = doc_grader.invoke(
            [
                (
                    "human",
                    f"Question: {state['question']}\n\n"
                    f"Document: {doc}\n\n"
                    "Is this document relevant to the question? Answer yes or no.",
                )
            ]
        )
        if score.binary_score.lower() == "yes":
            filtered.append(doc)
        else:
            web_search_needed = "Yes"

    return {"documents": filtered, "web_search_needed": web_search_needed}


def transform_query(state: GraphState) -> GraphState:
    """Rewrite the question into a better standalone web-search query."""
    rewritten = llm.invoke(
        f"Rewrite this question to be better for a web search. "
        f"Return only the rewritten question, nothing else.\n\n"
        f"Question: {state['question']}"
    )
    return {"question": rewritten.content}


def web_search(state: GraphState) -> GraphState:
    """Fall back to a live web search and append results to documents."""
    tool = TavilySearchResults(k=WEB_SEARCH_RESULTS)
    results = tool.invoke(state["question"])
    web_content = "\n".join(r["content"] for r in results)
    return {"documents": state["documents"] + [web_content]}


def generate(state: GraphState) -> GraphState:
    """Generate an answer grounded in the currently held documents."""
    context = "\n\n".join(state["documents"])
    answer = llm.invoke(
        f"Answer the question using only this context. "
        f"If the context doesn't contain the answer, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['question']}"
    )
    return {
        "generation": answer.content,
        "retries": state.get("retries", 0) + 1,
    }
