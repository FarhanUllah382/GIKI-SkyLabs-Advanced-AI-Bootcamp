"""
llm.py
------
Builds the Gemini LLM used by all agent nodes.
"""

from langchain_google_genai import ChatGoogleGenerativeAI


def build_llm(gemini_api_key: str, model: str = "gemini-2.5-flash") -> ChatGoogleGenerativeAI:
    """Create and return a configured Gemini chat model instance."""
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=gemini_api_key,
    )
