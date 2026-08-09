"""Single place that constructs the Claude chat model used by every node."""

from __future__ import annotations

from functools import lru_cache

from langchain_anthropic import ChatAnthropic

MODEL_NAME = "claude-opus-5"


@lru_cache(maxsize=1)
def get_model() -> ChatAnthropic:
    return ChatAnthropic(model=MODEL_NAME, max_tokens=2048)
