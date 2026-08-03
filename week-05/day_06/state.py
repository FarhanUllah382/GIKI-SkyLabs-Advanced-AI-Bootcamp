"""
state.py
--------
Shared state definition passed between nodes in the LangGraph workflow.
"""

import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State shared across all nodes in the graph.

    Attributes:
        messages: Running list of messages exchanged between agents.
                  New messages are appended (see `operator.add`) rather
                  than replacing the list.
        next: Name of the next node the supervisor wants to route to.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
