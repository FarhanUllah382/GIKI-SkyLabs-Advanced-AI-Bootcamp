"""
Basic Agent (no human review)
------------------------------
A minimal LangGraph agent that can call tools (email, SQL, weather)
without any human-in-the-loop approval step. Every tool call is
executed automatically as soon as the LLM requests it.

Compare against ../human-in-the-loop-agent, which pauses for human
approval before running sensitive tools.
"""

import os
import getpass
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import tools, tools_by_name

# ---- LLM setup ----
API_KEY = os.environ.get("GEMINI_API_KEY") or getpass.getpass("Your Gemini API key: ")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY)
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def agent_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    outputs = []
    for tool_call in last_message.tool_calls:
        result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    return {"messages": outputs}


def route_after_agent(state: AgentState):
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return END
    return "tool_node"


builder = StateGraph(AgentState)
builder.add_node("agent_node", agent_node)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "agent_node")
builder.add_conditional_edges("agent_node", route_after_agent, ["tool_node", END])
builder.add_edge("tool_node", "agent_node")

graph = builder.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    thread = {"configurable": {"thread_id": "demo-1"}}
    user_input = {"messages": [HumanMessage(content="What's the weather in Karachi?")]}
    for event in graph.stream(user_input, thread, stream_mode="updates"):
        print(event)
