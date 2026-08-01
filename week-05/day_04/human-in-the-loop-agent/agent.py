"""
Human-in-the-Loop Agent
------------------------
Same idea as ../basic-agent, but any call to a "sensitive" tool
(sending an email, running a SQL query) pauses execution via a
graph interrupt and waits for human approval before running.
Safe tools (like weather) still execute automatically.
"""

import os
import getpass
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import tools, tools_by_name, SENSITIVE_TOOLS

# ---- LLM setup ----
API_KEY = os.environ.get("GEMINI_API_KEY") or getpass.getpass("Your Gemini API key: ")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY)
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def agent_node(state: AgentState):
    print("---Agent thinking---")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def human_approval(state: AgentState):
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]  # handling one call at a time for clarity

    print(f"---Pausing for approval of: {tool_call['name']}({tool_call['args']})---")
    decision = interrupt(
        {
            "action": "approve_tool_call",
            "tool_name": tool_call["name"],
            "tool_args": tool_call["args"],
        }
    )

    if decision.get("approved"):
        return {}  # go execute the tool as-is
    else:
        # Human rejected -- feed that back to the LLM as a ToolMessage so it can react
        reject_msg = ToolMessage(
            content=f"Human rejected this tool call. Reason: {decision.get('reason', 'not specified')}",
            tool_call_id=tool_call["id"],
        )
        return {"messages": [reject_msg]}


def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    outputs = []
    for tool_call in last_message.tool_calls:
        result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        print(f"---Executed {tool_call['name']} -> {result}---")
        outputs.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    return {"messages": outputs}


def route_after_agent(state: AgentState):
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return END
    tool_name = last_message.tool_calls[0]["name"]
    if tool_name in SENSITIVE_TOOLS:
        return "human_approval"
    return "tool_node"  # safe tool, skip human review


def route_after_approval(state: AgentState):
    last_message = state["messages"][-1]
    # If we just appended a rejection ToolMessage, go back to the agent to let it respond
    if isinstance(last_message, ToolMessage):
        return "agent_node"
    return "tool_node"


builder = StateGraph(AgentState)
builder.add_node("agent_node", agent_node)
builder.add_node("human_approval", human_approval)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "agent_node")
builder.add_conditional_edges(
    "agent_node", route_after_agent, ["human_approval", "tool_node", END]
)
builder.add_conditional_edges(
    "human_approval", route_after_approval, ["agent_node", "tool_node"]
)
builder.add_edge("tool_node", "agent_node")

graph = builder.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    thread = {"configurable": {"thread_id": "user-1"}}
    user_input = {
        "messages": [
            HumanMessage(content="Email john@example.com telling him the meeting is moved to 3pm.")
        ]
    }
    for event in graph.stream(user_input, thread, stream_mode="updates"):
        print(event)
        print()
