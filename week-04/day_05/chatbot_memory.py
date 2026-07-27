"""
Week 4, Day 5 — Persistent Memory Chatbot
A multi-turn chatbot using LangGraph checkpointing (MemorySaver) to retain
conversation history across turns within a thread.
"""

from langgraph.graph import StateGraph, START, END
from typing import Literal, Annotated, TypedDict, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from google.colab import userdata


# -------------------------------------------------
# State
# -------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------------------------------------
# Model setup
# -------------------------------------------------
API_KEY = userdata.get("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=API_KEY)


# -------------------------------------------------
# Node
# -------------------------------------------------
def chat_node(state: ChatState) -> dict:
    messages = state["messages"]
    response = model.invoke(messages)

    return {"messages": [response]}


# -------------------------------------------------
# Graph with checkpointing (persistent memory)
# -------------------------------------------------
checkpointer = MemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

app = graph.compile(checkpointer=checkpointer)


# -------------------------------------------------
# Run — interactive multi-turn chat loop
# -------------------------------------------------
if __name__ == "__main__":
    thread_id = "thread_id_1"
    config1 = {"configurable": {"thread_id": thread_id}}

    print("--- Chatbot Start ---")
    while True:
        user_message = input("You: ")
        if user_message.strip().lower() in ["exit", "quit"]:
            print("AI: Goodbye!")
            break

        initial_state = {"messages": [HumanMessage(content=user_message)]}

        result = app.invoke(initial_state, config=config1)
        print(f"AI: {result[\'messages\'][-1].content}")
