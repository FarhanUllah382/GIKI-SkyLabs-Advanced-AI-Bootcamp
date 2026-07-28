"""
Week 4, Day 6 — Persistent Memory Chatbot (Streamlit version)
A multi-turn chatbot using LangGraph checkpointing (MemorySaver) to retain
conversation history across turns within a thread.

Run with:
    streamlit run streamlit_memory_chatbot.py
"""

import uuid
import streamlit as st

from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Persistent Memory Chatbot", page_icon="\U0001F9E0")
st.title("\U0001F9E0 Persistent Memory Chatbot")
st.caption("LangGraph + MemorySaver checkpointing --- remembers context across turns in this thread.")

# -------------------------------------------------
# State schema
# -------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------------------------------------
# Sidebar: API key + thread controls
# -------------------------------------------------
with st.sidebar:
    st.subheader("Settings")

    default_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input("Gemini API Key", value=default_key, type="password")

    model_name = st.text_input("Model name", value="gemini-1.5-flash")
    # NOTE: use a real Gemini model id here
    # (e.g. gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash, etc.)

    st.divider()
    st.text(f"Thread ID: {st.session_state.get(\'thread_id\', \'not started yet\')}")

    if st.button("\U0001F504 New conversation (reset memory)"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()

# -------------------------------------------------
# Initialize session state
# -------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = MemorySaver()

# -------------------------------------------------
# Build the graph once and cache it
# -------------------------------------------------
@st.cache_resource
def build_app(_checkpointer, api_key: str, model_name: str):
    model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    def chat_node(state: ChatState) -> dict:
        messages = state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)
    return graph.compile(checkpointer=_checkpointer)

# -------------------------------------------------
# Render existing chat history
# -------------------------------------------------
for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)

# -------------------------------------------------
# Chat input
# -------------------------------------------------
user_message = st.chat_input("Type your message...")

if user_message:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar first.")
        st.stop()

    st.session_state.chat_history.append(("user", user_message))
    with st.chat_message("user"):
        st.markdown(user_message)

    app = build_app(st.session_state.checkpointer, api_key, model_name)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    initial_state = {"messages": [HumanMessage(content=user_message)]}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = app.invoke(initial_state, config=config)
                ai_text = result["messages"][-1].content
            except Exception as e:
                ai_text = f"\u26a0\ufe0f Error: {e}"
        st.markdown(ai_text)

    st.session_state.chat_history.append(("assistant", ai_text))
