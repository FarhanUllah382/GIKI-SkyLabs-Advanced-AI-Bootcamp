"""
Week 4, Day 6 — Persistent Memory Chatbot (Streamlit version, SQLite-backed)

A multi-turn chatbot using LangGraph checkpointing (SqliteSaver) to retain
conversation history across turns AND across app restarts (stored in
chatbot.db). Includes:
    - Resume chat: pick any past thread_id from the sidebar and continue it
    - Streaming: assistant responses stream token-by-token

Install requirements:
    pip install streamlit langgraph langgraph-checkpoint-sqlite langchain-google-genai

Run with:
    streamlit run streamlit_memory_chatbot_sqlite.py
"""

import uuid
import sqlite3
import streamlit as st

from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Persistent Memory Chatbot", page_icon="\U0001F9E0")
st.title("\U0001F9E0 Persistent Memory Chatbot")
st.caption("LangGraph + SqliteSaver --- resumable threads, streaming responses, disk-persisted memory.")

DB_PATH = "chatbot.db"

# -------------------------------------------------
# State schema
# -------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------------------------------------
# Cached resources: sqlite connection + checkpointer (singletons for the app)
# -------------------------------------------------
@st.cache_resource
def get_checkpointer():
    conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)
    checkpointer.setup()
    return checkpointer, conn

checkpointer, conn = get_checkpointer()

# -------------------------------------------------
# Cached resource: compiled graph (rebuilt only if model/key changes)
# -------------------------------------------------
@st.cache_resource
def build_app(_checkpointer, api_key: str, model_name: str):
    model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    def chat_node(state: ChatState) -> dict:
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)
    return graph.compile(checkpointer=_checkpointer)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def list_threads(conn) -> list[str]:
    """Return distinct thread_ids that have at least one saved checkpoint."""
    try:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id")
        return [row[0] for row in cur.fetchall()]
    except Exception:
        return []

def load_history(app, thread_id: str) -> list[tuple[str, str]]:
    """Rebuild the (role, text) list for display from a checkpointed thread."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = app.get_state(config)
    except Exception:
        return []
    if not state or not state.values:
        return []
    messages = state.values.get("messages", [])
    history = []
    for m in messages:
        if isinstance(m, HumanMessage):
            history.append(("user", m.content))
        elif isinstance(m, AIMessage):
            history.append(("assistant", m.content))
    return history

def stream_response(app, initial_state, config):
    """Generator yielding text chunks for st.write_stream, using LangGraph's
    'messages' stream mode (token-level streaming from the LLM inside the node)."""
    for chunk, metadata in app.stream(initial_state, config=config, stream_mode="messages"):
        text = getattr(chunk, "content", None)
        if text:
            yield text

# -------------------------------------------------
# Sidebar: settings + conversation management
# -------------------------------------------------
with st.sidebar:
    st.subheader("Settings")

    default_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input("Gemini API Key", value=default_key, type="password")
    model_name = st.text_input("Model name", value="gemini-1.5-flash")

    st.divider()
    st.subheader("Conversations")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    threads = list_threads(conn)
    options = ["\u2795 New conversation"] + threads

    current_index = 0
    if st.session_state.thread_id in threads:
        current_index = options.index(st.session_state.thread_id)

    choice = st.selectbox("Resume a past thread", options, index=current_index)

    if choice == "\u2795 New conversation":
        if st.button("Start new conversation"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.rerun()
    else:
        if choice != st.session_state.thread_id:
            if not api_key:
                st.warning("Enter your API key above to resume a conversation.")
            else:
                app_for_resume = build_app(checkpointer, api_key, model_name)
                st.session_state.thread_id = choice
                st.session_state.chat_history = load_history(app_for_resume, choice)
                st.rerun()

    st.divider()
    st.text(f"Current Thread ID:\n{st.session_state.thread_id}")

# -------------------------------------------------
# Render existing chat history
# -------------------------------------------------
for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)

# -------------------------------------------------
# Chat input + streaming response
# -------------------------------------------------
user_message = st.chat_input("Type your message...")

if user_message:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar first.")
        st.stop()

    st.session_state.chat_history.append(("user", user_message))
    with st.chat_message("user"):
        st.markdown(user_message)

    app = build_app(checkpointer, api_key, model_name)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    initial_state = {"messages": [HumanMessage(content=user_message)]}

    with st.chat_message("assistant"):
        try:
            full_response = st.write_stream(
                stream_response(app, initial_state, config)
            )
        except Exception as e:
            full_response = f"\u26a0\ufe0f Error: {e}"
            st.markdown(full_response)

    st.session_state.chat_history.append(("assistant", full_response))
