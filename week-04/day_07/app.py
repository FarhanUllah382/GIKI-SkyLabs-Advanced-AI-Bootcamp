"""
app.py

Main Streamlit Application
"""

import uuid
import sqlite3
import streamlit as st

from langgraph.checkpoint.sqlite import SqliteSaver

from graph import ChatGraph
from utils import (
    estimate_tokens,
    message_statistics,
)

# -------------------------------------------------------
# Page
# -------------------------------------------------------

st.set_page_config(

    page_title="AI Chatbot Pro",

    page_icon="🧠",

    layout="wide",

)

st.title("🧠 AI Chatbot Pro")

st.caption(

    "Persistent Memory • LangGraph • Gemini • SQLite"

)

# -------------------------------------------------------
# Session State
# -------------------------------------------------------

if "thread_id" not in st.session_state:

    st.session_state.thread_id = str(

        uuid.uuid4()

    )

if "history" not in st.session_state:

    st.session_state.history = []

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

with st.sidebar:

    st.header("Settings")

    api_key = st.text_input(

        "Gemini API Key",

        type="password",

    )

    model_name = st.selectbox(

        "Model",

        [

            "gemini-2.5-flash",

            "gemini-2.5-pro",

            "gemini-1.5-flash",

        ],

    )

    st.divider()

    st.header("Conversation")

    st.code(

        st.session_state.thread_id,

        language=None,

    )

    if st.button(

        "🆕 New Conversation",

        use_container_width=True,

    ):

        st.session_state.thread_id = str(

            uuid.uuid4()

        )

        st.session_state.history = []

        st.rerun()

# -------------------------------------------------------
# SQLite Checkpointer
# -------------------------------------------------------

@st.cache_resource

def get_checkpointer():

    conn = sqlite3.connect(

        "data/chatbot.db",

        check_same_thread=False,

    )

    saver = SqliteSaver(

        conn=conn

    )

    saver.setup()

    return saver

checkpointer = get_checkpointer()

# -------------------------------------------------------
# Graph
# -------------------------------------------------------

graph = None

if api_key:

    graph = ChatGraph(

        api_key=api_key,

        model_name=model_name,

        checkpointer=checkpointer,

    )

else:

    st.warning(

        "Please enter your Gemini API Key."

    )
    
    # -------------------------------------------------------
# Conversation Manager
# -------------------------------------------------------

if graph:

    conversations = graph.database.list_conversations()

    st.divider()

    st.header("💬 Conversations")

    if conversations:

        conversation_titles = []

        title_to_thread = {}

        for conv in conversations:

            title = conv["title"]

            if not title:
                title = "New Conversation"

            display = f"{title}"

            conversation_titles.append(display)

            title_to_thread[display] = conv["thread_id"]

        current_title = None

        for title, thread in title_to_thread.items():

            if thread == st.session_state.thread_id:

                current_title = title

                break

        selected = st.selectbox(

            "Resume Conversation",

            conversation_titles,

            index=conversation_titles.index(current_title)
            if current_title in conversation_titles
            else 0,

        )

        selected_thread = title_to_thread[selected]

        # ---------------------------------------------
        # Resume Conversation
        # ---------------------------------------------

        if selected_thread != st.session_state.thread_id:

            st.session_state.thread_id = selected_thread

            history = graph.load_history(selected_thread)

            st.session_state.history = history

            st.rerun()

        # ---------------------------------------------
        # Rename Thread
        # ---------------------------------------------

        with st.expander("✏ Rename Conversation"):

            new_title = st.text_input(

                "New Title",

                value=selected,

            )

            if st.button(

                "Save Title",

                use_container_width=True,

            ):

                graph.rename_thread(

                    selected_thread,

                    new_title,

                )

                st.success("Conversation renamed.")

                st.rerun()

        # ---------------------------------------------
        # Delete Thread
        # ---------------------------------------------

        with st.expander("🗑 Delete Conversation"):

            st.warning(

                "This action cannot be undone."

            )

            if st.button(

                "Delete",

                use_container_width=True,

            ):

                graph.delete_thread(

                    selected_thread

                )

                st.session_state.thread_id = str(

                    uuid.uuid4()

                )

                st.session_state.history = []

                st.success(

                    "Conversation deleted."

                )

                st.rerun()

    else:

        st.info(

            "No conversations yet."

        )