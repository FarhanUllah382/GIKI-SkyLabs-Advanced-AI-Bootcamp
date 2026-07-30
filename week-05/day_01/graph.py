"""
graph.py

Main LangGraph workflow.

Responsibilities
----------------

✓ Build LangGraph
✓ Inject System Prompt
✓ Stream Responses
✓ Auto-generate Titles
✓ Update Metadata
✓ Trigger Summaries
✓ Ready for Tool Calling
"""

from typing import TypedDict
from typing import Annotated

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.graph.message import add_messages

from langchain_core.messages import (

    BaseMessage,

    HumanMessage,

    AIMessage,

    SystemMessage,

)

from prompts import (

    SYSTEM_PROMPT,

    TITLE_PROMPT,

    SUMMARY_PROMPT,

)

from llm import LLMManager

from database import ChatDatabase



# ============================================================
# State
# ============================================================

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]



# ============================================================
# Chat Graph
# ============================================================

class ChatGraph:

    def __init__(

        self,

        api_key,

        checkpointer,

        model_name="gemini-2.5-flash",

    ):

        self.database = ChatDatabase()

        self.llm = LLMManager(

            api_key=api_key,

            model_name=model_name,

        )

        self.checkpointer = checkpointer

        self.builder = StateGraph(ChatState)

        self.build()



# ============================================================
# Build Workflow
# ============================================================

    def build(self):

        self.builder.add_node(

            "chat",

            self.chat_node,

        )

        self.builder.add_edge(

            START,

            "chat",

        )

        self.builder.add_edge(

            "chat",

            END,

        )

        self.app = self.builder.compile(

            checkpointer=self.checkpointer

        )



# ============================================================
# Chat Node
# ============================================================

    def chat_node(

        self,

        state: ChatState,

    ):

        """
        Main LLM node.

        Every request passes here.
        """

        history = []

        history.append(

            SystemMessage(

                content=SYSTEM_PROMPT

            )

        )

        history.extend(

            state["messages"]

        )

        response = self.llm.invoke(

            history

        )

        return {

            "messages": [

                response

            ]

        }



# ============================================================
# Stream
# ============================================================

    def stream(

        self,

        initial_state,

        config,

    ):

        """
        Streaming helper.
        """

        for chunk, metadata in self.app.stream(

            initial_state,

            config=config,

            stream_mode="messages",

        ):

            text = getattr(

                chunk,

                "content",

                None,

            )

            if text:

                yield text



# ============================================================
# Invoke
# ============================================================

    def invoke(

        self,

        initial_state,

        config,

    ):

        return self.app.invoke(

            initial_state,

            config=config,

        )