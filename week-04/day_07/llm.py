"""
llm.py

Centralized LLM initialization.

Advantages
----------
✓ One place to change providers
✓ Easy model switching
✓ Future support for OpenAI
✓ Future support for Ollama
✓ Future support for Anthropic
✓ Future support for Groq
"""

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

from prompts import SYSTEM_PROMPT


class LLMManager:

    def __init__(

        self,

        api_key: str,

        model_name="gemini-2.5-flash",

        temperature=0.3,

        top_p=0.95,

        max_tokens=None,

    ):

        self.model = ChatGoogleGenerativeAI(

            model=model_name,

            google_api_key=api_key,

            temperature=temperature,

            top_p=top_p,

            max_output_tokens=max_tokens,

        )

    # --------------------------------------------------
    # Normal Invoke
    # --------------------------------------------------

    def invoke(self, messages):

        return self.model.invoke(messages)

    # --------------------------------------------------
    # Stream
    # --------------------------------------------------

    def stream(self, messages):

        return self.model.stream(messages)

    # --------------------------------------------------
    # Chat with automatic System Prompt
    # --------------------------------------------------

    def chat(self, history):

        messages = [

            SystemMessage(content=SYSTEM_PROMPT)

        ]

        messages.extend(history)

        return self.model.invoke(messages)

    # --------------------------------------------------
    # Stream Chat
    # --------------------------------------------------

    def stream_chat(self, history):

        messages = [

            SystemMessage(content=SYSTEM_PROMPT)

        ]

        messages.extend(history)

        return self.model.stream(messages)

    # --------------------------------------------------
    # Generate Conversation Title
    # --------------------------------------------------

    def generate_title(

        self,

        prompt,

        conversation

    ):

        text = prompt.format(

            conversation=conversation

        )

        response = self.model.invoke(

            [

                HumanMessage(content=text)

            ]

        )

        return response.content.strip()

    # --------------------------------------------------
    # Generate Summary
    # --------------------------------------------------

    def summarize(

        self,

        prompt,

        conversation

    ):

        text = prompt.format(

            conversation=conversation

        )

        response = self.model.invoke(

            [

                HumanMessage(content=text)

            ]

        )

        return response.content.strip()

    # --------------------------------------------------
    # Reflection
    # --------------------------------------------------

    def reflect(

        self,

        prompt,

        answer

    ):

        response = self.model.invoke(

            [

                HumanMessage(

                    content=prompt + "\n\n" + answer

                )

            ]

        )

        return response.content.strip()

    # --------------------------------------------------
    # Health Check
    # --------------------------------------------------

    def ping(self):

        try:

            self.model.invoke(

                [

                    HumanMessage(

                        content="Reply with OK."

                    )

                ]

            )

            return True

        except Exception:

            return False

    # --------------------------------------------------
    # Change Model
    # --------------------------------------------------

    def set_model(

        self,

        api_key,

        model_name

    ):

        self.model = ChatGoogleGenerativeAI(

            model=model_name,

            google_api_key=api_key,

        )

    # --------------------------------------------------
    # Information
    # --------------------------------------------------

    @property
    def provider(self):

        return "Google Gemini"