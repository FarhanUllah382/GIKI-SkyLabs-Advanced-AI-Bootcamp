"""
config.py
---------
Centralized environment/configuration setup.

IMPORTANT: This module must be imported BEFORE any LangChain/LangGraph
objects (LLMs, graphs, etc.) are created, since it sets the LangSmith
tracing environment variables that those objects read at construction time.

This version is written for a normal Python environment (using python-dotenv
+ os.environ) instead of Google Colab's `userdata.get`. If you are running
this in Google Colab, see the README for the small swap needed to pull
secrets from `google.colab.userdata` instead of a `.env` file.
"""

import os
from dotenv import load_dotenv

# Load variables from a local .env file (if present) into the environment.
load_dotenv()


def setup_environment() -> str:
    """Configure LangSmith tracing and Gemini credentials.

    Reads the following environment variables (typically from a `.env` file):
        LANGSMITH_API_KEY   - your LangSmith API key
        LANGCHAIN_PROJECT   - LangSmith project name (must already exist,
                               case-sensitive)
        GEMINI_API_KEY      - your Google Gemini API key

    Returns:
        The Gemini API key, for convenience when building the LLM.
    """
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    langchain_project = os.getenv("LANGCHAIN_PROJECT", "My-first-project")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not langsmith_api_key:
        raise RuntimeError(
            "LANGSMITH_API_KEY is not set. Add it to your .env file or "
            "export it in your shell before running."
        )
    if not gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env file or "
            "export it in your shell before running."
        )

    # LangSmith tracing config — must be set BEFORE instantiating any
    # LangChain/LangGraph objects.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = langchain_project

    # Some langchain-google integrations read GOOGLE_API_KEY instead of a
    # constructor argument, so set both.
    os.environ["GOOGLE_API_KEY"] = gemini_api_key

    print("Tracing enabled:", os.environ.get("LANGCHAIN_TRACING_V2"))
    print("LangSmith key set:", os.environ.get("LANGCHAIN_API_KEY") is not None)
    print("Project:", os.environ.get("LANGCHAIN_PROJECT"))

    return gemini_api_key


def verify_langsmith_connection() -> None:
    """Verify LangSmith auth works BEFORE running the graph.

    Makes a cheap API call that raises clearly if the API key or project
    configuration is wrong.
    """
    from langsmith import Client

    try:
        client = Client()
        projects = list(client.list_projects(limit=1))
        sample = projects[0].name if projects else "no projects yet"
        print("✅ LangSmith auth OK. Sample project object:", sample)
    except Exception as e:
        print("❌ LangSmith auth/project check failed:", e)
        print("Fix your API key or project name before continuing.")
        raise
