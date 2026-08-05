"""
Central configuration for the CRAG (Corrective RAG) pipeline.

Holds environment loading and the single shared LLM instance used by
every node/grader in the graph, so we don't instantiate a new client
in every module.
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

# --- Required API keys (raise early instead of failing deep in a node) ---
REQUIRED_ENV_VARS = ["ANTHROPIC_API_KEY", "TAVILY_API_KEY"]
missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
if missing:
    raise EnvironmentError(
        f"Missing required environment variable(s): {', '.join(missing)}. "
        f"Copy .env.example to .env and fill in the values."
    )

# --- Shared model config ---
MODEL_NAME = os.getenv("CRAG_MODEL_NAME", "claude-sonnet-4-5")
TEMPERATURE = float(os.getenv("CRAG_TEMPERATURE", "0"))

# Single shared LLM instance, reused by graders and generation.
llm = ChatAnthropic(model=MODEL_NAME, temperature=TEMPERATURE)

# --- Pipeline tunables ---
MAX_RETRIES = int(os.getenv("CRAG_MAX_RETRIES", "3"))
WEB_SEARCH_RESULTS = int(os.getenv("CRAG_WEB_SEARCH_RESULTS", "3"))
