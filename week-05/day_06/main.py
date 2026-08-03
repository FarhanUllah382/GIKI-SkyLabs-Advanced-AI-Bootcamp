"""
main.py
-------
Entry point: sets up the environment, verifies LangSmith connectivity,
builds the LLM and graph, and runs a sample query through it.

Run with:
    python main.py
"""

import os

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from config import setup_environment, verify_langsmith_connection
from llm import build_llm
from graph import build_graph


def main():
    # 1. Configure environment variables (must happen before creating any
    #    LangChain/LangGraph objects).
    gemini_api_key = setup_environment()

    # 2. Verify LangSmith auth works before running the graph.
    verify_langsmith_connection()

    # 3. Build the LLM and compile the graph.
    llm = build_llm(gemini_api_key)
    graph = build_graph(llm)

    # 4. Run the graph — this run will show up in your LangSmith project.
    inputs = {
        "messages": [
            HumanMessage(content="Impact of AI agents on software engineering jobs")
        ],
        "next": "",
    }
    config = RunnableConfig(run_name="ai-swe-jobs-run", tags=["colab-test"])

    for output in graph.stream(inputs, config=config):
        for node_name, value in output.items():
            print(f"--- {node_name} ---")
            if "messages" in value:
                print(value["messages"][-1].content)
            print()

    print("\n✅ Check https://smith.langchain.com and open project:", os.environ["LANGCHAIN_PROJECT"])


if __name__ == "__main__":
    main()
