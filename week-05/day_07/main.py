"""
CLI entry point for the CRAG pipeline.

Usage:
    python main.py "What is agent memory?"
    python main.py            # falls back to an interactive prompt
"""

import sys

from src.graph import app


def run(question: str) -> str:
    final_state = app.invoke(
        {
            "question": question,
            "generation": "",
            "documents": [],
            "web_search_needed": "No",
            "retries": 0,
        }
    )
    return final_state["generation"]


def main():
    question = " ".join(sys.argv[1:]).strip() or input("Question: ").strip()
    if not question:
        print("No question provided.")
        return

    print(f"\nQuestion: {question}\n")
    answer = run(question)
    print(f"Answer:\n{answer}")


if __name__ == "__main__":
    main()
