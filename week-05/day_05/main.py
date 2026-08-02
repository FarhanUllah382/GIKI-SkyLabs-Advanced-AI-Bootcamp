"""
Entry point — runs the parent graph over a few sample queries and
optionally renders a Mermaid PNG of the graph structure.
"""
from parent_graph import app

TEST_QUERIES = [
    "Why was I charged twice on my last invoice?",
    "The app keeps crashing when I open settings",
    "hi",
]


def run_tests():
    for q in TEST_QUERIES:
        print("=" * 60)
        print(f"USER QUERY: {q}")
        print("=" * 60)
        result = app.invoke({
            "query": q,
            "intent": None,
            "draft_response": None,
            "approved": False,
            "retry_count": 0,
        })
        print("\nFINAL STATE:", result)
        print()


def render_graph(output_path: str = "graph.png"):
    """Optional: requires the extra deps used by draw_mermaid_png()."""
    png_bytes = app.get_graph().draw_mermaid_png()
    with open(output_path, "wb") as f:
        f.write(png_bytes)
    print(f"Graph diagram saved to {output_path}")


if __name__ == "__main__":
    run_tests()
    # Uncomment to render a PNG of the graph (requires internet/mermaid deps):
    # render_graph()
