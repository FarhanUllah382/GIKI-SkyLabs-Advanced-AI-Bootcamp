"""
Week 4, Day 2 — Conditional Workflow: Blog Generation
A LangGraph pipeline that generates a blog outline, then a full blog post,
using the Gemini API via google-genai client.
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict

from google import genai
from google.colab import userdata

# -------------------------------------------------
# Model setup
# -------------------------------------------------
API_KEY = userdata.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
model = "gemini-3.6-flash"


# -------------------------------------------------
# State
# -------------------------------------------------
class Blog(TypedDict):
    topic: str
    model_response: str
    blog: str


# -------------------------------------------------
# Nodes
# -------------------------------------------------
def generate_outline(state: Blog) -> Blog:
    "This node will generate the outline on the topic"
    topic = state['topic']

    prompt = f"""
    You are an expert content writer.

    Generate a 30 words outline for a blog on the topic:

    "{topic}"

    Only return the outline.
    """

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    state['model_response'] = response
    return state


def generate_blog(state: Blog) -> Blog:
    "Generate a blog"
    topic = state['topic']
    outline = state['model_response']

    prompt = f"""
    You are an expert blog writer.

    Write a detailed blog post about 100 words using the following information.

    Topic:
    {topic}

    Outline:
    {outline}

    Return only the blog.
    """

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    state['blog'] = response
    return state


# -------------------------------------------------
# Graph
# -------------------------------------------------
graph = StateGraph(Blog)

graph.add_node("outline", generate_outline)
graph.add_node("blog", generate_blog)

graph.add_edge(START, "outline")
graph.add_edge("outline", "blog")
graph.add_edge("blog", END)

app = graph.compile()


# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    topic = "Tourists in Pakistan"
    response = app.invoke({"topic": topic})

    fetched_topic = response['topic']
    fetched_outline = response['model_response'].candidates[0].content.parts[0].text
    fetched_blog = response['blog'].candidates[0].content.parts[0].text

    print(f"Topic: {fetched_topic}")
    print(f"
Outline:
{fetched_outline}")
    print(f"
Blog Post:
{fetched_blog}")
