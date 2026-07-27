"""
Week 4, Day 4 — Sequential Prompt Chaining
Blog Generator: generates an outline for a topic, then writes a full blog
post based on that outline.
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from google import genai

from google.colab import userdata
API_KEY = userdata.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
model = "gemini-3.6-flash"


class Blog(TypedDict):
    topic: str
    outline: str
    blog: str


def generate_outline(state: Blog) -> dict:
    "This node will generate the outline on the topic"
    topic = state["topic"]

    prompt = f"""You are an expert content writer.

Generate a 30 words outline for a blog on the topic:

"{topic}"

Only return the outline."""

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    outline_text = response.candidates[0].content.parts[0].text

    return {"outline": outline_text}


def generate_blog(state: Blog) -> dict:
    "Generate a blog post using the topic and outline"
    topic = state["topic"]
    outline = state["outline"]

    prompt = f"""You are an expert blog writer.

Write a detailed blog post about 100 words using the following information.

Topic:
{topic}

Outline:
{outline}

Return only the blog."""

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    blog_text = response.candidates[0].content.parts[0].text

    return {"blog": blog_text}


graph = StateGraph(Blog)

graph.add_node("outline", generate_outline)
graph.add_node("blog", generate_blog)

graph.add_edge(START, "outline")
graph.add_edge("outline", "blog")
graph.add_edge("blog", END)

app = graph.compile()


if __name__ == "__main__":
    topic = "Tourists in Pakistan"

    result = app.invoke({"topic": topic})

    print(f"Topic: {result[\'topic\']}")
    print(f"\nOutline:\n{result[\'outline\']}")
    print(f"\nBlog Post:\n{result[\'blog\']}")
