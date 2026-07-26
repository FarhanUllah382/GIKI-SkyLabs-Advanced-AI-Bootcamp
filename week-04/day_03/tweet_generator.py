"""
Week 4, Day 3 — Evaluator-Optimizer Loop
Generates a funny tweet, evaluates it, and iteratively optimizes it based on
feedback until it is approved or a max iteration count is reached.
"""

from langgraph.graph import StateGraph, START, END
from typing import Literal, Annotated, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from google.colab import userdata
API_KEY = userdata.get("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=API_KEY)


# -------------------------------------------------
# Structured output schema
# -------------------------------------------------
class EvaluationResult(BaseModel):
    score: int = Field(description="Quality score from 1-10")
    approved: bool = Field(description="True if the tweet is genuinely funny and ready to post, False if it needs another pass")
    feedback: str = Field(description="Specific, actionable feedback on what is working or not working - reference the actual tweet, not generic comments")


# -------------------------------------------------
# Graph state
# -------------------------------------------------
class TweetState(BaseModel):
    topic: str
    tweet_draft: str = ""
    score: int = 0
    max_iteration: int = 5
    iteration: int = 0
    approved: bool = False
    feedback: str = ""


# -------------------------------------------------
# Prompts
# -------------------------------------------------
generate_prompt = """You are a comedy writer specializing in short, punchy tweets for X (Twitter).

Write a funny, original tweet about the given topic. Follow these constraints:
- Maximum 280 characters
- Should feel like something a real, witty person would post, not a corporate joke or a dad-joke pun unless that is genuinely the funniest angle
- Use a clear comedic technique: exaggeration, unexpected twist, relatable observation, or absurd specificity
- No hashtags unless they are part of the joke itself
- No generic "as an AI" framing - write as a person

Write a funny tweet about this topic."""

evaluation_prompt = """You are a sharp, honest comedy editor reviewing tweet drafts before they go live.

Evaluate the tweet on:
1. Is it actually funny, or just tweet-shaped? (structure alone does not count as funny)
2. Is the comedic timing and punchline landing, or is it too predictable or generic?
3. Does it feel authentic and specific, or vague and safe?

Give a score from 1-10. Approve (approved=True) only if the tweet would genuinely make someone laugh or smile and feels ready to post as-is - a 7+ typically qualifies. Otherwise mark approved=False.

Your feedback must be specific to THIS tweet - point to the exact phrase or joke structure that is weak, and suggest a concrete direction for improvement. Do not give generic notes like "make it funnier."

Evaluate this tweet."""

optimize_prompt = """You are a comedy writer revising a tweet draft based on editor feedback.

You will be given the original topic, the current tweet draft, and specific feedback on what is not working. Rewrite the tweet to directly address that feedback - do not just reword it superficially, actually fix the comedic issue that was flagged.

Keep it under 280 characters. Keep the tone consistent with the original topic, but make the joke land harder.

Write an improved version of this tweet."""


# -------------------------------------------------
# Nodes
# -------------------------------------------------
def generate(state: TweetState) -> dict:
    "Generate a funny tweet post for the given topic"

    result = model.invoke([
        ("system", generate_prompt),
        ("user", f"Topic:\n{state.topic}")
    ])

    return {
        "tweet_draft": result.content,
        "iteration": state.iteration + 1,
    }


def evaluate(state: TweetState) -> dict:
    "Evaluate the quality of the tweet"

    structured_llm = model.with_structured_output(EvaluationResult)
    result: EvaluationResult = structured_llm.invoke([
        ("system", evaluation_prompt),
        ("user", f"Topic:\n{state.topic}\n\nTweet draft:\n{state.tweet_draft}")
    ])

    return {
        "score": result.score,
        "approved": result.approved,
        "feedback": result.feedback,
    }


def optimize(state: TweetState) -> dict:
    "Optimize the tweet based on feedback"

    result = model.invoke([
        ("system", optimize_prompt),
        ("user", f"Topic:\n{state.topic}\n\nTweet draft:\n{state.tweet_draft}\n\nFeedback:\n{state.feedback}")
    ])

    return {
        "tweet_draft": result.content,
        "iteration": state.iteration + 1,
    }


# -------------------------------------------------
# Conditional routing
# -------------------------------------------------
def next_node(state: TweetState) -> str:
    if state.approved:
        return "end"
    elif state.iteration >= state.max_iteration:
        return "end"
    else:
        return "optimize"


# -------------------------------------------------
# Graph: generate -> evaluate -> (optimize -> evaluate)* -> end
# -------------------------------------------------
graph = StateGraph(TweetState)

graph.add_node("generator", generate)
graph.add_node("evaluator", evaluate)
graph.add_node("optimizer", optimize)

graph.add_edge(START, "generator")
graph.add_edge("generator", "evaluator")

graph.add_conditional_edges(
    "evaluator",
    next_node,
    {
        "end": END,
        "optimize": "optimizer",
    }
)

graph.add_edge("optimizer", "evaluator")

app = graph.compile()


# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    initial_state = {"topic": "My first job interview"}
    result = app.invoke(initial_state)

    print("Final tweet:", result["tweet_draft"])
    print("Score:", result["score"])
    print("Approved:", result["approved"])
    print("Iterations:", result["iteration"])
