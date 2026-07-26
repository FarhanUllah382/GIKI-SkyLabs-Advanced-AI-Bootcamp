"""
Week 4, Day 2 — Conditional Workflow + Structured Output
LLM-Based Review Handling: classifies sentiment, diagnoses negative reviews,
and routes to a tailored positive or negative reply generator.
"""

from langgraph.graph import StateGraph, START, END
from typing import Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from google.colab import userdata
API_KEY = userdata.get("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=API_KEY)


# -------------------------------------------------
# Structured output schemas
# -------------------------------------------------
class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Overall sentiment of the review")


class DiagnosisReport(BaseModel):
    issue_type: str = Field(description="Short category of the complaint, e.g. shipping delay, product defect, billing error, customer service")
    urgency: Literal["low", "medium", "high"] = Field(description="How urgently this needs to be addressed")
    user_tone: Literal["frustrated", "angry", "disappointed", "confused", "neutral"] = Field(description="The emotional tone of the reviewer")


# -------------------------------------------------
# Graph state
# -------------------------------------------------
class ReviewState(BaseModel):
    review_text: str

    sentiment: str = Field(default="")

    issue_type: str = Field(default="")
    urgency: str = Field(default="")
    user_tone: str = Field(default="")

    final_response: str = Field(default="")


# -------------------------------------------------
# Prompts
# -------------------------------------------------
sentiment_prompt = """You are a sentiment classifier analyzing customer reviews.

Read the review and classify its overall sentiment as either "positive" or "negative".
Base this on the reviewer\'s overall satisfaction, not isolated phrases \u2014 a review with one complaint but overall satisfaction is still positive, and vice versa.

Classify the sentiment of this review."""

diagnosis_prompt = """You are a customer support analyst diagnosing a negative review before a response is written.

Analyze the review and determine:
1. issue_type \u2014 the core category of complaint (e.g. shipping delay, product defect, billing error, poor customer service, quality issue).
2. urgency \u2014 how urgently this needs a response (low / medium / high), based on severity of the complaint and how strongly it\'s expressed.
3. user_tone \u2014 the reviewer\'s emotional tone (frustrated, angry, disappointed, confused, or neutral).

Be precise and base your diagnosis only on what\'s actually written in the review."""

negative_response_prompt = """You are a customer support agent writing a reply to a negative review.

You have been given a diagnosis of the review\'s issue_type, urgency, and user_tone. Use this to write an empathetic, specific reply that:
- Acknowledges the specific issue (don\'t write a generic apology)
- Matches an appropriate tone given the user\'s emotional state \u2014 de-escalate if they\'re angry/frustrated
- If urgency is high, include a clear next step or timeframe for resolution
- Avoids sounding scripted or robotic"""

positive_response_prompt = """You are a customer support agent writing a reply to a positive review.

Write a warm, genuine thank-you reply that:
- References something specific from the review (not a generic "thanks for your feedback")
- Feels human, not templated
- Keeps it brief \u2014 2-4 sentences"""


# -------------------------------------------------
# Nodes
# -------------------------------------------------
def find_sentiment(state: ReviewState) -> dict:
    structured_llm = model.with_structured_output(SentimentResult)
    result: SentimentResult = structured_llm.invoke([
        ("system", sentiment_prompt),
        ("user", f"Review:\n{state.review_text}")
    ])
    return {"sentiment": result.sentiment}


def run_diagnosis(state: ReviewState) -> dict:
    structured_llm = model.with_structured_output(DiagnosisReport)
    result: DiagnosisReport = structured_llm.invoke([
        ("system", diagnosis_prompt),
        ("user", f"Review:\n{state.review_text}")
    ])
    return {
        "issue_type": result.issue_type,
        "urgency": result.urgency,
        "user_tone": result.user_tone,
    }


def negative_response(state: ReviewState) -> dict:
    user_message = f"""
Original review:
{state.review_text}

Diagnosis:
- Issue type: {state.issue_type}
- Urgency: {state.urgency}
- User tone: {state.user_tone}
"""
    result = model.invoke([
        ("system", negative_response_prompt),
        ("user", user_message)
    ])
    return {"final_response": result.content}


def positive_response(state: ReviewState) -> dict:
    result = model.invoke([
        ("system", positive_response_prompt),
        ("user", f"Review:\n{state.review_text}")
    ])
    return {"final_response": result.content}


# -------------------------------------------------
# Conditional routing
# -------------------------------------------------
def next_node(state: ReviewState) -> str:
    if state.sentiment == "negative":
        return "run_diagnosis"
    else:
        return "positive_response"


# -------------------------------------------------
# Graph
# -------------------------------------------------
graphs = StateGraph(ReviewState)

graphs.add_node("find_sentiment", find_sentiment)
graphs.add_node("run_diagnosis", run_diagnosis)
graphs.add_node("negative_response", negative_response)
graphs.add_node("positive_response", positive_response)

graphs.add_edge(START, "find_sentiment")

graphs.add_conditional_edges(
    "find_sentiment",
    next_node,
    {
        "run_diagnosis": "run_diagnosis",
        "positive_response": "positive_response",
    }
)

graphs.add_edge("run_diagnosis", "negative_response")
graphs.add_edge("negative_response", END)
graphs.add_edge("positive_response", END)

app = graphs.compile()


# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    test_reviews = [
        "I ordered this next day delivery item 9 days ago and it still hasn\'t arrived. No one at customer service is responding. This is unacceptable and I want a refund immediately.",
        "This coffee maker has completely changed my morning routine. Fast, easy to clean, and tastes better than my old machine. Highly recommend!",
    ]

    for review in test_reviews:
        result = app.invoke({"review_text": review})
        print("=" * 60)
        print("Review:", review)
        print("Sentiment:", result["sentiment"])
        if result["sentiment"] == "negative":
            print("Issue type:", result["issue_type"])
            print("Urgency:", result["urgency"])
            print("User tone:", result["user_tone"])
        print("Response:", result["final_response"])
