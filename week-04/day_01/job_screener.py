"""
Week 4, Day 1 — LangGraph Parallel Workflow + Structured Output
AI Job Screener: runs skills / experience / education analysis in parallel,
then aggregates into a final structured hiring recommendation.
"""

from langgraph.graph import StateGraph, START, END
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from google.colab import userdata
API_KEY = userdata.get("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=API_KEY)


class SkillsReport(BaseModel):
    tech_skills: List[str] = Field(description="Technical skills found in resume")
    soft_skills: List[str] = Field(description="Soft skills with clear evidence")
    skill_level: Literal["junior", "mid", "senior"]


class EducationReport(BaseModel):
    degree: str = Field(description="Highest degree obtained")
    field: str = Field(description="Field of study")
    meet_requirements: bool = Field(description="Whether degree meets minimum requirements")


class ExperienceReport(BaseModel):
    year_exp: float = Field(description="Total years of relevant professional experience")
    relevent_roles: List[str] = Field(description="Roles most relevant to apparent career track")
    career_gaps: bool = Field(description="Whether there are unexplained career gaps of 6+ months")


class HiringRecommendation(BaseModel):
    overall_score: int = Field(description="1-10 score")
    recommendation: Literal["strong_yes", "yes", "maybe", "no"]
    summary: str
    key_strengths: List[str]
    key_concerns: List[str]


class jobScreener(BaseModel):
    resume_text: str
    tech_skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    skill_level: str = Field(default="")
    year_exp: float = Field(default=0.0)
    relevent_roles: List[str] = Field(default_factory=list)
    career_gaps: bool = Field(default=False)
    degree: str = Field(default="")
    field: str = Field(default="")
    meet_requirements: bool = Field(default=False)
    overall_score: int = Field(default=0)
    recommendation: str = Field(default="")
    summary: str = Field(default="")
    key_strengths: List[str] = Field(default_factory=list)
    key_concerns: List[str] = Field(default_factory=list)


skills_prompt = """You are a technical recruiter analyzing a candidate\'s resume to extract their skill profile.

Read the resume carefully and identify:
1. Technical skills.
2. Soft skills, ONLY if there is real evidence in the text.
3. Overall skill level (junior / mid / senior).

Extract the candidate\'s skills profile."""

experience_prompt = """You are a recruiter analyzing a candidate\'s work history.

1. Calculate total years of relevant professional experience.
2. List the roles most relevant to their apparent career track.
3. Determine if there are unexplained career gaps of 6+ months.

Analyze the candidate\'s work experience."""

education_prompt = """You are a recruiter checking a candidate\'s educational background.

1. Identify the highest degree obtained and field of study.
2. Determine whether it meets the stated minimum requirement.

Evaluate whether the candidate meets this requirement."""

prompt_aggregator = """You are a senior hiring manager making a final recommendation.

Weigh the skills, experience, and education reports together using judgment,
not simple averaging. Be specific in key_strengths and key_concerns."""


def find_skills(state: jobScreener) -> dict:
    structured_llm = model.with_structured_output(SkillsReport)
    result = structured_llm.invoke([
        ("system", skills_prompt),
        ("user", f"Resume:\n{state.resume_text}")
    ])
    return {"tech_skills": result.tech_skills, "soft_skills": result.soft_skills, "skill_level": result.skill_level}


def find_education(state: jobScreener) -> dict:
    structured_llm = model.with_structured_output(EducationReport)
    result = structured_llm.invoke([
        ("system", education_prompt),
        ("user", f"Resume:\n{state.resume_text}")
    ])
    return {"degree": result.degree, "field": result.field, "meet_requirements": result.meet_requirements}


def find_experience(state: jobScreener) -> dict:
    structured_llm = model.with_structured_output(ExperienceReport)
    result = structured_llm.invoke([
        ("system", experience_prompt),
        ("user", f"Resume:\n{state.resume_text}")
    ])
    return {"year_exp": result.year_exp, "relevent_roles": result.relevent_roles, "career_gaps": result.career_gaps}


def aggregator_node(state: jobScreener) -> dict:
    structured_llm = model.with_structured_output(HiringRecommendation)
    user_message = f"""
Skills: {state.tech_skills}, {state.soft_skills}, {state.skill_level}
Experience: {state.year_exp} yrs, {state.relevent_roles}, gaps={state.career_gaps}
Education: {state.degree}, {state.field}, meets_req={state.meet_requirements}
"""
    result = structured_llm.invoke([
        ("system", prompt_aggregator),
        ("user", user_message)
    ])
    return {
        "overall_score": result.overall_score,
        "recommendation": result.recommendation,
        "summary": result.summary,
        "key_strengths": result.key_strengths,
        "key_concerns": result.key_concerns,
    }


graphs = StateGraph(jobScreener)
graphs.add_node("skills", find_skills)
graphs.add_node("experience", find_experience)
graphs.add_node("education", find_education)
graphs.add_node("aggregate", aggregator_node)
graphs.add_edge(START, "skills")
graphs.add_edge(START, "experience")
graphs.add_edge(START, "education")
graphs.add_edge("skills", "aggregate")
graphs.add_edge("experience", "aggregate")
graphs.add_edge("education", "aggregate")
graphs.add_edge("aggregate", END)
app = graphs.compile()


if __name__ == "__main__":
    resume = "PASTE YOUR RESUME TEXT HERE"
    result = app.invoke({"resume_text": resume})
    print(result)
