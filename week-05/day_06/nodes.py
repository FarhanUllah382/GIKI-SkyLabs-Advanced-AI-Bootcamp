"""
nodes.py
--------
Node functions for the multi-agent graph: researcher, writer, and
supervisor (router).
"""

from langchain_core.messages import AIMessage

from state import AgentState


def make_researcher_node(llm):
    """Return a researcher_node function bound to the given LLM."""

    def researcher_node(state: AgentState):
        query = state["messages"][-1].content
        prompt = f"""You are a Research Agent. Break down this topic and provide
3-4 key factual points as if you researched it. Topic: {query}"""
        result = llm.invoke(prompt)
        return {"messages": [AIMessage(content=result.content, name="Researcher")]}

    return researcher_node


def make_writer_node(llm):
    """Return a writer_node function bound to the given LLM."""

    def writer_node(state: AgentState):
        research = state["messages"][-1].content
        prompt = f"""You are a Writer Agent. Turn this research into a short,
clear summary (max 150 words):\n\n{research}"""
        result = llm.invoke(prompt)
        return {"messages": [AIMessage(content=result.content, name="Writer")]}

    return writer_node


def supervisor_node(state: AgentState):
    """Route to the next node based on who spoke last."""
    last_speaker = getattr(state["messages"][-1], "name", None)
    if last_speaker == "Writer":
        return {"next": "end"}
    elif last_speaker == "Researcher":
        return {"next": "writer"}
    else:
        return {"next": "researcher"}
