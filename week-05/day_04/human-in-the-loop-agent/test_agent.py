"""
Test harness for the human-in-the-loop agent.

Run with:
    python test_agent.py
"""

import uuid
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent import graph


def run_test(name, user_message, resume_value=None, expect_interrupt=True):
    """
    Runs a fresh thread through the graph, checks whether it interrupts as
    expected, optionally resumes it, and prints a pass/fail style summary.
    """
    print(f"\n{'=' * 60}\nTEST: {name}\n{'=' * 60}")
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    user_input = {"messages": [HumanMessage(content=user_message)]}

    interrupted = False
    interrupt_payload = None

    for event in graph.stream(user_input, thread, stream_mode="updates"):
        if "__interrupt__" in event:
            interrupted = True
            interrupt_payload = event["__interrupt__"][0].value
        print(event)

    status = "PASS" if interrupted == expect_interrupt else "FAIL"
    print(f"\n[{status}] Expected interrupt={expect_interrupt}, got interrupt={interrupted}")
    if interrupted:
        print(f"   -> Interrupt payload: {interrupt_payload}")

    if interrupted and resume_value is not None:
        for event in graph.stream(Command(resume=resume_value), thread, stream_mode="updates"):
            print(event)

    return {"interrupted": interrupted, "payload": interrupt_payload, "thread": thread}


def test_weather_no_interrupt():
    """A safe tool (weather) should run without pausing for approval."""
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    user_input = {"messages": [HumanMessage(content="What's the weather in Karachi?")]}

    interrupted = False
    for event in graph.stream(user_input, thread, stream_mode="updates"):
        print(event)
        if "__interrupt__" in event:
            interrupted = True

    final_state = graph.get_state(thread)
    final_answer = final_state.values["messages"][-1].content
    print("\nFinal LLM answer:", final_answer)

    assert interrupted is False, "Weather tool should NOT require human approval"
    assert final_state.next == (), "Graph should have completed, not stayed paused"
    print("\n✅ Weather tool ran without requiring approval.")


if __name__ == "__main__":
    run_test(
        "Email requires approval",
        "Email john@example.com telling him the meeting is moved to 3pm.",
        resume_value={"approved": True},
    )
    test_weather_no_interrupt()
