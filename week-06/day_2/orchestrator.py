"""Wires the four agents into a pipeline with a critique/revision loop."""

from __future__ import annotations

from dataclasses import dataclass, field

from agents import Critic, Planner, Researcher, Writer


@dataclass
class Result:
    topic: str
    plan: str = ""
    notes: str = ""
    draft: str = ""
    verdict: str = ""
    revisions: int = 0
    transcript: list[tuple[str, str]] = field(default_factory=list)


class Crew:
    """Runs planner -> researcher -> writer -> critic, looping on rejection."""

    def __init__(self, client=None, max_revisions: int = 2, verbose: bool = True):
        self.planner = Planner(client, verbose)
        self.researcher = Researcher(client, verbose)
        self.writer = Writer(client, verbose)
        self.critic = Critic(client, verbose)
        self.max_revisions = max_revisions
        self.verbose = verbose

    def run(self, topic: str) -> Result:
        result = Result(topic=topic)

        self._step("1/4 planning")
        result.plan = self.planner.plan(topic)
        result.transcript.append(("planner", result.plan))

        self._step("2/4 researching")
        result.notes = self.researcher.research(topic, result.plan)
        result.transcript.append(("researcher", result.notes))

        feedback = ""
        for attempt in range(self.max_revisions + 1):
            self._step(f"3/4 writing (attempt {attempt + 1})")
            result.draft = self.writer.write(topic, result.notes, feedback)
            result.transcript.append(("writer", result.draft))

            self._step("4/4 reviewing")
            result.verdict = self.critic.review(result.draft, result.notes)
            result.transcript.append(("critic", result.verdict))

            if self.critic.is_approved(result.verdict):
                self._step("critic approved the draft")
                break

            feedback = result.verdict
            if attempt < self.max_revisions:
                result.revisions = attempt + 1
                self._step("critic requested changes, revising")
        else:
            self._step("revision budget spent, returning the last draft")

        return result

    def _step(self, message: str) -> None:
        if self.verbose:
            print(f"\033[33m>\033[0m {message}")
