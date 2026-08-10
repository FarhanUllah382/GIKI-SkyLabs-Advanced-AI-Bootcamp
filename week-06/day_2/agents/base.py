"""Base class shared by every agent in the crew."""

from __future__ import annotations

import os
import textwrap

MODEL = os.getenv("MODEL", "claude-sonnet-4-5")


class Agent:
    """A single-purpose worker with a name, a role and a system prompt.

    Every agent talks to the same LLM backend. What makes them different is
    the system prompt they carry and the task the orchestrator hands them.
    """

    name: str = "agent"
    role: str = "generic worker"
    system_prompt: str = "You are a helpful assistant."

    def __init__(self, client=None, verbose: bool = True):
        self.client = client  # None => mock mode
        self.verbose = verbose

    # ------------------------------------------------------------------ #

    def run(self, task: str) -> str:
        """Run the agent on a task and return its text output."""
        self._log(f"working on: {task.splitlines()[0][:70]}")

        if self.client is None:
            return self._mock(task)

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=self.system_prompt,
            messages=[{"role": "user", "content": task}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

    # ------------------------------------------------------------------ #

    def _mock(self, task: str) -> str:
        """Deterministic fake output so the pipeline runs with no API key."""
        return textwrap.dedent(
            f"""\
            [mock output from {self.name}]
            role: {self.role}
            received a task of {len(task)} characters.
            Replace this by exporting ANTHROPIC_API_KEY and dropping --mock.
            """
        ).strip()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"  \033[36m[{self.name}]\033[0m {message}")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Agent {self.name}: {self.role}>"
