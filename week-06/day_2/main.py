"""CLI entry point: python main.py "your topic" """

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from orchestrator import Crew


def build_client(mock: bool):
    if mock:
        return None
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        sys.exit(
            "No ANTHROPIC_API_KEY found.\n"
            "Either export one, or run with --mock to see the pipeline work."
        )
    try:
        import anthropic
    except ImportError:
        sys.exit("Install dependencies first: pip install -r requirements.txt")
    return anthropic.Anthropic(api_key=key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small multi-agent research crew.")
    parser.add_argument("topic", help="what the crew should research")
    parser.add_argument("--mock", action="store_true", help="run with no API key")
    parser.add_argument("--out", type=Path, help="write the briefing to a file")
    parser.add_argument("--max-revisions", type=int, default=2)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    crew = Crew(
        client=build_client(args.mock),
        max_revisions=args.max_revisions,
        verbose=not args.quiet,
    )
    result = crew.run(args.topic)

    print("\n" + "=" * 60)
    print(result.draft)
    print("=" * 60)
    print(f"revisions: {result.revisions}  |  verdict: {result.verdict.splitlines()[0]}")

    if args.out:
        args.out.write_text(result.draft, encoding="utf-8")
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
