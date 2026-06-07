"""CLI harness for fast iteration: ask the agent a question from the terminal.

Usage:
    python -m app.ask "which customer is most at risk of churn?"
    python -m app.ask "what about their renewal?" --thread demo

Prints the answer and the tool calls the agent made (count + per-tool tally),
so prompt/tool changes can be judged on accuracy *and* tool-call efficiency.
"""

from __future__ import annotations

import argparse
from collections import Counter

from dotenv import load_dotenv

from app.agent import ask, build_agent


def main() -> None:
    load_dotenv()  # pick up OPENAI_API_KEY etc. from a local .env

    parser = argparse.ArgumentParser(description="Ask the Slack Q&A agent a question.")
    parser.add_argument("question", help="the question to ask")
    parser.add_argument(
        "--thread",
        default="cli",
        help="thread id for multi-turn memory (default: cli)",
    )
    args = parser.parse_args()

    agent = build_agent()
    result = ask(agent, args.question, thread_id=args.thread)

    print(result.answer)
    print("\n---")
    tally = ", ".join(f"{name} x{n}" for name, n in Counter(result.tool_calls).items())
    print(f"tool calls ({result.tool_call_count}): {tally or 'none'}")
    if result.hit_limit:
        print("note: stopped at the recursion limit")


if __name__ == "__main__":
    main()
