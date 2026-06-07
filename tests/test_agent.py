"""Hermetic smoke test for the prebuilt agent.

Drives the real agent graph and the real tools with a fake chat model, so we
test the wiring (the agent calls a tool, the tool runs against the DB, the
answer comes back, tool calls are counted) deterministically and without an
API key.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from langchain_core.messages import AIMessage

from app.agent import ask, build_agent


@pytest.mark.usefixtures("seeded_db")
def test_agent_answers_within_tool_budget(
    fake_model: Callable[[list[AIMessage]], object],
) -> None:
    # Scripted model: first turn calls run_sql, second turn answers.
    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "run_sql",
                    "args": {"query": "SELECT COUNT(*) AS n FROM artifacts"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="There are 4 artifacts in the knowledge base."),
    ]
    agent = build_agent(model=fake_model(scripted))

    result = ask(agent, "How many artifacts are there?")

    assert "4 artifacts" in result.answer
    assert result.tool_calls == ["run_sql"]
    assert result.tool_call_count <= 5  # well within the recursion budget
    assert result.hit_limit is False
