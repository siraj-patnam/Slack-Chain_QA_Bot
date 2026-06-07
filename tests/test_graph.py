"""Tests for the custom StateGraph (rewrite -> agent <-> tools -> ground)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.agent import ask
from app.graph import _artifact_ids, _route_after_agent, build_graph


def _tool_call(name: str, query: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"query": query}, "id": call_id, "type": "tool_call"}],
    )


# --- pure helpers -----------------------------------------------------------


def test_artifact_ids_extracted_in_order() -> None:
    messages = [
        ToolMessage(content="[art_001] foo [art_002] bar", tool_call_id="1", name="search_text"),
        ToolMessage(content="see art_002 and art_003", tool_call_id="2", name="search_text"),
    ]
    assert _artifact_ids(messages) == ["art_001", "art_002", "art_003"]


def test_route_after_agent() -> None:
    assert _route_after_agent({"messages": [_tool_call("run_sql", "x", "1")]}) == "tools"
    assert _route_after_agent({"messages": [AIMessage(content="done")]}) == "ground"


# --- graph behavior ---------------------------------------------------------


@pytest.mark.usefixtures("seeded_db")
def test_graph_grounds_artifact_answer_from_full_content(
    fake_model: Callable[[list[AIMessage]], object],
) -> None:
    # agent: search_text -> draft -> (ground) final answer
    scripted = [
        _tool_call("search_text", "taxonomy rollout", "c1"),
        AIMessage(content="Draft: BlueHarbor had a taxonomy issue."),
        AIMessage(content="Grounded: BlueHarbor proof-of-fix plan (source: art_001)."),
    ]
    graph = build_graph(model=fake_model(scripted))
    result = ask(graph, "taxonomy rollout proof plan?", thread_id="g1")
    assert "search_text" in result.tool_calls
    assert result.answer == "Grounded: BlueHarbor proof-of-fix plan (source: art_001)."


@pytest.mark.usefixtures("seeded_db")
def test_graph_passes_through_structured_answer(
    fake_model: Callable[[list[AIMessage]], object],
) -> None:
    # run_sql returns no artifact ids, so ground leaves the draft untouched.
    scripted = [
        _tool_call("run_sql", "SELECT COUNT(*) AS n FROM artifacts", "c1"),
        AIMessage(content="There are 4 artifacts."),
    ]
    graph = build_graph(model=fake_model(scripted))
    result = ask(graph, "how many artifacts?", thread_id="g2")
    assert result.answer == "There are 4 artifacts."
