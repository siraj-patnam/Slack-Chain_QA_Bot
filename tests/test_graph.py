"""Tests for the custom StateGraph (rewrite -> agent <-> tools -> generate -> grade)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import ask
from app.graph import AnswerGrade, _route_after_agent, _route_after_grade, build_graph


def _tool_call(name: str, query: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"query": query}, "id": call_id, "type": "tool_call"}],
    )


def _assert_tool_calls_balanced(messages: list) -> None:
    """Every assistant tool_calls id must have a ToolMessage response, else the
    saved history is invalid for the next turn's model replay."""
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                assert tc["id"] in answered, f"unanswered tool_call {tc['id']}"


# --- pure routing -----------------------------------------------------------


def test_route_after_agent() -> None:
    assert _route_after_agent({"messages": [_tool_call("run_sql", "x", "1")]}) == "tools"
    assert _route_after_agent({"messages": [AIMessage(content="done")]}) == "generate"


def test_route_after_grade() -> None:
    assert _route_after_grade({"messages": [HumanMessage(content="feedback")]}) == "agent"
    assert _route_after_grade({"messages": [AIMessage(content="answer")]}) == "end"


# --- graph behavior ---------------------------------------------------------


@pytest.mark.usefixtures("seeded_db")
def test_generate_grounds_and_grade_passes(
    fake_model: Callable[[list[AIMessage]], object],
    fake_grader: Callable[[list[object]], object],
) -> None:
    main = fake_model(
        [
            _tool_call("search_text", "taxonomy", "c1"),
            AIMessage(content="Draft (ignored)."),
            AIMessage(content="Grounded answer (source: art_001)."),
        ]
    )
    grader = fake_grader([AnswerGrade(grounded=True, complete=True, feedback="")])
    graph = build_graph(model=main, grader=grader)
    result = ask(graph, "what proof plan for the taxonomy issue?", thread_id="g1")
    assert "search_text" in result.tool_calls
    assert result.answer == "Grounded answer (source: art_001)."


@pytest.mark.usefixtures("seeded_db")
def test_grade_triggers_one_retry_then_passes(
    fake_model: Callable[[list[AIMessage]], object],
    fake_grader: Callable[[list[object]], object],
) -> None:
    # Structured (run_sql, no artifacts) answers pass through generate untouched,
    # so the agent emits the answers directly; grade drives the retry.
    main = fake_model(
        [
            _tool_call("run_sql", "SELECT 1", "c1"),
            AIMessage(content="Incomplete answer."),
            _tool_call("run_sql", "SELECT 2", "c2"),
            AIMessage(content="Complete answer with the full set."),
        ]
    )
    grader = fake_grader(
        [
            AnswerGrade(grounded=True, complete=False, feedback="missing accounts"),
            AnswerGrade(grounded=True, complete=True, feedback=""),
        ]
    )
    graph = build_graph(model=main, grader=grader)
    result = ask(graph, "which accounts share the pattern?", thread_id="g2")
    assert result.answer == "Complete answer with the full set."


@pytest.mark.usefixtures("seeded_db")
def test_structured_answer_not_resynthesized(
    fake_model: Callable[[list[AIMessage]], object],
    fake_grader: Callable[[list[object]], object],
) -> None:
    # A run_sql answer with NO artifacts in the evidence: generate must keep the
    # agent's answer verbatim. Re-synthesizing the same rows the agent already read
    # adds nothing and risks dropping set members or mangling the citation. Only
    # two scripted responses are given — if generate wrongly re-synthesized it
    # would consume a third and IndexError.
    main = fake_model(
        [
            _tool_call("run_sql", "SELECT name FROM customers", "c1"),
            AIMessage(content="A and B (source: scenarios.pain_point)."),
        ]
    )
    grader = fake_grader([AnswerGrade(grounded=True, complete=True, feedback="")])
    graph = build_graph(model=main, grader=grader)
    result = ask(graph, "which accounts are A vs B?", thread_id="g4")
    assert "run_sql" in result.tool_calls
    assert result.answer == "A and B (source: scenarios.pain_point)."


@pytest.mark.usefixtures("seeded_db")
def test_budget_cap_leaves_balanced_history(
    fake_model: Callable[[list[AIMessage]], object],
    fake_grader: Callable[[list[object]], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With the budget at 2, the agent's 3rd tool call is never executed — it
    # would dangle in the saved history and 400 the model on the next turn.
    monkeypatch.setattr("app.graph.TOOL_CALL_LIMIT", 2)
    main = fake_model(
        [
            _tool_call("run_sql", "SELECT 1", "c1"),
            _tool_call("run_sql", "SELECT 2", "c2"),
            _tool_call("run_sql", "SELECT 3", "c3"),  # dangling — budget hit here
            AIMessage(content="Answer from what we gathered."),  # generate's synthesis
        ]
    )
    grader = fake_grader([])  # grade short-circuits on the spent budget; never called
    graph = build_graph(model=main, grader=grader)
    state = graph.invoke(
        {"messages": [HumanMessage("a question needing many lookups")]},
        config={"configurable": {"thread_id": "b1"}, "recursion_limit": 40},
    )
    _assert_tool_calls_balanced(state["messages"])
    assert state["messages"][-1].content == "Answer from what we gathered."


@pytest.mark.usefixtures("seeded_db")
def test_no_retrieval_keeps_agent_answer(
    fake_model: Callable[[list[AIMessage]], object],
    fake_grader: Callable[[list[object]], object],
) -> None:
    # Agent answers directly (no tools), so generate has no evidence and keeps it.
    main = fake_model([AIMessage(content="I couldn't find that in the data.")])
    grader = fake_grader([AnswerGrade(grounded=True, complete=True, feedback="")])
    graph = build_graph(model=main, grader=grader)
    result = ask(graph, "what is Northstar's stock price?", thread_id="g3")
    assert result.answer == "I couldn't find that in the data."
    assert result.tool_calls == []
