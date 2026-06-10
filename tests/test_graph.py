"""Tests for the custom StateGraph (rewrite -> agent <-> tools -> generate -> grade)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import BUDGET_STOP_NOTE, ask, default_checkpointer
from app.graph import (
    GENERATE_PROMPT,
    MAX_HISTORY_MESSAGES,
    REVIEWER_NAME,
    AnswerGrade,
    _history_skeleton,
    _route_after_agent,
    _route_after_grade,
    build_graph,
)
from app.prompt import SCHEMA_PROMPT


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


def test_history_skeleton_keeps_capped_qa_only() -> None:
    # Each old turn collapses to its Q and final A: tool traffic and reviewer
    # feedback drop out, and the cap keeps only the most recent messages.
    messages: list = []
    for i in range(10):
        messages.append(HumanMessage(content=f"q{i}"))
        messages.append(_tool_call("run_sql", f"SELECT {i}", f"c{i}"))
        messages.append(ToolMessage(content="rows", tool_call_id=f"c{i}", name="run_sql"))
        messages.append(HumanMessage(content="reviewer feedback", name=REVIEWER_NAME))
        messages.append(AIMessage(content=f"a{i}"))
    skeleton = _history_skeleton(messages, len(messages))
    assert len(skeleton) == MAX_HISTORY_MESSAGES
    assert [m.content for m in skeleton][-2:] == ["q9", "a9"]  # most recent kept
    assert "q0" not in [m.content for m in skeleton]  # oldest dropped by the cap
    assert not any(isinstance(m, ToolMessage) for m in skeleton)
    assert not any(isinstance(m, AIMessage) and m.tool_calls for m in skeleton)
    assert not any(getattr(m, "name", None) == REVIEWER_NAME for m in skeleton)


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
def test_budget_is_per_turn_not_per_thread(
    fake_model: Callable[[list[AIMessage]], object],
    fake_grader: Callable[[list[object]], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The checkpointer accumulates messages across turns. The budget must be
    # measured per TURN: here turn 1 spends the whole budget (2), and turn 2 of
    # the SAME thread must still get its own fresh 2 — not arrive pre-starved
    # with its first tool call budget-stopped.
    monkeypatch.setattr("app.graph.TOOL_CALL_LIMIT", 2)
    main = fake_model(
        [
            # turn 1: spends the full budget, then answers.
            _tool_call("run_sql", "SELECT 1", "c1"),
            _tool_call("run_sql", "SELECT 2", "c2"),
            AIMessage(content="First answer."),
            # turn 2: rewrite consumes one response, then ONE tool call.
            AIMessage(content="Standalone follow-up?"),
            _tool_call("run_sql", "SELECT 3", "c3"),
            AIMessage(content="Second answer."),
        ]
    )
    # grade short-circuits on the spent budget in turn 1; consulted in turn 2.
    grader = fake_grader([AnswerGrade(grounded=True, complete=True, feedback="")])
    graph = build_graph(model=main, grader=grader)

    first = ask(graph, "a question needing many lookups", thread_id="turns")
    assert first.tool_calls == ["run_sql", "run_sql"]

    second = ask(graph, "and a follow-up?", thread_id="turns")
    assert second.answer == "Second answer."
    # The turn's single call EXECUTED (per-turn count 1 of 2) and is reported
    # turn-scoped — not [] (budget-stopped) and not 3 (whole-thread count).
    assert second.tool_calls == ["run_sql"]
    state = graph.get_state({"configurable": {"thread_id": "turns"}})  # type: ignore[arg-type]
    stop_notes = [
        m
        for m in state.values["messages"]
        if isinstance(m, ToolMessage) and str(m.content) == BUDGET_STOP_NOTE
    ]
    assert stop_notes == []


@pytest.mark.usefixtures("seeded_db")
def test_retries_reset_per_turn(
    fake_model: Callable[[list[AIMessage]], object],
    fake_grader: Callable[[list[object]], object],
) -> None:
    # Same starvation bug as the tool budget, for the grader's retry counter:
    # turn 1 exhausts MAX_RETRIES (2); on turn 2 grade must be consulted again
    # and still able to drive a retry, not short-circuit on the stale count.
    main = fake_model(
        [
            # turn 1: three drafts — grade rejects two, then retries run out.
            AIMessage(content="Draft 1."),
            AIMessage(content="Draft 2."),
            AIMessage(content="Draft 3."),
            # turn 2: rewrite consumes one response; grade rejects the first
            # draft, and the retry produces the final answer.
            AIMessage(content="Standalone follow-up?"),
            AIMessage(content="Draft 4."),
            AIMessage(content="Final answer."),
        ]
    )
    grader = fake_grader(
        [
            AnswerGrade(grounded=True, complete=False, feedback="missing detail"),
            AnswerGrade(grounded=True, complete=False, feedback="still missing"),
            AnswerGrade(grounded=True, complete=False, feedback="missing again"),
            AnswerGrade(grounded=True, complete=True, feedback=""),
        ]
    )
    graph = build_graph(model=main, grader=grader)

    first = ask(graph, "a hard question", thread_id="retries")
    assert first.answer == "Draft 3."  # retries exhausted, third draft accepted

    second = ask(graph, "a follow-up?", thread_id="retries")
    # Stale retries would accept "Draft 4." without consulting the grader.
    assert second.answer == "Final answer."


@pytest.mark.usefixtures("seeded_db")
def test_agent_view_drops_prior_turn_tool_traffic(
    fake_model: Callable[[list[AIMessage]], object],
    fake_grader: Callable[[list[object]], object],
) -> None:
    # Turn 2's model input must contain turn 1's question and answer but NONE
    # of its tool traffic — replaying old tool dumps verbatim grew the prompt
    # without bound on long threads.
    main = fake_model(
        [
            _tool_call("run_sql", "SELECT 1", "c1"),
            AIMessage(content="First answer."),
            AIMessage(content="Standalone follow-up?"),  # rewrite, turn 2
            AIMessage(content="Second answer."),
        ]
    )
    grader = fake_grader(
        [
            AnswerGrade(grounded=True, complete=True, feedback=""),
            AnswerGrade(grounded=True, complete=True, feedback=""),
        ]
    )
    graph = build_graph(model=main, grader=grader)
    ask(graph, "first question", thread_id="bounded")
    second = ask(graph, "a follow-up", thread_id="bounded")
    assert second.answer == "Second answer."

    # The last agent call — identified exactly: the planner is the one node
    # whose input leads with the schema system prompt.
    agent_inputs = [
        c
        for c in main.calls
        if c[0].content == SCHEMA_PROMPT  # type: ignore[attr-defined]
    ]
    view = agent_inputs[-1]
    contents = [str(m.content) for m in view]
    assert "first question" in contents  # prior turn's Q kept
    assert "First answer." in contents  # prior turn's final A kept
    assert "Standalone follow-up?" in contents  # this turn's question
    assert not any(isinstance(m, ToolMessage) for m in view)
    assert not any(isinstance(m, AIMessage) and m.tool_calls for m in view)


@pytest.mark.usefixtures("seeded_db")
def test_generate_evidence_scoped_to_current_turn(
    fake_model: Callable[[list[AIMessage]], object],
    fake_grader: Callable[[list[object]], object],
) -> None:
    # Turn 1 retrieves artifact a1; turn 2 retrieves a2. Turn 2's grounding
    # synthesis must see a2 only — turn 1's artifact must not ride along into
    # the current answer's evidence.
    main = fake_model(
        [
            _tool_call("search_text", "taxonomy proof renewal", "c1"),
            AIMessage(content="Turn one draft."),
            AIMessage(content="Grounded answer one."),  # generate synthesis, t1
            AIMessage(content="Standalone: patch window rollback?"),  # rewrite, t2
            _tool_call("search_text", "patch window rollback", "c2"),
            AIMessage(content="Turn two draft."),
            AIMessage(content="Grounded answer two."),  # generate synthesis, t2
        ]
    )
    grader = fake_grader(
        [
            AnswerGrade(grounded=True, complete=True, feedback=""),
            AnswerGrade(grounded=True, complete=True, feedback=""),
        ]
    )
    graph = build_graph(model=main, grader=grader)
    ask(graph, "what proof plan for the taxonomy issue?", thread_id="ev")
    second = ask(graph, "and the patch window?", thread_id="ev")
    assert second.answer == "Grounded answer two."

    generate_inputs = [
        c
        for c in main.calls
        if str(c[0].content) == GENERATE_PROMPT  # type: ignore[attr-defined]
    ]
    evidence = str(generate_inputs[-1][1].content)
    assert "art_0000000000a2" in evidence  # this turn's artifact grounds it
    assert "art_0000000000a1" not in evidence  # prior turn's artifact does not


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


@pytest.mark.usefixtures("seeded_db")
def test_sqlite_checkpointer_persists_thread_across_instances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_model: Callable[[list[AIMessage]], object],
    fake_grader: Callable[[list[object]], object],
) -> None:
    # default_checkpointer persists to CHECKPOINT_DB_PATH; a thread's state must be
    # readable by a fresh saver on the same file — i.e. it survives a restart.
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "cp.sqlite"))
    config = {"configurable": {"thread_id": "persist-1"}}

    graph = build_graph(
        model=fake_model([AIMessage(content="remembered answer")]),
        grader=fake_grader([AnswerGrade(grounded=True, complete=True, feedback="")]),
        checkpointer=default_checkpointer(),
    )
    graph.invoke({"messages": [HumanMessage("hi")]}, config)  # type: ignore[arg-type]

    # A brand-new saver + graph on the same file see the saved thread.
    reopened = build_graph(
        model=fake_model([]),
        grader=fake_grader([]),
        checkpointer=default_checkpointer(),
    )
    state = reopened.get_state(config)  # type: ignore[arg-type]
    contents = [getattr(m, "content", None) for m in state.values["messages"]]
    assert "remembered answer" in contents
