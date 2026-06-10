"""Custom StateGraph: rewrite -> agent <-> tools -> generate -> grade.

A Self-RAG style agent (LangChain's agentic-RAG + CRAG/Self-RAG patterns):

* **rewrite** resolves follow-ups against thread history ("their pricing" ->
  "Acme's pricing") so multi-turn retrieval has a standalone question.
* **agent** is the planner: a ReAct loop that decides for itself whether to use
  run_sql, search_text, or both, and in what order. The LLM plans its own
  retrieval — there is no brittle pre-classifier.
* **generate** synthesizes the final answer grounded strictly in the evidence
  the agent gathered (SQL rows and/or artifact content), with citations.
* **grade** is a structured-output (Pydantic) check: is the answer grounded in
  the evidence and complete? If not, its feedback is fed back and the agent
  retrieves again, up to MAX_RETRIES.

This keeps every routing/grading decision typed (no string-matching, no regex),
and the enumerate-vs-detail distinction dissolves: generate answers from
whatever evidence exists, and grade catches under-retrieval generically. The
prebuilt create_agent path stays available behind the USE_GRAPH flag.

Thread context is BOUNDED: prior turns are replayed to the model only as a
capped Q/A skeleton (questions + final answers, no old tool traffic), and each
turn's evidence, tool budget, and retries are scoped to that turn — so a long
Slack thread cannot grow the prompt without bound or starve later turns.
"""

from __future__ import annotations

import re
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from app.agent import BUDGET_STOP_NOTE, TOOLS, default_model
from app.db import readonly_connection
from app.prompt import SCHEMA_PROMPT

# At most this many self-correction retries, so cost stays bounded.
MAX_RETRIES = 2
# The real retrieval budget: at most this many tool calls per question. Once
# reached the agent stops retrieving and answers from what it has. This is a
# count of executed tool calls, NOT graph super-steps — the RECURSION_LIMIT in
# app.agent is only a coarse runaway guard sitting above this.
TOOL_CALL_LIMIT = 14
# Prior turns are replayed to the model as a Q/A skeleton capped to this many
# messages (~6 exchanges), so the prompt stays bounded on long threads. The cap
# is by COUNT and keeps final answers verbatim — on this bot those are the
# largest messages (set/enumeration answers run ~1-2k tokens), so the skeleton's
# ceiling sits around 10-15k tokens: O(1) in thread length, deliberately a loose
# constant. If token cost becomes the binding concern, the next lever is a token
# budget on the skeleton, or summarizing older turns instead of replaying their
# answers verbatim.
MAX_HISTORY_MESSAGES = 12
# grade's retry-feedback messages carry this name, so history bounding can tell
# them apart from real user turns.
REVIEWER_NAME = "reviewer"
_EVIDENCE_MAX_EACH = 2000
_EVIDENCE_MAX_TOTAL = 9000
_ID_RE = re.compile(r"art_[0-9a-f]{6,}")
_GROUND_MAX_ARTIFACTS = 5
_GROUND_MAX_CHARS = 4500


class QAState(MessagesState):
    """Conversation messages, the self-correction retry count, and the standalone
    question for THIS turn.

    ``question`` is captured once by ``rewrite`` (the rewritten, pronoun-resolved
    form) and read by ``generate``/``grade``. It must NOT be re-derived from the
    last HumanMessage: a retry appends a reviewer-feedback HumanMessage, so
    "last human message" would silently become the feedback string instead of the
    user's actual question.

    ``turn_tool_start`` is the executed-tool-call count at the START of this
    turn. The checkpointer accumulates messages across a thread's turns, so the
    budget must measure only THIS turn's spend — counting the whole history
    would let earlier turns starve later ones (a long Slack thread would hit
    "budget reached" on its first tool call). ``retries`` is reset at the same
    point for the same reason. Both are stamped by ``rewrite``, which runs
    exactly once per turn.

    ``turn_msg_start`` is the message index where this turn begins (its user
    question). ``agent`` replays everything before it only as a capped Q/A
    skeleton, and generate/grade scope their evidence to messages from this
    index on — prior turns' tool results must not bleed into the current
    answer's grounding.
    """

    retries: int
    question: str
    turn_tool_start: int
    turn_msg_start: int


class AnswerGrade(BaseModel):
    """Structured grade of a drafted answer against the retrieved evidence."""

    grounded: bool = Field(
        description="True if every claim in the answer is supported by the evidence. "
        "An honest 'I couldn't find that' counts as grounded."
    )
    complete: bool = Field(
        description="True if the answer addresses every part of the question, including "
        "listing the COMPLETE set when a set of items is requested."
    )
    feedback: str = Field(
        description="If grounded or complete is false, briefly say what is missing or "
        "unsupported and what to retrieve next. Empty otherwise."
    )


REWRITE_PROMPT = (
    "Rewrite the user's latest question into a standalone question using the "
    "prior conversation, resolving pronouns and references (e.g. 'their pricing' "
    "-> 'Acme's pricing'). Return ONLY the rewritten question. If it is already "
    "standalone, return it unchanged."
)

GENERATE_PROMPT = (
    "Answer the question for an internal Q&A bot, grounded STRICTLY in the "
    "retrieved evidence below (tool results — SQL rows and/or artifact content). "
    "Use only this evidence. Be complete and specific: include exact names, "
    "dates, windows, commands, metrics, and the steps of any plan; if the "
    "question asks for a set of items, list EVERY one present in the evidence. "
    "Cite sources: for a fact from an artifact, its id, e.g. (source: art_1a2b3c); "
    "for a fact from a table, the column it came from, e.g. "
    "(source: scenarios.pain_point). Never write the word 'table' as a literal "
    "citation. If the evidence does not support an answer, say so plainly."
)

GRADE_PROMPT = (
    "You grade a drafted answer for an internal Q&A bot. Given the QUESTION, the "
    "EVIDENCE retrieved, and the ANSWER, judge whether it is grounded and "
    "complete.\n"
    "- A greeting, a description of the bot's capabilities, or a clarifying "
    "question needs no evidence — mark it grounded and complete.\n"
    "- But if the answer makes factual claims about the company's data while the "
    "EVIDENCE is empty or does not support them, it is NOT grounded (the bot "
    "should have retrieved first).\n"
    "- If a set of accounts/items was requested, it is complete only if every "
    "member present in the evidence is listed.\n"
    "- An honest 'I couldn't find that in the data' is grounded. But it is "
    "COMPLETE only if the evidence shows retrieval for that specific thing was "
    "genuinely exhausted — queries or searches aimed at it coming back empty. If "
    "a PART of the question is met with 'couldn't find' while most of the tool "
    "budget is unused and the evidence shows no real attempt at that part (e.g. "
    "the entity's artifacts were never listed or read), mark it incomplete and "
    "say in the feedback exactly what to retrieve next."
)


def _history_skeleton(messages: list, end: int) -> list:
    """Prior turns condensed to their conversational skeleton, capped.

    Keeps only the user's questions and the bot's final answers from
    ``messages[:end]``. Prior turns' tool traffic (tool-calling AIMessages and
    their ToolMessages) is the bulk of a thread's tokens and is never needed
    again — whatever it proved already lives in the kept answers; dropping the
    pairs together also keeps the view valid for the model API (no tool_calls
    stub without its response). Reviewer-feedback messages from grade retries
    are dropped too. Capped to the most recent MAX_HISTORY_MESSAGES so the
    replayed history stays bounded no matter how long the thread runs.
    """
    skeleton = [
        m
        for m in messages[:end]
        if (isinstance(m, HumanMessage) and m.content and m.name != REVIEWER_NAME)
        or (isinstance(m, AIMessage) and m.content and not m.tool_calls)
    ]
    return skeleton[-MAX_HISTORY_MESSAGES:]


def _tool_results(messages: list) -> str:
    """The agent's tool outputs (SQL rows, search hits, fetched content) as evidence.

    When the caps bind, evidence keeps the most RECENT tool results (and drops the
    oldest): retrieval converges toward the answer, so the late results hold the
    decisive joins and reads while the early ones are exploration. Keeping
    first-N instead made a budget-stopped run answer from the opening
    distinct_values dumps and claim the (retrieved, but dropped) facts were
    never found. Output stays in chronological order.
    """
    parts: list[str] = []
    total = 0
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            text = str(msg.content)[:_EVIDENCE_MAX_EACH]
            parts.append(f"[{msg.name}]\n{text}")
            total += len(text)
            if total >= _EVIDENCE_MAX_TOTAL:
                break
    return "\n\n".join(reversed(parts))


def _artifact_ids(messages: list) -> list[str]:
    """Artifact ids that appeared in tool results, first-seen order."""
    seen: list[str] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            for aid in _ID_RE.findall(str(msg.content)):
                if aid not in seen:
                    seen.append(aid)
    return seen


def _fetch_full_content(ids: list[str]) -> str:
    """Force-fetch the FULL content_text of retrieved artifacts (the reliable
    grounding step), so the answer is built from complete sources, not excerpts."""
    ids = ids[:_GROUND_MAX_ARTIFACTS]
    if not ids:
        return ""
    placeholders = ",".join("?" * len(ids))
    with readonly_connection() as conn:
        rows = conn.execute(
            "SELECT a.artifact_id AS artifact_id, a.artifact_type AS artifact_type, "
            "a.title AS title, a.content_text AS content_text, cu.name AS customer_name "
            "FROM artifacts a LEFT JOIN customers cu ON cu.customer_id = a.customer_id "
            f"WHERE a.artifact_id IN ({placeholders})",
            ids,
        ).fetchall()
    by_id = {r["artifact_id"]: r for r in rows}
    blocks = []
    for aid in ids:
        row = by_id.get(aid)
        if row is None:
            continue
        content = str(row["content_text"])[:_GROUND_MAX_CHARS]
        customer = row["customer_name"] or "(no customer)"
        meta = f"({row['artifact_type']}, customer: {customer})"
        blocks.append(f"[{aid}] {meta} {row['title']}\n{content}")
    return "\n\n---\n\n".join(blocks)


def _last_human_text(messages: list) -> str:
    return next((str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)), "")


def _last_ai_text(messages: list) -> str:
    return next(
        (str(m.content) for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
        "",
    )


def _tool_call_count(messages: list) -> int:
    """Number of executed tool calls in the message history.

    Synthetic budget-stop messages are excluded: they close out UNexecuted calls
    in the saved history and must not count as spent retrieval.
    """
    return sum(
        1 for m in messages if isinstance(m, ToolMessage) and str(m.content) != BUDGET_STOP_NOTE
    )


def _budget_stop_messages(messages: list) -> list[ToolMessage]:
    """Synthetic responses for tool_calls on the trailing AIMessage that the
    budget left unexecuted.

    When ``_budget_reached`` routes a tool-calling AIMessage straight to
    ``generate``, it skips the tools node, so those calls get no ToolMessage. That
    is fine within the turn (generate/grade build isolated prompts), but the
    checkpointer SAVES the history — and on the next turn ``agent`` replays it to
    the model, which rejects an assistant tool_calls message not followed by a
    ToolMessage per id. Closing the calls out here keeps the saved history valid.
    """
    if not messages:
        return []
    last = messages[-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return []
    return [
        ToolMessage(
            content=BUDGET_STOP_NOTE,
            tool_call_id=tc["id"],
            name=tc.get("name", ""),
        )
        for tc in last.tool_calls
    ]


def _turn_tool_calls(state: QAState) -> int:
    """Executed tool calls made during THIS turn (the retrieval budget unit).

    The checkpointer accumulates messages across a thread's turns, so the count
    is taken relative to ``turn_tool_start`` (stamped by ``rewrite`` at the top
    of every turn). Counting the whole history instead would let earlier turns
    starve later ones — by turn three of a long Slack thread the budget would
    already read as spent before the first new tool call.
    """
    return _tool_call_count(state["messages"]) - state.get("turn_tool_start", 0)


def _budget_reached(state: QAState) -> bool:
    """This turn's retrieval budget (TOOL_CALL_LIMIT) is spent.

    Centralized so the agent-router and the grader stay in lockstep: the router
    stops the tool loop and the grader stops requesting retries at the SAME
    point. That lockstep is load-bearing — when the budget cuts the loop short,
    the last AIMessage still carries unsatisfied tool_calls; if the grader then
    requested a retry, that message would be re-sent to the model and the API
    would reject it. Both call this one predicate so the invariant can't drift.
    """
    return _turn_tool_calls(state) >= TOOL_CALL_LIMIT


def _route_after_agent(state: QAState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        if _budget_reached(state):
            return "generate"  # budget spent — answer from what was gathered
        return "tools"
    return "generate"


def _route_after_grade(state: MessagesState) -> str:
    # grade appends a HumanMessage with feedback only when it wants a retry.
    return "agent" if isinstance(state["messages"][-1], HumanMessage) else "end"


def build_graph(
    model: BaseChatModel | None = None,
    grader: Runnable | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    base = model or default_model()
    grader = grader if grader is not None else base.with_structured_output(AnswerGrade)
    model_with_tools = base.bind_tools(TOOLS)
    tool_node = ToolNode(TOOLS)

    def rewrite(state: QAState) -> dict:
        messages = state["messages"]
        # This turn's question is the message just appended by the caller. The
        # SAME index is both the turn_msg_start stamp (read by agent/generate/
        # grade) and the end of the prior-turn context used below — one value,
        # so the stamp and the context window can't drift apart.
        turn_msg_start = max(len(messages) - 1, 0)
        # rewrite runs exactly once per turn, so it is where the per-turn
        # counters reset: the budget measures from here (turn_tool_start) and
        # the self-correction retries start fresh. Without this, a thread's
        # earlier turns would permanently consume later turns' budget/retries.
        turn_start = {
            "turn_tool_start": _tool_call_count(messages),
            "turn_msg_start": turn_msg_start,
            "retries": 0,
        }
        humans = [m for m in messages if isinstance(m, HumanMessage)]
        if not humans:
            return turn_start
        latest = humans[-1]
        if len(humans) <= 1:
            # First turn: already standalone. Still record it as the turn's
            # question so generate/grade never fall back to scanning messages.
            return {**turn_start, "question": str(latest.content)}
        context = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Bot'}: {m.content}"
            for m in _history_skeleton(messages, turn_msg_start)
        )
        resolved = base.invoke(
            [
                SystemMessage(content=REWRITE_PROMPT),
                HumanMessage(
                    content=f"Conversation so far:\n{context}\n\nLatest question: {latest.content}"
                ),
            ]
        )
        standalone = str(resolved.content)
        # Same id replaces the original human turn via the add_messages reducer.
        return {
            **turn_start,
            "messages": [HumanMessage(content=standalone, id=latest.id)],
            "question": standalone,
        }

    def agent(state: QAState) -> dict:
        messages = state["messages"]
        start = state.get("turn_msg_start", 0)
        # Bounded context: prior turns as a capped Q/A skeleton, THIS turn in
        # full (its live tool loop is what the model is reasoning over).
        # Replaying whole turns verbatim grew the prompt without bound — every
        # old tool dump rode along on every later call of the thread.
        view = [
            SystemMessage(content=SCHEMA_PROMPT),
            *_history_skeleton(messages, start),
            *messages[start:],
        ]
        return {"messages": [model_with_tools.invoke(view)]}

    def generate(state: QAState) -> dict:
        messages = state["messages"]
        # Evidence is THIS turn's retrieval only: prior turns' tool results and
        # artifacts must not bleed into the current answer's grounding (their
        # conclusions already live in the conversation history).
        turn = messages[state.get("turn_msg_start", 0) :]
        # Close out any tool_calls the budget left dangling, so the SAVED history
        # is valid on the next turn (see _budget_stop_messages).
        stop = _budget_stop_messages(messages)
        last = messages[-1]
        agent_answered = isinstance(last, AIMessage) and bool(last.content) and not last.tool_calls
        full = _fetch_full_content(_artifact_ids(turn))
        # When the agent already produced an answer and there is no FULL artifact
        # content to add, keep that answer. Grounding only earns its keep by
        # replacing a search-EXCERPT answer with the complete source; with no
        # artifacts retrieved (a refusal, or a structured run_sql enumeration) the
        # agent already saw exactly these rows, so re-synthesizing adds nothing and
        # can only lose set members or mangle a citation.
        if agent_answered and not full:
            return {"messages": stop} if stop else {}
        # Otherwise synthesize: either to ground an excerpt answer in full artifact
        # content, or because the tool-call budget stopped the agent before it
        # answered and we must answer from whatever was gathered.
        tool_evidence = _tool_results(turn)
        if not tool_evidence and not full:
            return {"messages": stop} if stop else {}
        question = state.get("question") or _last_human_text(messages)
        evidence = tool_evidence
        if full:
            evidence = f"{tool_evidence}\n\n--- full artifact content ---\n\n{full}"
        answer = base.invoke(
            [
                SystemMessage(content=GENERATE_PROMPT),
                HumanMessage(content=f"Question:\n{question}\n\nEvidence:\n{evidence}"),
            ]
        )
        return {"messages": [*stop, AIMessage(content=str(answer.content))]}

    def grade(state: QAState) -> dict:
        if state.get("retries", 0) >= MAX_RETRIES:
            return {}  # retry budget spent — accept the current answer
        if _budget_reached(state):
            return {}  # tool budget spent — no point asking the agent to retrieve more
        messages = state["messages"]
        # Same turn scoping as generate: grade judges THIS turn's answer
        # against THIS turn's evidence.
        turn = messages[state.get("turn_msg_start", 0) :]
        question = state.get("question") or _last_human_text(messages)
        result = cast(
            AnswerGrade,
            grader.invoke(
                [
                    SystemMessage(content=GRADE_PROMPT),
                    HumanMessage(
                        content=(
                            f"QUESTION:\n{question}\n\n"
                            f"TOOL BUDGET: {_turn_tool_calls(state)} of "
                            f"{TOOL_CALL_LIMIT} calls used\n\n"
                            f"EVIDENCE:\n{_tool_results(turn) or '(none)'}\n\n"
                            f"ANSWER:\n{_last_ai_text(turn)}"
                        )
                    ),
                ]
            ),
        )
        if result.grounded and result.complete:
            return {}
        return {
            "messages": [
                HumanMessage(
                    content=f"A reviewer flagged the previous answer: {result.feedback} "
                    "Retrieve what you still need and answer again.",
                    name=REVIEWER_NAME,
                )
            ],
            "retries": state.get("retries", 0) + 1,
        }

    graph = StateGraph(QAState)
    graph.add_node("rewrite", rewrite)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_node("generate", generate)
    graph.add_node("grade", grade)
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "agent")
    graph.add_conditional_edges(
        "agent", _route_after_agent, {"tools": "tools", "generate": "generate"}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("generate", "grade")
    graph.add_conditional_edges("grade", _route_after_grade, {"agent": "agent", "end": END})
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
