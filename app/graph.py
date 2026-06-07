"""Custom StateGraph: rewrite -> agent <-> tools -> ground+cite.

The prebuilt agent finds the right artifacts but tends to answer from search
snippets, missing specifics. This graph adds two nodes around the same ReAct
loop:

* **rewrite** resolves follow-ups against thread history ("their pricing" ->
  "Acme's pricing") so multi-turn retrieval has a standalone question.
* **ground** takes over for artifact-backed answers: it reads the FULL
  content_text of the retrieved artifacts and writes the final answer from that
  evidence with citations (or an honest "couldn't find"). Structured answers
  (counts/lists) and honest refusals pass through untouched.

The prebuilt create_agent path stays available behind the USE_GRAPH flag.
"""

from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from app.agent import TOOLS, default_model
from app.db import readonly_connection
from app.prompt import SCHEMA_PROMPT

_ID_RE = re.compile(r"art_[0-9a-f]{3,}")
_GROUND_MAX_ARTIFACTS = 6
_GROUND_MAX_CHARS = 2500

REWRITE_PROMPT = (
    "Rewrite the user's latest question into a standalone question using the "
    "prior conversation, resolving pronouns and references (e.g. 'their pricing' "
    "-> 'Acme's pricing'). Return ONLY the rewritten question. If it is already "
    "standalone, return it unchanged."
)

GROUND_PROMPT = (
    "You are finalizing an answer for an internal Q&A bot, grounded strictly in "
    "retrieved evidence. Using ONLY the evidence below (full artifact contents), "
    "answer the question completely and specifically — include exact names, "
    "dates, windows, commands, metrics, and the steps of any plan where present. "
    "Cite the artifact ids you used as (source: art_...). If the evidence does "
    "not actually support an answer, reply exactly: I couldn't find that in the "
    "data."
)


def _artifact_ids(messages: list) -> list[str]:
    """Artifact ids that appeared in tool results, in first-seen order."""
    seen: list[str] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            for aid in _ID_RE.findall(str(msg.content)):
                if aid not in seen:
                    seen.append(aid)
    return seen


def _fetch_evidence(ids: list[str]) -> str:
    """Full content_text for the retrieved artifacts, as a labeled evidence block."""
    ids = ids[:_GROUND_MAX_ARTIFACTS]
    if not ids:
        return ""
    placeholders = ",".join("?" * len(ids))
    with readonly_connection() as conn:
        rows = conn.execute(
            "SELECT artifact_id, artifact_type, title, content_text "
            f"FROM artifacts WHERE artifact_id IN ({placeholders})",
            ids,
        ).fetchall()
    by_id = {r["artifact_id"]: r for r in rows}
    blocks = []
    for aid in ids:
        row = by_id.get(aid)
        if row is None:
            continue
        content = str(row["content_text"])[:_GROUND_MAX_CHARS]
        blocks.append(f"[{aid}] ({row['artifact_type']}) {row['title']}\n{content}")
    return "\n\n---\n\n".join(blocks)


def _route_after_agent(state: MessagesState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "ground"


def build_graph(
    model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    base = model or default_model()
    model_with_tools = base.bind_tools(TOOLS)
    tool_node = ToolNode(TOOLS)

    def rewrite(state: MessagesState) -> dict:
        messages = state["messages"]
        humans = [m for m in messages if isinstance(m, HumanMessage)]
        if len(humans) <= 1:
            return {}  # first turn: nothing to resolve
        latest = humans[-1]
        context = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Bot'}: {m.content}"
            for m in messages
            if isinstance(m, (HumanMessage, AIMessage)) and m.content and m is not latest
        )
        resolved = base.invoke(
            [
                SystemMessage(content=REWRITE_PROMPT),
                HumanMessage(
                    content=f"Conversation so far:\n{context}\n\nLatest question: {latest.content}"
                ),
            ]
        )
        # Same id replaces the original human turn via the add_messages reducer.
        return {"messages": [HumanMessage(content=str(resolved.content), id=latest.id)]}

    def agent(state: MessagesState) -> dict:
        messages = [SystemMessage(content=SCHEMA_PROMPT), *state["messages"]]
        return {"messages": [model_with_tools.invoke(messages)]}

    def ground(state: MessagesState) -> dict:
        messages = state["messages"]
        ids = _artifact_ids(messages)
        if not ids:
            return {}  # structured answer or honest refusal — leave the draft as-is
        evidence = _fetch_evidence(ids)
        if not evidence:
            return {}
        question = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        final = base.invoke(
            [
                SystemMessage(content=GROUND_PROMPT),
                HumanMessage(content=f"Question:\n{question}\n\nEvidence:\n{evidence}"),
            ]
        )
        return {"messages": [AIMessage(content=str(final.content))]}

    graph = StateGraph(MessagesState)
    graph.add_node("rewrite", rewrite)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_node("ground", ground)
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "agent")
    graph.add_conditional_edges("agent", _route_after_agent, {"tools": "tools", "ground": "ground"})
    graph.add_edge("tools", "agent")
    graph.add_edge("ground", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
