"""The prebuilt agent: an LLM that uses the retrieval tools in a bounded loop.

Built on LangChain's ``create_agent`` (the current v1 ReAct agent). It is
deliberately small — model + two tools + the schema-primed prompt + a
checkpointer for multi-turn memory + a hard recursion limit so a single
question can never spiral into dozens of tool calls.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from app.prompt import SCHEMA_PROMPT
from app.tools import distinct_values, find_customer, run_sql, search_text

DEFAULT_MODEL = "gpt-4o"

# Coarse runaway guard on graph super-steps. The REAL retrieval budget is
# graph.TOOL_CALL_LIMIT (14 tool calls); this just sits above it so a pathology
# in the rewrite/agent/tools/generate/grade/retry loop can't spin forever.
# Reaching 14 tool calls costs ~30 super-steps, so this must stay comfortably
# above that or the recursion guard would trip before the tool budget binds.
RECURSION_LIMIT = 40

TOOLS = [run_sql, search_text, distinct_values, find_customer]

# Where multi-turn thread state is persisted. A separate file from the read-only
# knowledge base; data/ is gitignored.
DEFAULT_CHECKPOINT_PATH = "data/checkpoints.sqlite"


def default_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        temperature=0,
    )


def default_checkpointer() -> BaseCheckpointSaver:
    """Persistent multi-turn memory: a SqliteSaver at ``CHECKPOINT_DB_PATH``
    (default ``data/checkpoints.sqlite``), so a thread's history survives a
    process restart. The connection uses ``check_same_thread=False`` because the
    Slack bot runs the agent in background threads (SqliteSaver serializes access
    with its own lock). PostgresSaver is the drop-in production swap."""
    path = os.environ.get("CHECKPOINT_DB_PATH", DEFAULT_CHECKPOINT_PATH)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def build_agent(
    model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build the prebuilt agent.

    ``model`` is injectable so tests can drive the graph with a fake chat model;
    in production it defaults to ChatOpenAI configured from the environment.
    """
    return create_agent(
        model=model or default_model(),
        tools=TOOLS,
        system_prompt=SCHEMA_PROMPT,
        checkpointer=checkpointer or InMemorySaver(),
    )


def build_default(
    model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Return the active agent: the custom graph by default, or the prebuilt
    create_agent when USE_GRAPH is falsey (a safe, always-available fallback).

    This is the production entry point, so it persists thread memory with a
    SqliteSaver by default; pass an explicit checkpointer (e.g. InMemorySaver) to
    override. The lower-level build_agent / build_graph keep an in-memory default
    so tests stay hermetic."""
    checkpointer = checkpointer or default_checkpointer()
    if os.environ.get("USE_GRAPH", "true").lower() in ("0", "false", "no"):
        return build_agent(model=model, checkpointer=checkpointer)
    from app.graph import build_graph

    return build_graph(model=model, checkpointer=checkpointer)


@dataclass
class AgentResult:
    """The answer plus the tool calls made while producing it."""

    answer: str
    tool_calls: list[str] = field(default_factory=list)
    hit_limit: bool = False

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)


def ask(agent: CompiledStateGraph, question: str, thread_id: str = "cli") -> AgentResult:
    """Run one question through the agent and report the answer + tool usage."""
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }
    try:
        result = agent.invoke({"messages": [HumanMessage(question)]}, config=config)
    except GraphRecursionError:
        return AgentResult(
            answer="I hit my search limit before I could fully answer.",
            hit_limit=True,
        )

    messages = result["messages"]
    answer = messages[-1].content if messages else ""
    tool_calls = [m.name for m in messages if isinstance(m, ToolMessage) and m.name]
    return AgentResult(answer=str(answer), tool_calls=tool_calls)


@dataclass
class AgentProgress:
    """An interim milestone emitted while the agent is still working."""

    tool_calls: int
    tool_names: list[str] = field(default_factory=list)


def stream_run(
    agent: CompiledStateGraph, question: str, thread_id: str = "cli"
) -> Iterator[AgentProgress | AgentResult]:
    """Stream the agent run: yield AgentProgress milestones, then a final AgentResult.

    Lets a caller (the Slack bot) show live progress while the agent works, since
    Slack can't stream tokens. Each tool result is a milestone; the last yielded
    value is always the final AgentResult.
    """
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }
    tool_names: list[str] = []
    answer = ""
    try:
        for chunk in agent.stream(
            {"messages": [HumanMessage(question)]}, config=config, stream_mode="updates"
        ):
            for payload in chunk.values():
                for msg in (payload or {}).get("messages", []):
                    if isinstance(msg, ToolMessage) and msg.name:
                        tool_names.append(msg.name)
                        yield AgentProgress(len(tool_names), list(tool_names))
                    elif isinstance(msg, AIMessage) and msg.content:
                        answer = str(msg.content)
    except GraphRecursionError:
        yield AgentResult(
            answer="I hit my search limit before I could fully answer.",
            tool_calls=tool_names,
            hit_limit=True,
        )
        return
    yield AgentResult(answer=answer, tool_calls=tool_names)
