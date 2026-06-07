"""The prebuilt agent: an LLM that uses the retrieval tools in a bounded loop.

Built on LangChain's ``create_agent`` (the current v1 ReAct agent). It is
deliberately small — model + two tools + the schema-primed prompt + a
checkpointer for multi-turn memory + a hard recursion limit so a single
question can never spiral into dozens of tool calls.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from app.prompt import SCHEMA_PROMPT
from app.tools import run_sql, search_text

DEFAULT_MODEL = "gpt-4o"

# Hard cap on graph super-steps. A ReAct turn costs ~2 steps per tool call
# (model node + tool node), so ~14 comfortably allows a handful of tool calls
# while still stopping a runaway loop.
RECURSION_LIMIT = 14

TOOLS = [run_sql, search_text]


def default_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        temperature=0,
    )


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
    create_agent when USE_GRAPH is falsey (a safe, always-available fallback)."""
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
