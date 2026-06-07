"""The prebuilt agent: an LLM that uses the retrieval tools in a bounded loop.

This is the first working version, built on LangChain's ``create_agent`` (the
current v1 ReAct agent). It is deliberately small — model + two tools + the
schema-primed prompt + a checkpointer for multi-turn memory + a hard recursion
limit so a single question can never spiral into dozens of tool calls.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
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


def _default_model() -> ChatOpenAI:
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
        model=model or _default_model(),
        tools=TOOLS,
        system_prompt=SCHEMA_PROMPT,
        checkpointer=checkpointer or InMemorySaver(),
    )


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
