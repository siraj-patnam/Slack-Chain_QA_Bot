"""Shared test fixtures.

Builds a tiny SQLite database that mirrors the real schema (the ``artifacts``
table plus the ``artifacts_fts`` FTS5 index) and seeds it with a handful of
distinctive artifacts. This keeps the tool tests hermetic — no real KB, no
network — while exercising the exact MATCH→join path used in production.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class FakeToolCallingModel(BaseChatModel):
    """A chat model that returns a fixed script of AIMessages in order.

    Supports ``bind_tools`` (a no-op returning itself) so it can stand in for a
    tool-calling model inside ``create_agent``. Each invocation returns the next
    scripted message, letting a test choreograph a deterministic tool-call loop;
    every input is recorded in ``calls`` so a test can also assert what the
    model actually SAW (e.g. that history bounding dropped old tool traffic).
    """

    responses: list[AIMessage]
    index: int = 0
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        message = self.responses[self.index]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        return self


@pytest.fixture
def fake_model() -> Callable[[list[AIMessage]], FakeToolCallingModel]:
    """Factory: build a FakeToolCallingModel from a list of scripted AIMessages."""

    def _make(responses: list[AIMessage]) -> FakeToolCallingModel:
        return FakeToolCallingModel(responses=responses)

    return _make


class FakeGrader:
    """Stands in for ``model.with_structured_output(...)``; returns scripted grades."""

    def __init__(self, grades: list[Any]) -> None:
        self._grades = grades
        self._index = 0

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        grade = self._grades[self._index]
        self._index += 1
        return grade


@pytest.fixture
def fake_grader() -> Callable[[list[Any]], FakeGrader]:
    """Factory: build a FakeGrader from a list of scripted grade objects."""

    def _make(grades: list[Any]) -> FakeGrader:
        return FakeGrader(grades)

    return _make


# (customer_id, name, region, account_health)
_SEED_CUSTOMERS = [
    ("cust_bh", "BlueHarbor Logistics", "North America West", "at risk"),
    ("cust_vb", "City of Verdant Bay", "Canada", "recovering"),
    ("cust_mh", "MapleHarvest Grocers", "Canada", "watch list"),
    ("cust_au", "Aureum Payments Pty Ltd", "ANZ", "recovering"),
]

# (artifact_id, customer_id, type, title, summary, content_text)
_SEED_ARTIFACTS = [
    (
        "art_0000000000a1",
        "cust_bh",
        "customer_call",
        "QBR call with BlueHarbor Logistics",
        "Search relevance degraded after the taxonomy rollout; renewal at risk.",
        "After the taxonomy rollout the top saved searches regressed. Northstar "
        "proposed a 7-10 business day proof-of-fix: reweight the index, add a "
        "taxonomy mapping layer, and run an A/B test targeting an 80 percent "
        "top-5 hit rate before renewal.",
    ),
    (
        "art_0000000000a2",
        "cust_vb",
        "internal_document",
        "Verdant Bay emergency playbook",
        "Approved live patch window and rollback procedure.",
        "The approved live patch window is 2026-03-24 from 02:00 to 04:00 local "
        "time. If validation checks fail, run orchestrator rollback to restore "
        "the prior ruleset and replay the invalidation hook.",
    ),
    (
        "art_0000000000a3",
        "cust_mh",
        "internal_communication",
        "MapleHarvest Quebec pilot schema transform",
        "Temporary router transform maps fields ahead of the workshop.",
        "Temporary transform in the router maps txn_id to transaction_id and "
        "total_amount to amount_cents, coercing strings to integers for the "
        "Quebec pilot.",
    ),
    (
        "art_0000000000a4",
        "cust_au",
        "support_ticket",
        "Aureum SCIM attribute conflict",
        "Conflicting department and businessUnit SCIM fields.",
        "Aureum is sending both department and businessUnit variants. Jin "
        "proposed a hot-reloadable Signal Ingest preprocessing rule to normalize "
        "the attributes into one canonical field.",
    ),
]


def _build_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE customers (
              customer_id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              region TEXT NOT NULL,
              account_health TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE artifacts (
              artifact_id TEXT PRIMARY KEY,
              customer_id TEXT,
              artifact_type TEXT NOT NULL,
              title TEXT NOT NULL,
              created_at TEXT NOT NULL,
              summary TEXT NOT NULL,
              content_text TEXT NOT NULL,
              metadata_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE artifacts_fts USING fts5(
              artifact_id UNINDEXED, title, summary, content_text,
              tokenize = 'unicode61'
            )
            """
        )
        conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", _SEED_CUSTOMERS)
        for aid, cid, atype, title, summary, content in _SEED_ARTIFACTS:
            conn.execute(
                "INSERT INTO artifacts VALUES (?, ?, ?, ?, '2026-03-01T00:00:00Z', ?, ?, '{}')",
                (aid, cid, atype, title, summary, content),
            )
            conn.execute(
                "INSERT INTO artifacts_fts (artifact_id, title, summary, content_text) "
                "VALUES (?, ?, ?, ?)",
                (aid, title, summary, content),
            )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def seeded_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Create the seeded DB and point DB_PATH at it for the duration of a test."""
    db_path = tmp_path / "test.sqlite"
    _build_db(db_path)
    monkeypatch.setenv("DB_PATH", str(db_path))
    yield db_path
