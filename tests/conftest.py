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


class FakeToolCallingModel(BaseChatModel):
    """A chat model that returns a fixed script of AIMessages in order.

    Supports ``bind_tools`` (a no-op returning itself) so it can stand in for a
    tool-calling model inside ``create_agent``. Each invocation returns the next
    scripted message, letting a test choreograph a deterministic tool-call loop.
    """

    responses: list[AIMessage]
    index: int = 0

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


# (artifact_id, type, title, summary, content_text)
_SEED_ARTIFACTS = [
    (
        "art_001",
        "customer_call",
        "QBR call with BlueHarbor Logistics",
        "Search relevance degraded after the taxonomy rollout; renewal at risk.",
        "After the taxonomy rollout the top saved searches regressed. Northstar "
        "proposed a 7-10 business day proof-of-fix: reweight the index, add a "
        "taxonomy mapping layer, and run an A/B test targeting an 80 percent "
        "top-5 hit rate before renewal.",
    ),
    (
        "art_002",
        "internal_document",
        "Verdant Bay emergency playbook",
        "Approved live patch window and rollback procedure.",
        "The approved live patch window is 2026-03-24 from 02:00 to 04:00 local "
        "time. If validation checks fail, run orchestrator rollback to restore "
        "the prior ruleset and replay the invalidation hook.",
    ),
    (
        "art_003",
        "internal_communication",
        "MapleHarvest Quebec pilot schema transform",
        "Temporary router transform maps fields ahead of the workshop.",
        "Temporary transform in the router maps txn_id to transaction_id and "
        "total_amount to amount_cents, coercing strings to integers for the "
        "Quebec pilot.",
    ),
    (
        "art_004",
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
            CREATE TABLE artifacts (
              artifact_id TEXT PRIMARY KEY,
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
        for aid, atype, title, summary, content in _SEED_ARTIFACTS:
            conn.execute(
                "INSERT INTO artifacts VALUES (?, ?, ?, '2026-03-01T00:00:00Z', ?, ?, '{}')",
                (aid, atype, title, summary, content),
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
