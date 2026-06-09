"""Tests for the Slack Events API webhook: loop guard, filtering, dedup, ack."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import slack_app
from app.agent import AgentProgress, AgentResult
from app.slack_app import (
    _Dedup,
    clean_text,
    create_app,
    is_target_event,
    should_process,
    thread_key,
)

SECRET = "shhh"


# --- pure helpers -----------------------------------------------------------


def test_clean_text_strips_mention() -> None:
    assert clean_text("<@U123> what is the patch window?") == "what is the patch window?"


def test_thread_key_uses_workspace_channel_thread() -> None:
    event = {"team": "T1", "channel": "C1", "ts": "9.9"}
    assert thread_key(event) == "T1:C1:9.9"
    threaded = {"team": "T1", "channel": "C1", "ts": "9.9", "thread_ts": "5.5"}
    assert thread_key(threaded) == "T1:C1:5.5"


def test_should_process_loop_guard() -> None:
    assert should_process({"user": "UHUMAN"}, "UBOT")
    assert not should_process({"bot_id": "B1"}, "UBOT")  # the bot itself
    assert not should_process({"user": "UBOT"}, "UBOT")  # our own user id
    assert not should_process({"subtype": "message_changed"}, "UBOT")  # edit


def test_is_target_event() -> None:
    assert is_target_event({"type": "app_mention"})
    assert is_target_event({"type": "message", "channel_type": "im"})
    assert not is_target_event({"type": "message", "channel_type": "channel"})
    assert not is_target_event({"type": "reaction_added"})


def test_dedup_remembers_event_ids() -> None:
    d = _Dedup()
    assert not d.seen_before("Ev1")
    assert d.seen_before("Ev1")
    assert not d.seen_before("Ev2")


# --- endpoint ---------------------------------------------------------------


class FakeSlackClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def chat_postMessage(self, **kwargs: Any) -> dict:
        self.calls.append(("post", kwargs))
        return {"ts": "111.222"}

    def chat_update(self, **kwargs: Any) -> dict:
        self.calls.append(("update", kwargs))
        return {"ok": True}


class _InlineThread:
    """Run the target synchronously so background work is deterministic in tests."""

    def __init__(self, target: Any = None, args: tuple = (), daemon: bool = False) -> None:
        self._target = target
        self._args = args

    def start(self) -> None:
        self._target(*self._args)


def _fake_stream_run(agent: Any, question: str, thread_id: str = "cli"):
    yield AgentProgress(1, ["run_sql"])
    yield AgentResult(answer=f"Answer to: {question}", tool_calls=["run_sql"])


def _sign(timestamp: str, body: bytes) -> str:
    base = b"v0:" + timestamp.encode() + b":" + body
    return "v0=" + hmac.new(SECRET.encode(), base, hashlib.sha256).hexdigest()


def _post(client: TestClient, payload: dict, *, ts: str = "10000", sign: bool = True) -> Any:
    body = json.dumps(payload).encode()
    sig = _sign(ts, body) if sign else "v0=bad"
    return client.post(
        "/slack/events",
        content=body,
        headers={
            "X-Slack-Request-Timestamp": ts,
            "X-Slack-Signature": sig,
            "Content-Type": "application/json",
        },
    )


@pytest.fixture
def fake_client() -> FakeSlackClient:
    return FakeSlackClient()


@pytest.fixture
def app_client(fake_client: FakeSlackClient, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(slack_app.threading, "Thread", _InlineThread)
    monkeypatch.setattr(slack_app, "stream_run", _fake_stream_run)
    # freeze the verifier's clock so a fixed timestamp stays "fresh"
    monkeypatch.setattr(slack_app.time, "time", lambda: 10000.0)
    api = create_app(agent=object(), client=fake_client, signing_secret=SECRET, bot_user_id="UBOT")
    return TestClient(api)


def test_rejects_bad_signature(app_client: TestClient) -> None:
    resp = _post(app_client, {"type": "event_callback"}, sign=False)
    assert resp.status_code == 401


def test_url_verification_challenge(app_client: TestClient) -> None:
    resp = _post(app_client, {"type": "url_verification", "challenge": "abc123"})
    assert resp.status_code == 200
    assert resp.json()["challenge"] == "abc123"


def test_app_mention_is_answered_in_thread(
    app_client: TestClient, fake_client: FakeSlackClient
) -> None:
    event = {
        "type": "app_mention",
        "user": "UHUMAN",
        "team": "T1",
        "channel": "C1",
        "ts": "9.9",
        "text": "<@UBOT> what is the patch window?",
    }
    resp = _post(app_client, {"type": "event_callback", "event_id": "Ev1", "event": event})
    assert resp.status_code == 200
    placeholder = fake_client.calls[0]
    assert placeholder[0] == "post"
    assert placeholder[1]["thread_ts"] == "9.9"  # replied in-thread
    final = fake_client.calls[-1]
    assert final[0] == "update"
    assert final[1]["text"] == "Answer to: what is the patch window?"


def test_midrun_error_updates_placeholder_instead_of_hanging(
    fake_client: FakeSlackClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An unhandled error mid-run must replace the "Looking into that…" placeholder
    # with a friendly notice, never leave it hanging or kill the request.
    def _boom(agent, question, thread_id="cli"):
        yield AgentProgress(1, ["run_sql"])
        raise RuntimeError("LLM 500")

    monkeypatch.setattr(slack_app.threading, "Thread", _InlineThread)
    monkeypatch.setattr(slack_app, "stream_run", _boom)
    monkeypatch.setattr(slack_app.time, "time", lambda: 10000.0)
    api = create_app(agent=object(), client=fake_client, signing_secret=SECRET, bot_user_id="UBOT")
    client = TestClient(api)

    event = {
        "type": "app_mention",
        "user": "UHUMAN",
        "channel": "C1",
        "ts": "9.9",
        "text": "<@UBOT> what is the patch window?",
    }
    resp = _post(client, {"type": "event_callback", "event_id": "EvErr", "event": event})
    assert resp.status_code == 200  # the request still acks cleanly
    final = fake_client.calls[-1]
    assert final[0] == "update"
    assert final[1]["text"] == slack_app.ERROR_NOTICE  # placeholder replaced, not hung


def test_self_message_is_ignored(app_client: TestClient, fake_client: FakeSlackClient) -> None:
    event = {"type": "app_mention", "user": "UBOT", "channel": "C1", "ts": "9.9", "text": "hi"}
    resp = _post(app_client, {"type": "event_callback", "event_id": "Ev2", "event": event})
    assert resp.status_code == 200
    assert fake_client.calls == []  # loop guard: nothing posted


def test_duplicate_event_processed_once(
    app_client: TestClient, fake_client: FakeSlackClient
) -> None:
    event = {
        "type": "app_mention",
        "user": "UHUMAN",
        "channel": "C1",
        "ts": "9.9",
        "text": "<@UBOT> hello",
    }
    body = {"type": "event_callback", "event_id": "EvDup", "event": event}
    _post(app_client, body)
    posts_after_first = sum(1 for c in fake_client.calls if c[0] == "post")
    _post(app_client, body)  # retry, same event_id
    posts_after_second = sum(1 for c in fake_client.calls if c[0] == "post")
    assert posts_after_first == 1
    assert posts_after_second == 1  # not processed again
