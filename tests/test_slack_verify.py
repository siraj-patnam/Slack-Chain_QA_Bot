"""Tests for the Slack webhook HMAC verifier."""

import hashlib
import hmac

from app.slack_verify import verify_slack_signature

SECRET = "test_signing_secret"


def _sign(timestamp: str, body: bytes) -> str:
    basestring = b"v0:" + timestamp.encode() + b":" + body
    digest = hmac.new(SECRET.encode(), basestring, hashlib.sha256).hexdigest()
    return "v0=" + digest


def test_accepts_valid_signature() -> None:
    body = b'{"type":"event_callback"}'
    ts = "1000000"
    sig = _sign(ts, body)
    assert verify_slack_signature(SECRET, ts, body, sig, now=1000010)


def test_rejects_tampered_body() -> None:
    body = b'{"type":"event_callback"}'
    ts = "1000000"
    sig = _sign(ts, body)
    assert not verify_slack_signature(SECRET, ts, b'{"type":"evil"}', sig, now=1000010)


def test_rejects_wrong_secret() -> None:
    body = b"hello"
    ts = "1000000"
    sig = _sign(ts, body)
    assert not verify_slack_signature("other_secret", ts, body, sig, now=1000010)


def test_rejects_stale_timestamp_replay() -> None:
    body = b"hello"
    ts = "1000000"
    sig = _sign(ts, body)
    # 6 minutes later — outside the 5-minute window.
    assert not verify_slack_signature(SECRET, ts, body, sig, now=1000000 + 360)


def test_rejects_malformed_timestamp() -> None:
    body = b"hello"
    assert not verify_slack_signature(SECRET, "not-a-number", body, "v0=abc", now=1000010)
