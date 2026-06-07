"""Slack Events API request verification.

Every inbound webhook is verified before any processing. Slack signs each
request with the app's signing secret over the string ``v0:{timestamp}:{body}``;
we recompute that HMAC-SHA256 and compare it to the ``X-Slack-Signature`` header
in constant time, and reject requests whose timestamp is too old to block
replay attacks.
"""

from __future__ import annotations

import hashlib
import hmac
import time

# Reject requests whose timestamp is more than 5 minutes from now (replay guard).
MAX_SKEW_SECONDS = 60 * 5


def verify_slack_signature(
    signing_secret: str,
    timestamp: str,
    raw_body: bytes,
    signature: str,
    *,
    now: float | None = None,
) -> bool:
    """Return True iff the request is a genuine, fresh Slack request.

    ``raw_body`` MUST be the exact bytes Slack sent — re-serializing the JSON
    first would change the bytes and break the signature.
    """
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        return False

    current = int(now if now is not None else time.time())
    if abs(current - ts) > MAX_SKEW_SECONDS:
        return False

    basestring = b"v0:" + str(ts).encode() + b":" + raw_body
    digest = hmac.new(signing_secret.encode(), basestring, hashlib.sha256).hexdigest()
    expected = "v0=" + digest
    return hmac.compare_digest(expected, signature or "")
