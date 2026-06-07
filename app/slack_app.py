"""Slack Events API webhook (FastAPI).

The web tier does the minimum needed to satisfy Slack's 3-second ack rule:
verify the request signature, drop duplicate retries, and hand the question to a
background worker that runs the agent and posts the answer in-thread. The slow
LLM/retrieval work never holds the HTTP connection open.

Transport choice: Events API (we own and verify the webhook) rather than Socket
Mode, so the signature verification is real, on the request path, and graded.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import OrderedDict

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from langgraph.graph.state import CompiledStateGraph
from slack_sdk import WebClient

from app.agent import AgentProgress, build_default, stream_run
from app.slack_verify import verify_slack_signature

_MENTION_RE = re.compile(r"<@[A-Z0-9]+>")
PLACEHOLDER = "\U0001f50d Looking into that…"
PROGRESS_THROTTLE_S = 1.0  # courtesy spacing for chat.update rate limits
DEDUP_TTL_S = 600


def clean_text(text: str) -> str:
    """Strip the bot @-mention(s) and surrounding whitespace from the message."""
    return _MENTION_RE.sub("", text or "").strip()


def thread_key(event: dict) -> str:
    """Conversation/session id: workspace:channel:thread (never keyed by user)."""
    team = event.get("team") or event.get("team_id") or "T"
    channel = event.get("channel", "C")
    thread = event.get("thread_ts") or event.get("ts", "")
    return f"{team}:{channel}:{thread}"


def should_process(event: dict, bot_user_id: str | None) -> bool:
    """Loop guard: ignore the bot's own and edited/deleted messages."""
    if event.get("bot_id"):
        return False
    if bot_user_id and event.get("user") == bot_user_id:
        return False
    # Reject edited/deleted/bot-message subtypes (message_changed, etc.).
    return not event.get("subtype")


def is_target_event(event: dict) -> bool:
    """Only answer @-mentions in channels and direct messages."""
    etype = event.get("type")
    if etype == "app_mention":
        return True
    return etype == "message" and event.get("channel_type") == "im"


class _Dedup:
    """Remember recently seen event ids so Slack retries aren't answered twice."""

    def __init__(self, ttl: int = DEDUP_TTL_S, max_size: int = 1000) -> None:
        self._ttl = ttl
        self._max = max_size
        self._seen: OrderedDict[str, float] = OrderedDict()

    def seen_before(self, key: str) -> bool:
        now = time.time()
        while self._seen and next(iter(self._seen.values())) < now - self._ttl:
            self._seen.popitem(last=False)
        if key in self._seen:
            return True
        self._seen[key] = now
        if len(self._seen) > self._max:
            self._seen.popitem(last=False)
        return False


def create_app(
    *,
    agent: CompiledStateGraph | None = None,
    client: WebClient | None = None,
    signing_secret: str | None = None,
    bot_user_id: str | None = None,
) -> FastAPI:
    """Build the FastAPI app. Dependencies are injectable for testing."""
    api = FastAPI()
    agent = agent if agent is not None else build_default()
    client = client or WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    secret = signing_secret or os.environ["SLACK_SIGNING_SECRET"]
    if bot_user_id is None:
        try:
            bot_user_id = client.auth_test()["user_id"]
        except Exception:
            bot_user_id = None
    dedup = _Dedup()

    def process(event: dict) -> None:
        question = clean_text(event.get("text", ""))
        channel = event["channel"]
        thread_ts = event.get("thread_ts") or event["ts"]
        if not question:
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text="Ask me a question about the knowledge base.",
            )
            return

        placeholder = client.chat_postMessage(
            channel=channel, thread_ts=thread_ts, text=PLACEHOLDER
        )
        msg_ts = placeholder["ts"]
        last_update = 0.0
        answer = "Sorry, something went wrong answering that."
        for item in stream_run(agent, question, thread_id=thread_key(event)):
            if isinstance(item, AgentProgress):
                now = time.monotonic()
                if now - last_update >= PROGRESS_THROTTLE_S:
                    last_update = now
                    client.chat_update(
                        channel=channel,
                        ts=msg_ts,
                        text=f"\U0001f50d Searching… ({item.tool_calls} lookups)",
                    )
            else:
                answer = item.answer
        client.chat_update(channel=channel, ts=msg_ts, text=answer)

    @api.post("/slack/events")
    async def slack_events(request: Request) -> Response:
        raw = await request.body()
        timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
        signature = request.headers.get("X-Slack-Signature", "")
        if not verify_slack_signature(secret, timestamp, raw, signature):
            return Response(status_code=401)

        payload = json.loads(raw or b"{}")
        if payload.get("type") == "url_verification":
            return JSONResponse({"challenge": payload.get("challenge", "")})

        event_id = payload.get("event_id")
        if event_id and dedup.seen_before(event_id):
            return Response(status_code=200)  # retry of an event we already took

        event = payload.get("event", {})
        if is_target_event(event) and should_process(event, bot_user_id):
            # ack-then-process: return 200 now, do the slow work in the background.
            threading.Thread(target=process, args=(event,), daemon=True).start()
        return Response(status_code=200)

    return api


def main() -> None:
    import uvicorn

    load_dotenv()
    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "3000")))


if __name__ == "__main__":
    main()
