# Slack Q&A Bot

A Slack bot that answers natural-language questions grounded in a company SQLite
knowledge base (customer calls, support tickets, internal docs and comms,
competitor research). Built on **LangGraph**: a custom `StateGraph` plans its own
retrieval over a hybrid SQL + full-text layer, grounds every answer strictly in
the data, and replies in-thread with live progress.

## Features

- **Retrieval-planning agent.** A custom `StateGraph` decides for itself which
  tools to call (the prebuilt ReAct agent stays available behind `USE_GRAPH=false`
  as a fallback). `rewrite` resolves follow-ups against thread history; `agent` /
  `tools` is the ReAct loop under a tool-call budget; `generate` grounds the
  answer in the full content of retrieved artifacts; `grade` is a typed
  groundedness/completeness check that can trigger one self-correction.
- **Explore-then-commit retrieval.** Instead of guessing values from the schema,
  the agent discovers them: `distinct_values` lists a column's real categories and
  `find_customer` fuzzily resolves a customer name *before* it filters. `run_sql`
  reads full document text; `search_text` (FTS5) is the topic-search fallback.
  Every artifact is attributed to its owning customer by foreign key, never by a
  name read out of the prose.
- **Grounded and cited.** Answers are built strictly from retrieved evidence with
  artifact-id / table citations — or an honest "I couldn't find that in the data."
- **Persistent memory.** Multi-turn thread state is checkpointed to SQLite
  (`SqliteSaver`), so a conversation survives a process restart.

## Quick start

```bash
# 1. Install deps (uses uv)
uv sync

# 2. Configure secrets
cp .env.example .env   # fill in OPENAI_API_KEY, SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET

# 3. Add the knowledge-base DB (gitignored) at data/synthetic_startup.sqlite

# 4. Run the checks
make check

# 5. Ask a question from the CLI (no Slack needed)
python -m app.ask "what region is BlueHarbor Logistics in?"

# 6. Run the Slack bot (Events API webhook on :3000)
python -m app.slack_app
```

For Slack, expose `:3000` publicly (e.g. `cloudflared tunnel --url http://localhost:3000`)
and set the app's **Event Subscriptions → Request URL** to `https://<host>/slack/events`,
subscribing to `app_mention` and `message.im`.

## Run with Docker

```bash
docker build -t slack-qa-bot .

# Slack bot (default command — Events API webhook on :3000)
docker run --rm -p 3000:3000 --env-file .env -v "$PWD/data:/app/data" slack-qa-bot

# Or run the CLI in the same image (override the default command)
docker run --rm --env-file .env -v "$PWD/data:/app/data" slack-qa-bot \
  python -m app.ask "what region is BlueHarbor Logistics in?"
```

## Development

| Command | What it does |
|---|---|
| `make lint`  | `ruff check` + `ruff format --check` |
| `make type`  | `mypy app/` |
| `make test`  | `pytest -q` |
| `make check` | all of the above (the green/red gate) |
| `make eval`  | run the golden eval as a LangSmith experiment + regression gate |

## CI & branch protection

CI (`.github/workflows/ci.yml`) runs `make check` on every PR and push to `main`,
plus `make eval` as a regression gate on PRs. `main` is protected: required CI
pass, up-to-date branches, no direct pushes (feature branch → PR → squash-merge).

## Demo

A short screen recording of the bot answering questions in Slack:
[SlackBot_Demo (Google Drive)](https://drive.google.com/file/d/1SPtEwLTOezkrQqKu2jRfE2ryxRGK5r1U/view?usp=sharing)
