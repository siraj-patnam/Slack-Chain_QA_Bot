# Slack Q&A Bot

A Slack bot that answers natural-language questions grounded in a SQLite knowledge base
(customer calls, product details, implementation notes, internal comms, competitor
research). Built on **LangGraph** with a hybrid SQL + FTS5 retrieval layer.

## Quick start

```bash
# 1. Install deps (uses uv)
uv sync

# 2. Configure secrets
cp .env.example .env   # then fill in OPENAI_API_KEY, SLACK_* tokens

# 3. Add the knowledge-base DB (gitignored)
#    Download synthetic_startup.sqlite and place it at data/synthetic_startup.sqlite

# 4. Run the checks
make check

# 5. Run the bot locally (Slack Events API webhook on :3000)
python -m app.slack_app
```

### Run with Docker

```bash
docker build -t slack-qa-bot .
# Mount the data dir (knowledge base + checkpoints) and pass secrets via .env:
docker run --rm -p 3000:3000 --env-file .env -v "$PWD/data:/app/data" slack-qa-bot
```

## Development

| Command | What it does |
|---|---|
| `make lint`  | `ruff check` + `ruff format --check` |
| `make type`  | `mypy app/` |
| `make test`  | `pytest -q` |
| `make check` | all of the above (the green/red gate) |
| `make eval`  | run the golden eval as a LangSmith experiment + regression gate |

## Evaluation

`make eval` runs the golden set (the example queries + authored cases + an
honest-refusal case) as a **LangSmith experiment**: each case is scored with an
`openevals` correctness judge (prose) or substring/refusal checks (structured),
plus a per-case tool-call budget. Every run is traced to LangSmith, and a
committed baseline (`evals/baseline.json`, documented in `evals/BASELINE.md`)
fails the build on any accuracy or tool-call regression.

Current baseline (prebuilt agent, gpt-4o):

| Metric | Value |
|---|---|
| Accuracy | 58.3% (7/12) |
| Total tool calls | 59 |

Runs require `OPENAI_API_KEY` and `LANGSMITH_API_KEY`; the experiment URL is
printed at the end of each run for the traces and per-case feedback.

## Memory & self-correction

Multi-turn thread memory is checkpointed to SQLite (`SqliteSaver` at
`data/checkpoints.sqlite`, overridable via `CHECKPOINT_DB_PATH`), so a thread's
history survives a process restart. `PostgresSaver` is the drop-in production
swap — same interface. Tests keep an in-memory checkpointer, so they stay
hermetic; only the production entry point (`build_default`) persists.

Bounded SQL self-correction is deliberately *not* a dedicated node. `run_sql`
returns errors as observations rather than raising, so the agent fixes its own
query inside the ReAct loop; the tool-call budget bounds how many times it can
retry; and the `grade` node adds a higher-level groundedness/completeness retry.
A bespoke retry-then-fallback node would duplicate the agent loop and reintroduce
the brittle hand-routing the graph exists to avoid.

## Layout

```
app/      # bot, agent, tools, db, prompt
evals/    # eval harness
tests/    # unit tests
```

## CI & branch protection

CI (`.github/workflows/ci.yml`) runs `make check` on every PR to `main` and on
pushes to `main`. `main` is protected:

- Require the **CI / check** status check to pass before merging.
- Require branches to be up to date before merging.
- Disallow direct pushes to `main` (work on feature branches → PR → squash-merge).

CI also runs `make eval` as a regression gate (see `evals/BASELINE.md`).

