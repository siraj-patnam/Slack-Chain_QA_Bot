# Slack Q&A Bot

A Slack bot that answers natural-language questions grounded in a SQLite knowledge base
(customer calls, product details, implementation notes, internal comms, competitor
research). Built on **LangGraph** with a hybrid SQL + FTS5 retrieval layer.

> **Status:** under construction. This README is a skeleton; the full architecture
> write-up, eval results, and run instructions land in the final polish PR.
> The design rationale lives in **`DESIGN.md`** (hand-written).

## Quick start (preview)

```bash
# 1. Install deps (uses uv)
uv sync

# 2. Configure secrets
cp .env.example .env   # then fill in OPENAI_API_KEY, SLACK_* tokens

# 3. Add the knowledge-base DB (gitignored)
#    Download synthetic_startup.sqlite and place it at data/synthetic_startup.sqlite

# 4. Run the checks
make check
```

## Development

| Command | What it does |
|---|---|
| `make lint`  | `ruff check` + `ruff format --check` |
| `make type`  | `mypy app/` |
| `make test`  | `pytest -q` |
| `make check` | all of the above (the green/red gate) |

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

