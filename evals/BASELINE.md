# Eval baseline

The committed baseline that `make eval` gates against. Update it only on an
intended, reviewed improvement (`python -m evals.run --update`).

| Metric | Value |
|---|---|
| Agent | custom StateGraph (rewrite → agent ⇄ tools → ground), gpt-4o |
| Judge | openevals correctness (gpt-4o-mini) |
| Accuracy | **83.3%** (10 / 12) |
| Total tool calls | **68** |

## Improvement over the prebuilt agent

| Agent | Accuracy | Tool calls |
|---|---|---|
| Prebuilt `create_agent` | 58.3% (7/12) | 59 |
| Custom graph + grounding | **83.3% (10/12)** | 68 |

The lift comes from the grounding node: the prebuilt agent answered the
detail-heavy example queries (ex1–ex5) from search snippets and missed
specifics; the graph re-answers those from the full `content_text` of the
retrieved artifacts, recovering the exact dates, windows, metrics, and plan
steps — while structured answers (counts/lists) and honest refusals pass
through unchanged.

Set `USE_GRAPH=false` to fall back to the prebuilt agent.

Per-run traces and per-case feedback are in the LangSmith experiment whose URL
`make eval` prints.
