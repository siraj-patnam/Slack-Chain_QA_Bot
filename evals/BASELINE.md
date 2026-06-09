# Eval baseline

The committed baseline `make eval` gates against. Regenerate it only on an
intended, reviewed change: `python -m evals.run --sync --update`.

Harness: 16 cases (the 7 example queries + authored lookups + a held-out
generalization set + a refusal case), gpt-4o agent, gpt-4o-mini judge
pinned to **temperature 0**. Accuracy is gated on absolute floors; only the
aggregate tool-call count is ratcheted against `baseline.json`.

## Current numbers

| Config | Accuracy | Held-out | Tool calls |
|---|---|---|---|
| Custom graph (default) | 93.75% (15/16) | 100% (4/4) | 31 |
| Prebuilt `create_agent` (`USE_GRAPH=false`), same tools | 93.75% (15/16) | 100% (4/4) | 30 |

Numbers vary a few points run to run (a temperature-0 LLM agent is still not
fully deterministic); the floors absorb that.

## Where the improvement came from

The project started at **58.3% (7/12)** with the prebuilt agent. Most of the
accuracy gain came from the retrieval work:

- full `content_text` reads (cells were truncated to 300 chars, which hid answers
  that sit deep in a document);
- explore-first tools (`distinct_values`, `find_customer`) so the agent looks up
  real values instead of guessing a filter;
- foreign-key attribution in `search_text` and the grounding fetch.

Those tools are shared, so the prebuilt `create_agent` reaches the same ~94% as
the graph on this set: on accuracy alone, the two are within run-to-run noise of
each other.

That equivalence is specific to what this set measures. The 16 cases are
single-turn, so they don't exercise the graph's follow-up rewriting or multi-turn
memory, and at this size they can't resolve small structural differences. The
graph contributes the parts a real deployment needs — typed routing, an explicit
grounding + self-correction (`grade`) step, a tool-call budget, and follow-up
handling — at no accuracy cost on the cases we do measure.

## Remaining failure (1/16)

`ex5` ("which customer is most likely to defect to a cheaper tactical
competitor"). It is a multi-factor ranking question with a debatable gold answer
(BlueHarbor): the agent names a *different* account run to run (Pioneer Grid
Retail, NordFryst), none matching the gold. Part ambiguous gold (an account
running a live competitor PoC is a defensible answer), part the agent not
converging on the strongest signal. A ranking-judgment weakness, not a retrieval
or grounding miss.

## Methodology notes

- The judge is pinned to **temperature 0**: at the default it scored identical
  correct answers inconsistently (~1 in 6 false fails), which surfaced as phantom
  agent variance.
- `not_found` (refusal) cases are scored by the judge against a refusal
  reference, **not** a hardcoded keyword list.
- `baseline.json` records only `total_tool_calls`, the one low-variance metric.
  Comparing a single stochastic accuracy run to a stored one, with a noise-sized
  tolerance, would just fire on noise, so accuracy is gated on floors instead.
- `USE_GRAPH=false` falls back to the prebuilt agent.

Per-run traces and per-case feedback are in the LangSmith experiment whose URL
`make eval` prints.
