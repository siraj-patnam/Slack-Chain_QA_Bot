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
fully deterministic); the floors absorb that. These figures predate the `ex5`
rescore below (the old run's single failure); regenerate with `--sync --update`.

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

## ex5 — a deliberately ambiguous ranking case (rescored)

`ex5` ("which customer is most likely to defect to a cheaper tactical
competitor") was the long-standing 1/16 failure. Investigation showed it is not a
retrieval or grounding bug but a **multi-factor ranking question with more than
one defensible answer**, made unwinnable by a hardcoded gold:

- The "cheaper tactical competitor" is **NoiseGuard** (low-cost alert/dedupe). Of
  the six accounts that face NoiseGuard, exactly **two are `account_health = 'at
  risk'`**: **BlueHarbor Logistics** (the original gold) and **Pioneer Freight
  Solutions** — which is actually running a **live tactical NoiseGuard PoC** and
  says it will "extend NoiseGuard and reduce Northstar scope" if the 6-week
  remediation slips. Pioneer is at least as defensible as BlueHarbor.
- The one artifact that tips the call to BlueHarbor (its renewal call, where the
  customer muses about taking NoiseGuard as a cheap stopgap) contains **none** of
  the question's discriminating words ("defect" appears 0× in the whole corpus;
  the call says "cheap", the question says "cheaper" — different terms under
  FTS5's stemless match). Keyword search on the question buries that call at rank
  ~33–109 behind six near-identical "Competitor report: NoiseGuard" templates, so
  the agent sometimes answers from the wrong artifact (e.g. NordFryst's "Renewal
  Risk" doc → defect to *Patchway*, the wrong, enterprise-priced competitor).

Live agent runs bear this out: across sampled runs it names BlueHarbor, Pioneer
Freight, and the occasional genuine miss (NordFryst/Patchway).

**Rescoring (this change).** The old gold hardcoded `must_include=("BlueHarbor",)`,
which auto-failed an equally-correct Pioneer answer *before the judge ran*. We
removed that anchor and rewrote the rubric to score the **reasoning**: a correct
answer names **either** BlueHarbor **or** Pioneer Freight and gives that account's
real milestone; it is still marked wrong for naming a different account, a wrong
competitor (e.g. Patchway), or no milestone. Verified offline against four live
ex5 answers — the two BlueHarbor and the Pioneer answers pass; the
NordFryst→Patchway miss fails.

> **Re-sync required.** The LangSmith dataset caches the reference, so this rubric
> change only takes effect after `python -m evals.run --sync --update`, which also
> re-baselines `total_tool_calls`. The "Current numbers" above predate the rescore.

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
