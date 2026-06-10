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
| Custom graph (default) | 100% (16/16) | 100% (4/4) | 41 |
| Prebuilt `create_agent` (`USE_GRAPH=false`), same tools | 100% (16/16) | 100% (4/4) | 34 |

Numbers vary a few points run to run (a temperature-0 LLM agent is still not
fully deterministic); the floors absorb that. The graph row is from the run
that rewrote `baseline.json` after the ex5 agent-side fixes below (two
consecutive full runs scored 16/16). The aggregate tool-call count rose from
~31 to 41 — the intended price of the grader now sending the agent back to
retrieve when part of a question is unanswered, instead of accepting a
half-answer.

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

**Rescoring (first change).** The old gold hardcoded `must_include=("BlueHarbor",)`,
which auto-failed an equally-correct Pioneer answer *before the judge ran*. We
removed that anchor and rewrote the rubric to score the **reasoning**: a correct
answer names **either** BlueHarbor **or** Pioneer Freight and gives that account's
real milestone; it is still marked wrong for naming a different account, a wrong
competitor (e.g. Patchway), or no milestone. Verified offline against four live
ex5 answers — the two BlueHarbor and the Pioneer answers pass; the
NordFryst→Patchway miss fails.

### ex5, part 2 — the agent-side root causes (fixed)

After the rescore ex5 *still* failed every live run, for a different reason than
the one documented above: tracing the agent's actual tool-call **arguments**
(not just tool names) showed the failure mode had drifted. Four compounding
agent bugs, all general (none ex5-specific):

1. **Criterion substitution.** The agent matched the question's "defect /
   renewal" wording to the `pain_point` value *"renewal risk caused by noisy
   alerting"* and answered NordFryst — never checking the question's real
   discriminator (*which competitor is the cheap tactical one*) against the
   `competitors` table. Fix: the ranking rule in the prompt now says to resolve
   an entity-describing qualifier in that entity's OWN table (pricing words →
   `competitors.pricing_position`, role words → `segment`/`description`), then
   join through the scenario FKs — and to finish by reading the finalists'
   artifacts for any asked-for detail.
2. **FK-direction stumbles + a silent SQLite trap.** The agent guessed
   `scenarios.customer_id` (doesn't exist) three times, then wrote
   `customer_id IN (SELECT customer_id FROM scenarios WHERE ...)` — SQLite
   resolves the unknown column to the OUTER query, silently turning the filter
   into a no-op that returned ALL at-risk accounts as "facing NoiseGuard".
   Fixes: `run_sql` now appends a compact table(columns) catalog to
   no-such-column/table errors so one bad guess self-corrects in one retry; the
   prompt documents the FK direction and the correlated-subquery hazard.
3. **Evidence window dropped the decisive results.** `_tool_results` kept the
   FIRST ~9k chars of tool output, so when the budget bound, `generate` saw only
   the opening exploration dumps and claimed retrieved facts "weren't found".
   Fix: the window now keeps the most RECENT results (chronological order
   preserved) — retrieval converges toward the answer.
4. **The grader's refusal carve-out was too generous.** "I couldn't find the
   milestone" was accepted as complete with 9+ of 14 tool calls unspent and no
   query ever aimed at it. Fix: `grade` now sees the tool budget and only
   accepts a "couldn't find" as complete when the evidence shows retrieval for
   that specific part genuinely came up empty.

Also fixed while here: synthetic "budget reached" ToolMessages (which close out
unexecuted calls to keep saved history valid) no longer count as executed tool
calls in `ask()`/`stream_run` or the graph's budget — they had inflated the
per-case counts.

Validated: ex5 went 0/4 → 4/4 in live batches (three of four runs produce the
rubric's exact Pioneer milestone with citations), and two consecutive full-suite
runs scored 16/16 with held-out 4/4 — the generalization gate the prompt rules
must not regress.

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
