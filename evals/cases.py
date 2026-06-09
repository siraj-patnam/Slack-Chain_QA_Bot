"""The golden eval set: the example queries plus authored cases.

Each case carries a tool-call budget and a way to score the answer:
- ``judge``     : an LLM judge checks the answer against ``rubric`` (for prose).
- ``substring`` : every string in ``must_include`` must appear (for structured
                  / exact answers).
- ``not_found`` : the answer must be an honest "I couldn't find that".

``must_include`` is also used in ``judge`` mode as a cheap deterministic anchor.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    name: str
    question: str
    max_tool_calls: int
    score_mode: str = "judge"  # "judge" | "substring" | "not_found"
    must_include: tuple[str, ...] = ()
    rubric: str = ""


# The seven provided example queries.
EXAMPLE_CASES = [
    EvalCase(
        name="ex1_blueharbor_proof_plan",
        question=(
            "which customer's issue started after the 2026-02-20 taxonomy "
            "rollout, and what proof plan did we propose to get them comfortable "
            "with renewal?"
        ),
        max_tool_calls=8,
        must_include=("BlueHarbor",),
        rubric=(
            "Identifies BlueHarbor Logistics, and describes the proposed "
            "proof-of-fix plan: update index weighting, add a taxonomy mapping "
            "layer, and run an A/B test on the top saved searches with a target "
            "top-5 correct hit rate around 80%."
        ),
    ),
    EvalCase(
        name="ex2_verdant_bay_patch_rollback",
        question=(
            "for Verdant Bay, what's the approved live patch window, and exactly "
            "how do we roll back if the validation checks fail?"
        ),
        max_tool_calls=6,
        must_include=("Verdant Bay",),
        rubric=(
            "States the approved live patch window is 2026-03-24 from 02:00 to "
            "04:00 local time, and that on validation failure the rollback "
            "restores the prior ruleset (orchestrator rollback to the prior "
            "ruleset sha) and replays the invalidation hook."
        ),
    ),
    EvalCase(
        name="ex3_mapleharvest_transform_workshop",
        question=(
            "in the MapleHarvest Quebec pilot, what temporary field mappings are "
            "we planning in the router transform, and what is the March 23 "
            "workshop supposed to produce?"
        ),
        max_tool_calls=8,
        must_include=("transaction_id", "amount_cents"),
        rubric=(
            "Describes the temporary router transform mapping txn_id to "
            "transaction_id and total_amount to amount_cents with type coercion, "
            "and that the 2026-03-23 workshop is to agree the canonical schema / "
            "alias mappings and produce a signed schema document."
        ),
    ),
    EvalCase(
        name="ex4_aureum_scim_fix",
        question=(
            "what SCIM fields were conflicting at Aureum, and what fast fix did "
            "Jin propose so we don't have to wait on Okta change control?"
        ),
        max_tool_calls=8,
        must_include=("department", "businessUnit"),
        rubric=(
            "Identifies the conflicting SCIM fields department vs businessUnit at "
            "Aureum, and Jin's fast fix as a hot-reloadable Signal Ingest "
            "preprocessing rule that normalizes those attributes into one "
            "canonical field (plus SCIM tracing), avoiding Okta change control."
        ),
    ),
    EvalCase(
        name="ex5_defection_risk_milestone",
        question=(
            "which customer looks most likely to defect to a cheaper tactical "
            "competitor if we miss the next promised milestone, and what exactly "
            "is that milestone?"
        ),
        max_tool_calls=14,
        # NO substring anchor here (unlike the other example cases). This is a
        # multi-factor RANKING-JUDGMENT question with more than one defensible
        # answer, so hardcoding a single customer name (the old
        # must_include=("BlueHarbor",)) auto-failed an equally-correct answer
        # before the judge ever ran. The data backs this up: the "cheaper tactical
        # competitor" is NoiseGuard, and exactly TWO accounts are BOTH 'at risk'
        # AND facing NoiseGuard — BlueHarbor Logistics (the original gold) and
        # Pioneer Freight Solutions, which is actually running a LIVE tactical
        # NoiseGuard PoC. Live agent runs name each of them (plus the occasional
        # genuine miss). So we drop the anchor and let the judge score the
        # REASONING: did it pick an at-risk NoiseGuard account and give that
        # account's real milestone — not whether it echoed one memorized name.
        rubric=(
            "This is a ranking-judgment question: which account is most likely to "
            "defect to the cheaper, tactical competitor (NoiseGuard — the low-cost "
            "alert/dedupe option) if the next promised milestone slips, and what "
            "that milestone is. TWO answers are acceptable, because two 'at risk' "
            "accounts both face NoiseGuard: BlueHarbor Logistics and Pioneer "
            "Freight Solutions. Mark the answer CORRECT if it (a) names EITHER "
            "BlueHarbor Logistics OR Pioneer Freight Solutions as the account at "
            "risk of defecting, AND (b) gives that account's concrete next "
            "milestone — the right timeframe and what must be proven, grounded with "
            "a citation. For BlueHarbor the milestone is the proof-of-fix that "
            "search relevance is measurably improved within a short window of about "
            "7-10 business days (its success target is a top-5 correct hit rate of "
            "~80%); an answer that identifies this proof-of-fix and its ~10-"
            "business-day window is correct whether or not it quotes the exact 80% "
            "figure. For Pioneer Freight the milestone is the search-relevance "
            "remediation gate (schema verification around 2026-03-22 within a "
            "~6-week recovery, acceptance e.g. precision@10 >= 0.8). Naming "
            "NoiseGuard explicitly STRENGTHENS the answer but is NOT required — "
            "both accounts' competitor is "
            "NoiseGuard, so naming the account already implies it; likewise the "
            "exact metric threshold is a bonus, not required. Mark the answer "
            "INCORRECT if it names any OTHER account (e.g. NordFryst, HelioFab, "
            "Pioneer Grid Retail), if it attributes the threat to a competitor that "
            "is not a cheap tactical dedupe option (e.g. Patchway, an enterprise "
            "orchestration tool), or if it gives no concrete milestone."
        ),
    ),
    EvalCase(
        name="ex6_nawest_taxonomy_vs_duplicate",
        question=(
            "among the North America West Event Nexus accounts, which ones are "
            "really dealing with taxonomy/search semantics problems versus "
            "duplicate-action problems?"
        ),
        max_tool_calls=28,
        # One member from EACH group, so a one-sided answer (the common failure —
        # listing only the taxonomy group) fails the anchor instead of sneaking
        # past on taxonomy-only tokens.
        must_include=("BlueHarbor", "MedLogix"),
        rubric=(
            "Splits the accounts into a taxonomy/search-semantics group (Arcadia "
            "Cloudworks, BlueHarbor Logistics, CedarWind Renewables, HelioFab "
            "Systems, Pacific Health Network, Pioneer Freight Solutions) and a "
            "duplicate-action group (Helix Assemblies, LedgerBright Analytics, "
            "LedgerPeak Software, MedLogix Distribution, Peregrine Logistics "
            "Group, Pioneer Grid Retail). Getting most accounts in the right "
            "group counts as correct."
        ),
    ),
    EvalCase(
        name="ex7_canada_approval_bypass",
        question=(
            "do we have a recurring Canada approval-bypass pattern across "
            "accounts, or is MapleBridge basically a one-off? Give me the "
            "customer names and the shared failure pattern in plain English."
        ),
        max_tool_calls=8,
        must_include=("MapleBridge",),
        rubric=(
            "Concludes it is a recurring Canada approval-bypass pattern (not a "
            "one-off), names several affected accounts (e.g. MapleBridge "
            "Insurance, City of Verdant Bay, Maple Regional Transit Authority, "
            "MapleBay Marketplace, MapleFork Franchise Systems, MaplePath Career "
            "Institute, MapleWest Bank), and explains that after migration the "
            "global/country-default rules win over province/city/Canada-specific "
            "approval rules (bad precedence, stale caches, alias mismatches), so "
            "approvals get bypassed or misrouted."
        ),
    ),
]

# Authored cases: easy structured lookups plus an honest-refusal case.
AUTHORED_CASES = [
    EvalCase(
        name="auth_count_customer_calls",
        question="How many customer call artifacts are in the knowledge base?",
        max_tool_calls=4,
        score_mode="substring",
        must_include=("50",),
    ),
    EvalCase(
        name="auth_blueharbor_region",
        question="What region is BlueHarbor Logistics in?",
        max_tool_calls=6,
        score_mode="substring",
        must_include=("North America West",),
    ),
    EvalCase(
        name="auth_products",
        question="What products does Northstar Signal offer?",
        max_tool_calls=4,
        score_mode="substring",
        must_include=("Signal Ingest", "Event Nexus", "Orchestrator", "Signal Insights"),
    ),
    EvalCase(
        name="auth_cheap_tactical_competitor",
        question="Which competitor is positioned as a low-cost, tactical dedupe option?",
        max_tool_calls=6,
        score_mode="substring",
        must_include=("NoiseGuard",),
    ),
    EvalCase(
        name="auth_not_in_data",
        question="What is Northstar Signal's stock price?",
        max_tool_calls=6,
        score_mode="not_found",
    ),
]

# HELD-OUT cases: a generalization gate. These exercise the SAME general methods
# the prompt teaches (split a population by its classifying column, named-entity
# lookup, structured count, honest refusal) but over data the prompt was NEVER
# tuned against — different regions, different customers, different categories.
#
# DISCIPLINE: do NOT tune the prompt against these. If an EXAMPLE_CASE passes but
# its held-out twin fails, the prompt has memorized the example rather than
# learning the method. They are scored alongside the rest so that overfitting to
# EXAMPLE_CASES shows up as a held-out regression.
HELDOUT_CASES = [
    EvalCase(
        name="held_nordics_split",
        question=(
            "Among our Nordics accounts, which ones are dealing with renewal risk "
            "from noisy alerting versus executive-dashboard reporting latency?"
        ),
        max_tool_calls=8,
        must_include=("NordChemica", "FrostGrid"),
        rubric=(
            "Splits the Nordics accounts into a 'renewal risk from noisy alerting' "
            "group (NordChemica AB, NordMed Distribution AB, NordGrid Services AB, "
            "Aurora Dataworks AB, Fyrkrona Renewables AB, NordFryst AB, NordicChem "
            "AB) and an 'executive-dashboard reporting latency' group (FrostGrid "
            "Energi AB, NorrLog Freight AB, Svenska PolyChem AB, Nordic MedSupply "
            "AB, SentinelOps AB, Nordiska Grid Services AB). Getting most accounts "
            "into the right group counts as correct."
        ),
    ),
    EvalCase(
        name="held_nordics_count",
        question="How many customers are in the Nordics region?",
        max_tool_calls=4,
        score_mode="substring",
        must_include=("13",),
    ),
    EvalCase(
        name="held_nordchemica_region",
        question="What region is NordChemica in?",
        max_tool_calls=6,
        score_mode="substring",
        must_include=("Nordics",),
    ),
    EvalCase(
        name="held_not_in_data",
        question="How many employees does the competitor NoiseGuard have?",
        max_tool_calls=6,
        score_mode="not_found",
    ),
]

CASES: list[EvalCase] = [*EXAMPLE_CASES, *AUTHORED_CASES, *HELDOUT_CASES]
