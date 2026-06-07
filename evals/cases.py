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
        must_include=("BlueHarbor",),
        rubric=(
            "Names BlueHarbor Logistics as most likely to defect to a cheaper "
            "tactical competitor (NoiseGuard), and describes the next milestone "
            "as the 7-10 business day proof-of-fix for search relevance with a "
            "top-5 correct hit rate of at least 80%."
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
        must_include=("BlueHarbor", "taxonomy"),
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

CASES: list[EvalCase] = [*EXAMPLE_CASES, *AUTHORED_CASES]
