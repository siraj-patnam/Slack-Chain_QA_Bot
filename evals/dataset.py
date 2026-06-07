"""Sync the golden eval cases to a LangSmith dataset.

The cases in ``cases.py`` are the source of truth; this uploads them as a
LangSmith dataset so runs are scored as experiments with a shared, versioned
reference set (and an experiment-comparison UI for tracking improvement over
time).
"""

from __future__ import annotations

import os

from langsmith import Client

from evals.cases import CASES, EvalCase

DATASET_NAME = os.environ.get("LANGSMITH_DATASET", "slack-qa-bot-eval")


def _reference(case: EvalCase) -> str:
    if case.score_mode == "judge":
        return case.rubric
    if case.score_mode == "substring":
        return "Answer must include: " + ", ".join(case.must_include)
    return "An honest 'I couldn't find that in the data' (no fabrication)."


def _example_payload(case: EvalCase) -> dict:
    """The reference-output payload an evaluator needs to score this case."""
    return {
        "reference": _reference(case),
        "score_mode": case.score_mode,
        "must_include": list(case.must_include),
        "max_tool_calls": case.max_tool_calls,
    }


def ensure_dataset(client: Client, *, recreate: bool = False) -> str:
    """Create the dataset from CASES if missing (or recreate it). Return its name."""
    exists = client.has_dataset(dataset_name=DATASET_NAME)
    if exists and recreate:
        client.delete_dataset(dataset_name=DATASET_NAME)
        exists = False
    if not exists:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Slack Q&A bot golden eval: example queries + authored cases.",
        )
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[{"question": c.question} for c in CASES],
            outputs=[_example_payload(c) for c in CASES],
            metadata=[{"name": c.name} for c in CASES],
        )
    return DATASET_NAME
