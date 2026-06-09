"""Run the golden eval as a LangSmith experiment and gate CI on the results.

Scoring:
- judge cases  -> openevals correctness judge (LLM-as-judge) gated by a
                  deterministic substring anchor.
- substring    -> every required substring present.
- not_found    -> the answer is an honest refusal.
Plus a tool-call budget check per case.

Runs are uploaded to LangSmith as an experiment (dataset + per-example traces +
feedback scores). Accuracy (overall and held-out) is gated on absolute floors;
only the aggregate tool-call count is ratcheted against ``evals/baseline.json``
(it is the one low-variance metric on a stochastic eval).

    python -m evals.run            # run + gate against the baseline
    python -m evals.run --update   # run + (re)write the baseline
    python -m evals.run --sync     # recreate the LangSmith dataset from cases.py
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import Client, evaluate
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

from app.agent import ask, build_default
from evals.dataset import ensure_dataset
from evals.scoring import check_substrings

BASELINE_PATH = Path(__file__).parent / "baseline.json"

# Accuracy is gated on an absolute FLOOR, not against the baseline. The eval is
# stochastic (LLM agent + LLM judge) and individual cases flip pass/fail run to
# run, so comparing one run's accuracy to a single stored run — with a tolerance
# the size of that per-case noise — would fire on noise, not real regressions. A
# floor ("must be >= X") is noise-robust. Fixed on principle, not tuned to pass.
MIN_ACCURACY = 0.70
# The HELD-OUT subset (cases named "held_*") is the overfitting tripwire: it
# exercises the same general methods as the example set but over data the prompt
# was never tuned against. We gate it on its OWN floor so an overfit that tanks
# held-out accuracy can't be masked by gains on the tuned cases. (On only a few
# cases a baseline-delta check would be pure noise — a floor is the right tool.)
HELDOUT_PREFIX = "held_"
MIN_HELDOUT_ACCURACY = 0.70
# Efficiency IS ratcheted against the baseline: the AGGREGATE tool-call count is
# an all-cases sum, low-variance run-to-run, and has no natural absolute floor, so
# a real regression stands out. Per-case budgets are recorded as LangSmith
# feedback for visibility, not hard-gated.
TOOL_CALL_TOLERANCE = 0.30


def _make_target(agent):
    def target(inputs: dict) -> dict:
        outcome = ask(agent, inputs["question"], thread_id=f"eval-{uuid.uuid4().hex}")
        return {"answer": outcome.answer, "tool_calls": outcome.tool_call_count}

    return target


def _make_correctness_evaluator():
    # Pin the judge to temperature 0. At the default temperature gpt-4o-mini scores
    # the SAME correct answer inconsistently (~1 in 6 false fails, measured), which
    # surfaces as phantom run-to-run accuracy swings unrelated to the agent. A
    # deterministic judge removes that noise source.
    judge = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
        judge=ChatOpenAI(model=os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o-mini"), temperature=0),
    )

    def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        answer = outputs.get("answer", "")
        mode = reference_outputs["score_mode"]
        if mode == "substring":
            ok = check_substrings(answer, reference_outputs["must_include"])
        else:
            # Both prose ("judge") and honest-refusal ("not_found") cases are scored
            # by the LLM judge against the case reference. A deterministic substring
            # anchor (the must_include list, empty for refusal cases) gates it first,
            # so the judge can't pass an answer that's missing a required fact.
            anchor = check_substrings(answer, reference_outputs.get("must_include") or [])
            verdict = judge(
                inputs=inputs["question"],
                outputs=answer,
                reference_outputs=reference_outputs["reference"],
            )
            ok = anchor and bool(verdict["score"])
        return {"key": "correct", "score": 1 if ok else 0}

    return correctness


def within_budget(outputs: dict, reference_outputs: dict) -> dict:
    ok = outputs.get("tool_calls", 0) <= reference_outputs["max_tool_calls"]
    return {"key": "within_budget", "score": 1 if ok else 0}


@dataclass
class EvalSummary:
    accuracy: float
    total_tool_calls: int
    over_budget: list[str]
    heldout_accuracy: float
    heldout_total: int


def _aggregate(results) -> EvalSummary:
    total = correct = tool_total = 0
    held_total = held_correct = 0
    over_budget: list[str] = []
    for row in results:
        total += 1
        scores = {r.key: (r.score or 0) for r in row["evaluation_results"]["results"]}
        is_correct = int(scores.get("correct", 0))
        correct += is_correct
        tool_total += int((row["run"].outputs or {}).get("tool_calls", 0))
        name = (row["example"].metadata or {}).get("name", "?")
        if name.startswith(HELDOUT_PREFIX):
            held_total += 1
            held_correct += is_correct
        if not scores.get("within_budget", 1):
            over_budget.append(name)
    accuracy = correct / total if total else 0.0
    # No held-out cases => treat as a pass (1.0) rather than dividing by zero.
    heldout_accuracy = held_correct / held_total if held_total else 1.0
    return EvalSummary(accuracy, tool_total, over_budget, heldout_accuracy, held_total)


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the agent eval as a LangSmith experiment.")
    parser.add_argument("--update", action="store_true", help="(re)write the baseline")
    parser.add_argument("--sync", action="store_true", help="recreate the LangSmith dataset")
    args = parser.parse_args()

    if not os.environ.get("LANGSMITH_API_KEY"):
        print("ERROR: LANGSMITH_API_KEY is not set (needed for the LangSmith eval).")
        return 2

    # Always trace eval runs to LangSmith, regardless of the app's global setting.
    os.environ["LANGSMITH_TRACING"] = "true"

    client = Client()
    dataset_name = ensure_dataset(client, recreate=args.sync)

    agent = build_default()
    results = evaluate(
        _make_target(agent),
        data=dataset_name,
        evaluators=[_make_correctness_evaluator(), within_budget],
        experiment_prefix="slack-qa-agent",
        max_concurrency=4,
    )

    summary = _aggregate(results)
    accuracy = summary.accuracy
    total_tool_calls = summary.total_tool_calls
    heldout_accuracy = summary.heldout_accuracy
    print(
        f"\naccuracy: {accuracy:.2%} | "
        f"held-out: {heldout_accuracy:.2%} ({summary.heldout_total} cases) | "
        f"total tool calls: {total_tool_calls}"
    )
    if summary.over_budget:
        # Reported, not fatal — a single case's count is too noisy to hard-gate.
        print(f"note: over per-case budget (informational): {', '.join(summary.over_budget)}")

    failed = False
    if accuracy < MIN_ACCURACY:
        print(f"FAIL: accuracy {accuracy:.2%} below floor {MIN_ACCURACY:.0%}")
        failed = True
    if heldout_accuracy < MIN_HELDOUT_ACCURACY:
        print(
            f"FAIL: held-out accuracy {heldout_accuracy:.2%} below floor {MIN_HELDOUT_ACCURACY:.0%}"
        )
        failed = True

    baseline_exists = BASELINE_PATH.is_file()
    if baseline_exists and not args.update:
        baseline = json.loads(BASELINE_PATH.read_text())
        # Only the tool-call count is ratcheted against the baseline (see the
        # constants above for why accuracy is floored instead of compared).
        if total_tool_calls > baseline["total_tool_calls"] * (1 + TOOL_CALL_TOLERANCE):
            print(
                f"FAIL: tool calls regressed ({total_tool_calls} > {baseline['total_tool_calls']})"
            )
            failed = True

    if failed:
        return 1

    if args.update or not baseline_exists:
        # Only the tool-call count is ratcheted, so it's all the baseline holds;
        # accuracy is gated on absolute floors and needs nothing stored.
        BASELINE_PATH.write_text(
            json.dumps({"total_tool_calls": total_tool_calls}, indent=2) + "\n"
        )
        print(f"baseline written to {BASELINE_PATH.name}")

    print("eval PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
