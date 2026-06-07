"""Run the golden eval as a LangSmith experiment and gate CI on the results.

Scoring:
- judge cases  -> openevals correctness judge (LLM-as-judge) gated by a
                  deterministic substring anchor.
- substring    -> every required substring present.
- not_found    -> the answer is an honest refusal.
Plus a tool-call budget check per case.

Runs are uploaded to LangSmith as an experiment (dataset + per-example traces +
feedback scores), and a baseline ratchet (``evals/baseline.json``) fails CI on
any accuracy or tool-call regression.

    python -m evals.run            # run + gate against the baseline
    python -m evals.run --update   # run + (re)write the baseline
    python -m evals.run --sync     # recreate the LangSmith dataset from cases.py
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client, evaluate
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

from app.agent import ask, build_default
from evals.dataset import ensure_dataset
from evals.scoring import check_substrings, is_not_found

BASELINE_PATH = Path(__file__).parent / "baseline.json"

# Absolute floor, set below the prebuilt agent's observed run-to-run variance
# (~0.50-0.67) so the gate isn't flaky; the baseline ratchet below is the real
# regression detector. The grounding graph is expected to raise both.
MIN_ACCURACY = 0.45
ACCURACY_TOLERANCE = 0.10
# Efficiency is gated on the AGGREGATE tool-call count vs baseline (stable),
# not per-case budgets (a single case's count swings run-to-run with the LLM).
# Per-case budgets are still recorded as LangSmith feedback for visibility.
TOOL_CALL_TOLERANCE = 0.35


def _make_target(agent):
    def target(inputs: dict) -> dict:
        outcome = ask(agent, inputs["question"], thread_id=f"eval-{uuid.uuid4().hex}")
        return {"answer": outcome.answer, "tool_calls": outcome.tool_call_count}

    return target


def _make_correctness_evaluator():
    judge = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
        model="openai:" + os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o-mini"),
    )

    def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        answer = outputs.get("answer", "")
        mode = reference_outputs["score_mode"]
        if mode == "substring":
            ok = check_substrings(answer, reference_outputs["must_include"])
        elif mode == "not_found":
            ok = is_not_found(answer)
        else:  # judge: deterministic anchor must hold, then the LLM judge
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


def _aggregate(results) -> tuple[float, int, list[str]]:
    total = correct = tool_total = 0
    over_budget: list[str] = []
    for row in results:
        total += 1
        scores = {r.key: (r.score or 0) for r in row["evaluation_results"]["results"]}
        correct += int(scores.get("correct", 0))
        tool_total += int((row["run"].outputs or {}).get("tool_calls", 0))
        if not scores.get("within_budget", 1):
            over_budget.append((row["example"].metadata or {}).get("name", "?"))
    accuracy = correct / total if total else 0.0
    return accuracy, tool_total, over_budget


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

    accuracy, total_tool_calls, over_budget = _aggregate(results)
    print(f"\naccuracy: {accuracy:.2%} | total tool calls: {total_tool_calls}")
    if over_budget:
        # Reported, not fatal — a single case's count is too noisy to hard-gate.
        print(f"note: over per-case budget (informational): {', '.join(over_budget)}")

    failed = False
    if accuracy < MIN_ACCURACY:
        print(f"FAIL: accuracy {accuracy:.2%} below floor {MIN_ACCURACY:.0%}")
        failed = True

    baseline_exists = BASELINE_PATH.is_file()
    if baseline_exists and not args.update:
        baseline = json.loads(BASELINE_PATH.read_text())
        if accuracy < baseline["accuracy"] - ACCURACY_TOLERANCE:
            print(f"FAIL: accuracy regressed ({accuracy:.2%} < {baseline['accuracy']:.2%})")
            failed = True
        if total_tool_calls > baseline["total_tool_calls"] * (1 + TOOL_CALL_TOLERANCE):
            print(
                f"FAIL: tool calls regressed ({total_tool_calls} > {baseline['total_tool_calls']})"
            )
            failed = True

    if failed:
        return 1

    if args.update or not baseline_exists:
        BASELINE_PATH.write_text(
            json.dumps({"accuracy": accuracy, "total_tool_calls": total_tool_calls}, indent=2)
            + "\n"
        )
        print(f"baseline written to {BASELINE_PATH.name}")

    print("eval PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
