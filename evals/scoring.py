"""Deterministic scoring helper, kept pure so it is unit-testable.

Substring checks are used as a cheap, deterministic anchor in front of the LLM
judge (and as the whole score for exact structured lookups). Honest-refusal
cases are NOT matched against a hardcoded phrase list any more -- they are scored
by the judge against a refusal reference, which is robust to any phrasing and
removes a brittle keyword list from the harness.
"""

from __future__ import annotations

from collections.abc import Iterable


def check_substrings(answer: str, must_include: Iterable[str]) -> bool:
    """True if every required substring appears in the answer (case-insensitive)."""
    low = answer.lower()
    return all(sub.lower() in low for sub in must_include)
