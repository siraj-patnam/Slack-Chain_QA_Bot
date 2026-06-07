"""Deterministic scoring helpers (no LLM), kept pure so they are unit-testable."""

from __future__ import annotations

from collections.abc import Iterable

# Phrases that signal an honest "I don't know" rather than a fabricated answer.
_NOT_FOUND_MARKERS = (
    "couldn't find",
    "could not find",
    "couldn't locate",
    "could not locate",
    "unable to find",
    "don't have",
    "do not have",
    "not in the data",
    "no information",
    "no data",
    "isn't in the",
    "is not in the",
)


def check_substrings(answer: str, must_include: Iterable[str]) -> bool:
    """True if every required substring appears in the answer (case-insensitive)."""
    low = answer.lower()
    return all(sub.lower() in low for sub in must_include)


def is_not_found(answer: str) -> bool:
    """True if the answer is an honest refusal rather than a concrete claim."""
    low = answer.lower()
    return any(marker in low for marker in _NOT_FOUND_MARKERS)
