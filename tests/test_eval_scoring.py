"""Tests for the deterministic eval scoring helpers."""

from evals.cases import CASES
from evals.scoring import check_substrings


def test_check_substrings_all_present() -> None:
    assert check_substrings(
        "BlueHarbor Logistics is in North America West", ("blueharbor", "north america west")
    )


def test_check_substrings_missing_one() -> None:
    assert not check_substrings("Only BlueHarbor here", ("blueharbor", "verdant bay"))


def test_cases_are_well_formed() -> None:
    names = [c.name for c in CASES]
    assert len(names) == len(set(names)), "case names must be unique"
    assert len(CASES) >= 10
    for c in CASES:
        assert c.score_mode in {"judge", "substring", "not_found"}
        assert c.max_tool_calls > 0
        if c.score_mode == "substring":
            assert c.must_include, f"{c.name}: substring case needs must_include"
        if c.score_mode == "judge":
            assert c.rubric, f"{c.name}: judge case needs a rubric"
