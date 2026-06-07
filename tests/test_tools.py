"""Tests for the retrieval tools against a seeded fixture DB."""

from __future__ import annotations

import pytest

from app.tools import _distinct_values, _find_customer, _run_sql, _search_text


@pytest.mark.usefixtures("seeded_db")
def test_run_sql_navigates_customer_to_artifacts() -> None:
    # The text-to-SQL navigation pattern: resolve a customer, join to its artifacts.
    out = _run_sql(
        "SELECT artifact_id FROM artifacts WHERE customer_id = "
        "(SELECT customer_id FROM customers WHERE name LIKE '%Verdant Bay%')"
    )
    assert "art_0000000000a2" in out


@pytest.mark.usefixtures("seeded_db")
def test_search_text_finds_taxonomy_renewal_artifact() -> None:
    out = _search_text("taxonomy rollout proof-of-fix renewal", k=3)
    assert "art_0000000000a1" in out


@pytest.mark.usefixtures("seeded_db")
def test_search_text_finds_live_patch_window_artifact() -> None:
    out = _search_text("live patch window rollback validation", k=3)
    assert "art_0000000000a2" in out


@pytest.mark.usefixtures("seeded_db")
def test_search_text_no_match_is_honest() -> None:
    out = _search_text("zzz_nonexistent_term_qqq", k=3)
    assert out == "No matching artifacts."


@pytest.mark.usefixtures("seeded_db")
def test_run_sql_counts_rows() -> None:
    out = _run_sql("SELECT COUNT(*) AS n FROM artifacts")
    assert "4" in out


@pytest.mark.usefixtures("seeded_db")
def test_run_sql_filters_by_type() -> None:
    out = _run_sql("SELECT artifact_id FROM artifacts WHERE artifact_type = 'support_ticket'")
    assert "art_0000000000a4" in out
    assert "art_0000000000a1" not in out


@pytest.mark.usefixtures("seeded_db")
def test_run_sql_rejects_unsafe_via_tool() -> None:
    out = _run_sql("DROP TABLE artifacts")
    assert out.startswith("Error:")


@pytest.mark.usefixtures("seeded_db")
def test_distinct_values_lists_real_values_with_counts() -> None:
    # The explore-first step: see the actual stored values before filtering.
    out = _distinct_values("customers", "region")
    assert "North America West" in out
    assert "Canada (2)" in out  # two seeded customers share this region
    assert "ANZ" in out


@pytest.mark.usefixtures("seeded_db")
def test_distinct_values_rejects_unknown_table() -> None:
    out = _distinct_values("secrets", "region")
    assert out.startswith("Error: unknown table")


@pytest.mark.usefixtures("seeded_db")
def test_distinct_values_rejects_unknown_column() -> None:
    # Guards against SQL injection via an unvalidated column identifier.
    out = _distinct_values("customers", "region; DROP TABLE customers")
    assert out.startswith("Error: unknown column")


@pytest.mark.usefixtures("seeded_db")
def test_find_customer_resolves_spacing_variant() -> None:
    # "blue harbor" (with a space) must resolve to "BlueHarbor Logistics" — a raw
    # LIKE '%blue harbor%' would return nothing.
    out = _find_customer("blue harbor")
    first_line = out.splitlines()[1]  # header, then best candidate
    assert "BlueHarbor Logistics" in first_line


@pytest.mark.usefixtures("seeded_db")
def test_find_customer_tolerates_typo() -> None:
    out = _find_customer("verdent bay")  # typo: verdent -> Verdant
    assert "City of Verdant Bay" in out.splitlines()[1]


@pytest.mark.usefixtures("seeded_db")
def test_find_customer_always_returns_candidates() -> None:
    # Never a silent "no rows": even a poor term yields ranked candidates.
    out = _find_customer("zzz nonexistent")
    assert "Closest customers" in out
    assert "match" in out
