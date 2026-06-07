"""Tests for the retrieval tools against a seeded fixture DB."""

from __future__ import annotations

import pytest

from app.tools import _run_sql, _search_text


@pytest.mark.usefixtures("seeded_db")
def test_search_text_finds_taxonomy_renewal_artifact() -> None:
    out = _search_text("taxonomy rollout proof-of-fix renewal", k=3)
    assert "art_001" in out


@pytest.mark.usefixtures("seeded_db")
def test_search_text_finds_live_patch_window_artifact() -> None:
    out = _search_text("live patch window rollback validation", k=3)
    assert "art_002" in out


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
    assert "art_004" in out
    assert "art_001" not in out


@pytest.mark.usefixtures("seeded_db")
def test_run_sql_rejects_unsafe_via_tool() -> None:
    out = _run_sql("DROP TABLE artifacts")
    assert out.startswith("Error:")
