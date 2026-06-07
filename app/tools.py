"""Retrieval tools the agent is allowed to call.

Two sharp tools, mirroring the hybrid retrieval design:

* ``run_sql``    — validated, read-only SELECT for counts/filters/joins.
* ``search_text`` — FTS5 keyword search over the long-form artifact corpus.

Both open their own read-only connection and enforce hard caps so a single
call can never run forever or flood the model's context window.
"""

from __future__ import annotations

import re
import sqlite3
import time

from langchain_core.tools import tool

from app.db import assert_safe_select, readonly_connection

# Resource caps — a valid query still can't return the whole DB.
MAX_ROWS = 100  # rows returned to the model
MAX_CELL_CHARS = 300  # per-cell truncation so one huge transcript can't dominate
MAX_RESULT_CHARS = 12_000  # overall byte cap on the formatted result
QUERY_TIMEOUT_S = 5.0  # wall-clock cap per query

DEFAULT_SEARCH_K = 5
MAX_SEARCH_K = 20


def _truncate(value: object) -> str:
    text = "" if value is None else str(value)
    if len(text) > MAX_CELL_CHARS:
        return text[:MAX_CELL_CHARS] + "…"
    return text


def _format_rows(columns: list[str], rows: list[sqlite3.Row], truncated: bool) -> str:
    if not rows:
        return "No rows."
    lines = [" | ".join(columns)]
    for row in rows:
        lines.append(" | ".join(_truncate(row[c]) for c in columns))
    if truncated:
        lines.append(f"… (truncated to first {MAX_ROWS} rows)")
    out = "\n".join(lines)
    if len(out) > MAX_RESULT_CHARS:
        out = out[:MAX_RESULT_CHARS] + "\n… (result truncated; narrow your query)"
    return out


def _run_sql(query: str) -> str:
    """Core logic for run_sql; returns rows or an ``Error: ...`` string.

    Errors are returned as text (not raised) so the agent can read them as an
    observation and self-correct rather than crashing the run.
    """
    try:
        assert_safe_select(query)
    except ValueError as err:
        return f"Error: {err}"

    deadline = time.monotonic() + QUERY_TIMEOUT_S
    try:
        with readonly_connection() as conn:
            conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
            cursor = conn.execute(query)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(MAX_ROWS + 1)
            truncated = len(rows) > MAX_ROWS
            rows = rows[:MAX_ROWS]
    except sqlite3.OperationalError as err:
        if "interrupted" in str(err).lower():
            return f"Error: query exceeded the {QUERY_TIMEOUT_S:g}s time limit."
        return f"Error: {err}"
    except sqlite3.Error as err:
        return f"Error: {err}"

    return _format_rows(columns, rows, truncated)


@tool
def run_sql(query: str) -> str:
    """Run a single READ-ONLY SQL SELECT against the company database.

    Use for structured questions: counts, filters, joins, "how many / which /
    when". Only one SELECT is allowed (CTEs are fine); writes, PRAGMA, ATTACH
    and multiple statements are rejected. Results are capped to 100 rows. Use
    json_extract(metadata_json, '$.key') to read fields inside metadata_json.
    """
    return _run_sql(query)


# FTS5 query column positions (see artifacts_fts schema): 1=title, 2=summary,
# 3=content_text. We snippet the content column.
_FTS_CONTENT_COL = 3

_SEARCH_SQL = """
SELECT a.artifact_id AS artifact_id,
       a.artifact_type AS artifact_type,
       a.title AS title,
       a.created_at AS created_at,
       snippet(artifacts_fts, ?, '[', ']', ' … ', 12) AS snippet
FROM artifacts_fts
JOIN artifacts a ON a.artifact_id = artifacts_fts.artifact_id
WHERE artifacts_fts MATCH ?
ORDER BY bm25(artifacts_fts)
LIMIT ?
"""


def _fts_match_query(text: str) -> str | None:
    """Turn free text into a safe FTS5 MATCH string.

    Extract alphanumeric tokens and OR them as quoted terms. Quoting each token
    neutralizes FTS5 operators in user/agent text (so a stray ``-`` or ``"``
    can't raise a syntax error), and bm25 ranking still surfaces rows that match
    more of the terms first.
    """
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


def _search_text(query: str, k: int = DEFAULT_SEARCH_K) -> str:
    match = _fts_match_query(query)
    if match is None:
        return "Error: no searchable terms in query."
    k = max(1, min(k, MAX_SEARCH_K))

    deadline = time.monotonic() + QUERY_TIMEOUT_S
    try:
        with readonly_connection() as conn:
            conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
            rows = conn.execute(_SEARCH_SQL, (_FTS_CONTENT_COL, match, k)).fetchall()
    except sqlite3.Error as err:
        return f"Error: {err}"

    if not rows:
        return "No matching artifacts."

    blocks = []
    for row in rows:
        blocks.append(
            f"[{row['artifact_id']}] ({row['artifact_type']}) {row['title']}\n"
            f"  {row['created_at']}\n"
            f"  {_truncate(row['snippet'])}"
        )
    out = "\n\n".join(blocks)
    if len(out) > MAX_RESULT_CHARS:
        out = out[:MAX_RESULT_CHARS] + "\n… (results truncated)"
    return out


@tool
def search_text(query: str, k: int = DEFAULT_SEARCH_K) -> str:
    """Keyword full-text search over the long-form artifact corpus (FTS5).

    Use for unstructured questions — "what did X say about Y", finding the
    transcript/ticket/memo that discusses a topic. Returns the top-k artifacts
    (id, type, title, and a matched snippet), ranked by relevance. Cite the
    returned artifact ids in your answer. For counts/filters/joins use run_sql.
    """
    return _search_text(query, k)
