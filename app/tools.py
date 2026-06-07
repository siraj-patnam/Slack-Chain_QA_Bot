"""Retrieval tools the agent is allowed to call.

* ``run_sql``     — the primary tool: a validated read-only SELECT. The KB is
  fully relational, so the agent navigates it with SQL joins (customer/scenario
  -> artifacts) and reads the full content_text.
* ``search_text`` — fallback FTS5 keyword search for entity-less topic questions.

Each opens its own read-only connection and enforces hard caps so a single
call can never run forever or flood the model's context window.
"""

from __future__ import annotations

import difflib
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

# Long-form columns are NOT capped to MAX_CELL_CHARS: run_sql is the primary way
# the agent reads a full document, so truncating content_text to 300 chars would
# hide any detail that lives past the opening lines (the answer to most "exact
# command / window / steps" questions). Only the overall MAX_RESULT_CHARS cap
# applies to them, which still bounds a multi-row dump.
FULL_TEXT_COLUMNS = frozenset({"content_text"})

DEFAULT_SEARCH_K = 5
MAX_SEARCH_K = 20


def _truncate(value: object, limit: int = MAX_CELL_CHARS) -> str:
    text = "" if value is None else str(value)
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def _cell(column: str, value: object) -> str:
    limit = MAX_RESULT_CHARS if column in FULL_TEXT_COLUMNS else MAX_CELL_CHARS
    return _truncate(value, limit)


def _format_rows(columns: list[str], rows: list[sqlite3.Row], truncated: bool) -> str:
    if not rows:
        return "No rows."
    lines = [" | ".join(columns)]
    for row in rows:
        lines.append(" | ".join(_cell(c, row[c]) for c in columns))
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


# Chars of content_text returned per search hit — enough to ground an answer
# without a second fetch, capped so a few hits don't flood the context.
SEARCH_CONTENT_CHARS = 1200

# The owning customer is resolved via the FK and shown with each hit so the agent
# attributes facts to the artifact's customer — NOT to other names that happen to
# appear in the prose (which is how a snippet read produces a fabricated entity).
_SEARCH_SQL = """
SELECT a.artifact_id AS artifact_id,
       a.artifact_type AS artifact_type,
       a.title AS title,
       a.created_at AS created_at,
       a.content_text AS content_text,
       cu.name AS customer_name
FROM artifacts_fts
JOIN artifacts a ON a.artifact_id = artifacts_fts.artifact_id
LEFT JOIN customers cu ON cu.customer_id = a.customer_id
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
            rows = conn.execute(_SEARCH_SQL, (match, k)).fetchall()
    except sqlite3.Error as err:
        return f"Error: {err}"

    if not rows:
        return "No matching artifacts."

    blocks = []
    for row in rows:
        content = str(row["content_text"])[:SEARCH_CONTENT_CHARS]
        customer = row["customer_name"] or "(no customer)"
        meta = f"({row['artifact_type']}, customer: {customer})"
        header = f"[{row['artifact_id']}] {meta} {row['title']} — {row['created_at']}"
        blocks.append(f"{header}\n{content}")
    out = "\n\n---\n\n".join(blocks)
    if len(out) > MAX_RESULT_CHARS:
        out = out[:MAX_RESULT_CHARS] + "\n… (results truncated)"
    return out


@tool
def search_text(query: str, k: int = DEFAULT_SEARCH_K) -> str:
    """FALLBACK keyword full-text search over the artifact corpus (FTS5).

    Use only for topic questions not anchored to a known entity ("who mentioned
    SSO across all accounts"). Returns the top-k artifacts (id, type, title, and
    a content excerpt), ranked by relevance. When you know the customer/scenario,
    prefer run_sql to navigate to its artifacts by foreign key. Cite artifact ids.
    """
    return _search_text(query, k)


# Above this many distinct values a column is treated as high-cardinality / free
# text: we return a sample + the total count instead of the full list, so the
# tool response stays bounded no matter how large the table is.
MAX_DISTINCT_VALUES = 50


def _distinct_values(table: str, column: str) -> str:
    """Core logic for distinct_values; validates identifiers against the live
    catalog (so agent-supplied names can't inject SQL) before reading."""
    with readonly_connection() as conn:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        if table not in tables:
            listed = ", ".join(sorted(t for t in tables if not t.startswith("sqlite_")))
            return f"Error: unknown table {table!r}. Known tables: {listed}."
        # table is now a verified identifier from the catalog; PRAGMA cannot bind it.
        columns = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
        if column not in columns:
            cols = ", ".join(sorted(columns))
            return f"Error: unknown column {column!r} on {table}. Columns: {cols}."

        # table and column are both verified identifiers now — safe to quote-inject.
        total = conn.execute(f'SELECT COUNT(DISTINCT "{column}") FROM "{table}"').fetchone()[0]
        rows = conn.execute(
            f'SELECT "{column}" AS v, COUNT(*) AS n FROM "{table}" '
            f'GROUP BY "{column}" ORDER BY n DESC, v LIMIT ?',
            (MAX_DISTINCT_VALUES,),
        ).fetchall()

    lines = [f"{_truncate(r['v'])} ({r['n']})" for r in rows]
    if total > MAX_DISTINCT_VALUES:
        header = (
            f"{table}.{column}: {total} distinct values (high cardinality — likely free "
            f"text). Top {MAX_DISTINCT_VALUES} by frequency; filter precisely or use "
            f"search_text:"
        )
    else:
        header = f"{table}.{column}: {total} distinct values:"
    return header + "\n- " + "\n- ".join(lines)


@tool
def distinct_values(table: str, column: str) -> str:
    """List the actual DISTINCT values of a column (with row counts).

    Use this to DISCOVER real values BEFORE filtering a categorical / low-cardinality
    column — pain_point, trigger_event, status, account_health, region, crm_stage,
    deployment_model, artifact_type, and the like. Stored wording rarely matches a
    question's phrasing, so guessing a value (e.g. LIKE '%approval-bypass%') usually
    returns nothing; check the real values here, then filter run_sql on the exact
    one. High-cardinality / free-text columns return a frequency-ranked sample plus
    the total count, so the result is always bounded.
    """
    return _distinct_values(table, column)


MAX_CUSTOMER_CANDIDATES = 5


def _norm(text: str) -> str:
    """Casefold and drop everything but alphanumerics, so 'Blue Harbor', 'BlueHarbor'
    and 'blue-harbor' all collapse to the same key (spacing/punctuation/case)."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _match_score(term: str, name: str) -> float:
    """Fuzzy 0..1 score of how well a free-typed term matches a stored name. Combines
    a normalized substring test (spacing/punctuation/case), token overlap (word
    order / extra words like 'company'), and an edit-distance ratio (typos)."""
    term_n, name_n = _norm(term), _norm(name)
    if not term_n or not name_n:
        return 0.0
    substring = 1.0 if (term_n in name_n or name_n in term_n) else 0.0
    term_tokens = set(re.findall(r"[a-z0-9]+", term.lower()))
    name_tokens = set(re.findall(r"[a-z0-9]+", name.lower()))
    overlap = len(term_tokens & name_tokens) / len(term_tokens) if term_tokens else 0.0
    ratio = difflib.SequenceMatcher(None, term_n, name_n).ratio()
    return max(substring, overlap, ratio)


def _find_customer(term: str) -> str:
    """Core logic for find_customer. Loads names and ranks them in Python (no SQL
    interpolation of the term), so it never injects and never fails silently —
    it always returns the closest candidates with their scores for the agent to
    judge. (Loading all names is fine at this scale; production would back this
    with a trigram / search index.)"""
    if not _norm(term):
        return "Error: empty search term."
    with readonly_connection() as conn:
        rows = conn.execute("SELECT customer_id, name, region FROM customers").fetchall()
    ranked = sorted(
        ((_match_score(term, str(r["name"])), r) for r in rows),
        key=lambda sr: sr[0],
        reverse=True,
    )[:MAX_CUSTOMER_CANDIDATES]
    lines = [
        f"{r['name']} (id={r['customer_id']}, region={r['region']}) — match {score:.2f}"
        for score, r in ranked
    ]
    note = "" if ranked and ranked[0][0] >= 0.5 else " (no strong match — refine or pick none)"
    return f"Closest customers to {term!r}{note}:\n- " + "\n- ".join(lines)


@tool
def find_customer(term: str) -> str:
    """Resolve a customer NAME to its record(s) — use this instead of a hand-written
    `name LIKE '%term%'`.

    Returns the closest-matching customers (name, customer_id, region) ranked by a
    fuzzy score, tolerant of spacing, punctuation, casing, and typos — so a name
    typed with different spacing or a missing/added suffix still resolves instead of
    silently returning no rows. Pick the right candidate, then navigate from its
    customer_id. Always returns candidates (never "no rows"); judge from the scores.
    """
    return _find_customer(term)
