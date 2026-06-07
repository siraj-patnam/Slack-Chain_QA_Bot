"""Read-only access to the knowledge-base SQLite database.

The connection is opened in ``mode=ro`` so the agent's tools physically cannot
write, no matter what SQL the model generates. This is the first and strongest
layer of the tool-auth story; the ``assert_safe_select`` guardrail is the second.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import sqlglot
from sqlglot import exp

DEFAULT_DB_PATH = "data/synthetic_startup.sqlite"

# Node types that must never appear in an agent-issued query. The read-only
# connection already blocks writes at the driver level; this allowlist closes
# the door earlier (and rejects PRAGMA/ATTACH, which a read-only conn would
# otherwise permit) by parsing to an AST instead of matching strings.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Command,  # catch-all sqlglot uses for PRAGMA, ATTACH, VACUUM, etc.
    exp.Set,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
)


def assert_safe_select(query: str) -> None:
    """Raise ``ValueError`` unless ``query`` is a single, read-only ``SELECT``.

    Accepts a lone ``SELECT`` (including ``WITH ... SELECT`` CTEs). Rejects
    multiple statements, non-SELECT top-level statements (DDL/DML), and
    PRAGMA/ATTACH and friends. Parsing with sqlglot avoids the well-known
    fragility of regex-based SQL filtering.
    """
    try:
        statements = sqlglot.parse(query, read="sqlite")
    except sqlglot.errors.ParseError as err:
        raise ValueError(f"Could not parse SQL: {err}") from err

    statements = [s for s in statements if s is not None]
    if len(statements) == 0:
        raise ValueError("Empty query.")
    if len(statements) > 1:
        raise ValueError("Only a single statement is allowed.")

    stmt = statements[0]
    if not isinstance(stmt, exp.Select):
        raise ValueError("Only SELECT statements are allowed.")

    for node in stmt.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise ValueError(f"Disallowed SQL construct: {type(node).__name__.upper()}.")


def get_db_path() -> str:
    """Return the configured DB path (``DB_PATH`` env var, else the default)."""
    return os.environ.get("DB_PATH", DEFAULT_DB_PATH)


@contextmanager
def readonly_connection(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a read-only SQLite connection.

    Uses a ``file:...?mode=ro`` URI so the connection cannot mutate the database.
    Raises ``FileNotFoundError`` if the database file is missing, which gives a
    clearer error than sqlite's generic "unable to open database file".
    """
    path = db_path or get_db_path()
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Database not found at {path!r}. Set DB_PATH or place the file there."
        )
    uri = f"file:{Path(path).as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()
