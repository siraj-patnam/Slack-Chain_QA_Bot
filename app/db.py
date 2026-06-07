"""Read-only access to the knowledge-base SQLite database.

The connection is opened in ``mode=ro`` so the agent's tools physically cannot
write, no matter what SQL the model generates. This is the first and strongest
layer of the tool-auth story; the SQL guardrail (added next) is the second.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DEFAULT_DB_PATH = "data/synthetic_startup.sqlite"


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
