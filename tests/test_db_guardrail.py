"""Tests for the SQL guardrail (app.db.assert_safe_select)."""

import pytest

from app.db import assert_safe_select

VALID = [
    "SELECT 1",
    "SELECT * FROM artifacts WHERE artifact_type = 'customer_call' LIMIT 5",
    "SELECT COUNT(*) FROM customers WHERE region = 'North America West'",
    "WITH t AS (SELECT customer_id FROM customers) SELECT * FROM t",
]

INVALID = [
    "DROP TABLE artifacts",
    "SELECT 1; SELECT 2",
    "SELECT * FROM artifacts; DROP TABLE artifacts",
    "PRAGMA table_info(artifacts)",
    'ATTACH DATABASE "evil.db" AS e',
    "INSERT INTO customers (name) VALUES ('x')",
    "UPDATE customers SET name = 'x'",
    "DELETE FROM customers",
    "VACUUM",
    "",
]


@pytest.mark.parametrize("query", VALID)
def test_accepts_read_only_selects(query: str) -> None:
    assert_safe_select(query)  # should not raise


@pytest.mark.parametrize("query", INVALID)
def test_rejects_unsafe_statements(query: str) -> None:
    with pytest.raises(ValueError):
        assert_safe_select(query)
