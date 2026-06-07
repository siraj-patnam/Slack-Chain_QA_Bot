.PHONY: lint type test check eval

lint:
	uv run ruff check .
	uv run ruff format --check .

type:
	uv run mypy app/

test:
	uv run pytest -q

check: lint type test

eval:
	uv run python -m evals.run
