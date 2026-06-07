.PHONY: lint type test check eval

lint:
	uv run ruff check .
	uv run ruff format --check .

type:
	uv run mypy app/

test:
	uv run pytest -q

check: lint type test

# eval target is wired in PR3 (python -m evals.run); placeholder until then.
eval:
	@echo "make eval is added in PR3"
