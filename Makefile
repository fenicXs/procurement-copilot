.PHONY: setup lint test data build api eval

setup:
	pip install -e ".[dev]"

lint:
	ruff check .
	black --check .
	mypy src/

test:
	pytest

data:
	python scripts/download_far.py
	python scripts/fetch_usaspending_awards.py

build:
	python scripts/build_duckdb.py
	python scripts/chunk_far.py
	python scripts/build_vector_index.py
	python scripts/build_kg.py

api:
	uvicorn procurement_copilot.api.main:app --reload --host 0.0.0.0 --port 8000

eval:
	python scripts/run_eval.py
