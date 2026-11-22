PY ?= python
.DEFAULT_GOAL := help

.PHONY: help install test test-all lint format eval demo serve clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Editable install with dev extras
	$(PY) -m pip install -e ".[dev]"

test:  ## Fast suite: offline, no model, no server, ~1 second
	$(PY) -m pytest

test-all:  ## Everything, including tests needing a live Qdrant
	$(PY) -m pytest -m "integration or not integration"

lint:  ## Static checks
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:  ## Apply formatting and safe fixes
	$(PY) -m ruff check . --fix
	$(PY) -m ruff format .

eval:  ## Retrieval and groundedness metrics on the labelled set
	$(PY) eval/run_eval.py

demo:  ## Ask a question against the sample corpus, end to end
	$(PY) -m ragkit.cli ask "How much notice must an employee give before resigning?" \
		--doc eval/corpus/employee_handbook.md

demo-refusal:  ## The same, for a question the corpus cannot answer
	-$(PY) -m ragkit.cli ask "What is the capital of Mongolia?" \
		--doc eval/corpus/employee_handbook.md

serve:  ## Run the HTTP API on localhost:8000
	$(PY) -m ragkit.cli serve

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
