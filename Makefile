.PHONY: help install dev build run serve test test-api test-ci lint format ci clean

PYTEST_ARGS ?= tests/ -v
CI_PYTEST_ARGS ?= tests/ -v --tb=short --junitxml=test-results.xml
PYTHON ?= python

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -r requirements.txt

dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install ruff pytest httpx

build: ## Build the Docker image
	docker build -t semantic-search-engine .

run: ## Run the interactive demo
	python demo.py

serve: ## Start the REST API server
	uvicorn api:app --host 0.0.0.0 --port 8000 --reload

serve-docker: ## Run the API via Docker Compose
	docker compose up --build

test: ## Run the full test suite
	$(PYTHON) -m pytest $(PYTEST_ARGS)

test-api: ## Run API integration tests only
	$(PYTHON) -m pytest tests/test_api.py -v

test-ci: ## Run tests with CI-friendly tracebacks and JUnit output
	$(PYTHON) -m pytest $(CI_PYTEST_ARGS)

lint: ## Run linter (ruff)
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

ci: lint test-ci ## Run the same lint and test gates as GitHub Actions

format: ## Auto-format code
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

clean: ## Remove build artifacts and caches
	rm -rf __pycache__ .pytest_cache .ruff_cache *.egg-info dist build
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
