.DEFAULT_GOAL := help

.PHONY: help install install-examples install-docs lint lint-fix format format-check \
        test test-all test-unit test-integration test-fuzz test-cov \
        docs docs-serve build clean validate check-internal

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dev dependencies
	poetry install --with linter,unittest,tracking

install-examples: ## Install dev dependencies + examples
	poetry install --with examples,linter,unittest,tracking

install-docs: ## Install docs dependencies
	poetry install --with docs

check-internal: ## Check for internal-only keywords in source
	poetry run python tests/validate_internal.py

lint: check-internal ## Run linter + internal keyword check
	poetry run ruff check

lint-fix: ## Auto-fix lint issues
	poetry run ruff check --fix

format: ## Format code
	poetry run ruff format

format-check: ## Check formatting without changes
	poetry run ruff format --check

test: ## Run tests affected by recent changes (testmon)
	cd tests/unit && poetry run pytest --testmon -vvv
	cd tests/integration && poetry run pytest --testmon -vvv

test-all: test-unit test-integration ## Run full test suite

test-unit: ## Run all unit tests
	cd tests/unit && poetry run pytest -vvv

test-integration: ## Run all integration tests
	cd tests/integration && poetry run pytest -vvv

test-fuzz: ## Run fuzz tests (slow, Hypothesis with wide ranges)
	poetry run pytest tests/fuzz/ -vvv

test-cov: ## Run tests with coverage report
	poetry run pytest tests/unit tests/integration -vvv --cov=compileiq --cov-report=term-missing

test-cov-html: ## Run tests with coverage report
	poetry run pytest tests/unit tests/integration -vvv --cov=compileiq --cov-report=html

docs: ## Build documentation
	CIQ_DOCS_ENV=dev poetry run sphinx-multiversion -E -a docs/ public/

docs-serve: ## Build and serve docs locally
	bash dev/view_docs.sh

build: ## Build wheel and sdist
	poetry build

clean: ## Remove build artifacts and caches
	rm -rf dist/ public/ .coverage .testmondata
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find . -name '.cache' -exec rm -rf {} + 2>/dev/null || true
	find . -name '.testmondata' -exec rm -rf {} + 2>/dev/null || true

validate: lint test-unit ## Quick validation (lint + unit tests)