.DEFAULT_GOAL := help

.PHONY: help install install-examples install-docs lint lint-fix format format-check \
        typecheck test test-all test-unit test-integration test-fuzz test-cov \
        docs docs-serve build clean validate check-internal \
        build-search-space-manifest build-search-space-release-notes \
        build-search-space-release build-search-space-manifest-schema \
        check-search-space-manifest-schema clean-search-cache

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dev dependencies
	poetry install --with linter,typecheck,unittest,tracking

install-examples: ## Install dev dependencies + examples
	poetry install --with examples,linter,typecheck,unittest,tracking

install-docs: ## Install docs dependencies
	poetry install --with docs

check-internal: ## Check for internal-only keywords in source
	poetry run python tests/validate_internal.py

lint: ## Run linter
	poetry run ruff check

typecheck: ## Run pyright type checker
	poetry run pyright

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

validate: lint typecheck test-unit ## Quick validation (lint + typecheck + unit tests)

# ── Search-space release targets ─────────────────────────────────────────────

# Release tag written into manifest.json and release notes. Override for RCs or
# backfilled catalogs, e.g. SS_TAG=search-spaces-2026.05.12-rc1.
SS_TAG ?= search-spaces-$(shell date +%Y.%m.%d)

# Human-authored catalog source. Override to test a draft catalog without
# editing the checked-in launch source.
SS_MANIFEST_SOURCE ?= release/search-spaces/manifest-source.yaml

# Directory for generated manifest and release notes. This is safe to delete.
SS_OUTPUT_DIR ?= dist/search-space-release

# Generated release catalog path. Upload this as manifest.json in the GitHub release.
SS_MANIFEST_OUT ?= $(SS_OUTPUT_DIR)/manifest.json

# Generated Markdown release notes path. Paste or upload this with the release.
SS_RELEASE_NOTES_OUT ?= $(SS_OUTPUT_DIR)/release-notes.md

# Required input directory containing staged .bin assets named by SS_MANIFEST_SOURCE.
# Example:
#   make build-search-space-release SS_ARTIFACTS_DIR=/path/to/bins SS_TAG=search-spaces-YYYY.MM.DD
build-search-space-manifest: ## Build search-space release catalog manifest.json
	@test -n "$(SS_ARTIFACTS_DIR)" || (echo "ERROR: set SS_ARTIFACTS_DIR=/path/to/search-space-bins" >&2; exit 1)
	@test -d "$(SS_ARTIFACTS_DIR)" || (echo "ERROR: SS_ARTIFACTS_DIR does not exist: $(SS_ARTIFACTS_DIR)" >&2; exit 1)
	poetry run python dev/build_manifest.py --source "$(SS_MANIFEST_SOURCE)" --artifacts-dir "$(SS_ARTIFACTS_DIR)" --tag "$(SS_TAG)" --out "$(SS_MANIFEST_OUT)"
	@echo "Wrote $(SS_MANIFEST_OUT) for $(SS_TAG)."

build-search-space-release-notes: ## Build Markdown release notes from generated search-space manifest
	@test -f "$(SS_MANIFEST_OUT)" || (echo "ERROR: missing $(SS_MANIFEST_OUT); run build-search-space-manifest first" >&2; exit 1)
	poetry run python dev/build_search_space_release_notes.py --manifest "$(SS_MANIFEST_OUT)" --out "$(SS_RELEASE_NOTES_OUT)"
	@echo "Wrote $(SS_RELEASE_NOTES_OUT)."

build-search-space-release: build-search-space-manifest build-search-space-release-notes ## Build manifest + notes for manual release upload

build-search-space-manifest-schema: ## Regenerate the JSON Schema that validates release catalog manifests
	poetry run python dev/generate_manifest_schema.py

check-search-space-manifest-schema: ## Verify the checked-in search-space manifest JSON Schema is current
	poetry run python dev/generate_manifest_schema.py --check

clean-search-cache: ## Clear the local resolver cache (~/.cache/compileiq/)
	rm -rf $(HOME)/.cache/compileiq/
