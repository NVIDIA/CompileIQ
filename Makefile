.DEFAULT_GOAL := help

.PHONY: help install install-frontend install-complete materialize-native-runtime install-examples install-examples-complete install-docs install-release lint lint-fix format format-check \
        typecheck test test-all test-unit test-integration test-fuzz test-cov \
        docs docs-serve docs-preview build compose-complete-wheel verify-complete-wheel \
        test-complete-wheel-install clean validate \
        setup-compileiq-package-release \
        setup-search-space-release update-search-space-catalog \
        build-search-space-release check-search-space-staging check-search-space-assets \
        create-search-space-draft-release inspect-search-space-release \
        publish-search-space-release clear-search-space-latest check-search-space-published \
        build-search-space-manifest build-search-space-release-notes \
        build-search-space-manifest-schema \
        setup-booster-pack-release update-booster-pack-catalog \
        build-booster-pack-release check-booster-pack-staging check-booster-pack-assets \
        inspect-booster-pack-release publish-booster-pack-release \
        clear-booster-pack-latest check-booster-pack-published \
        check-search-space-manifest-schema clean-search-cache

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# An explicit materialization override. Without one, the lane manifest names
# the exact wheel selected from the preferred runtime directories below.
NATIVE_RUNTIME_WHEEL ?=
NATIVE_RUNTIME_MANIFEST ?=
NATIVE_RUNTIME_SOURCE ?=
NATIVE_RUNTIME_DIRECTORY ?= native-runtime-wheels
NATIVE_RUNTIME_CI_MANIFESTS = \
	$(wildcard dist/native-runtime/native-runtime-lane-manifest*.json)
NATIVE_RUNTIME_CHECKED_IN_MANIFESTS = \
	$(wildcard $(NATIVE_RUNTIME_DIRECTORY)/native-runtime-manifest*.json)
NATIVE_RUNTIME_MANIFESTS ?= $(if \
	$(NATIVE_RUNTIME_MANIFEST), \
	$(NATIVE_RUNTIME_MANIFEST), \
	$(if \
		$(NATIVE_RUNTIME_CI_MANIFESTS), \
		$(NATIVE_RUNTIME_CI_MANIFESTS), \
		$(NATIVE_RUNTIME_CHECKED_IN_MANIFESTS)))

install: ## Install the editable frontend and materialize the native amber runtime (needs git lfs pull)
	poetry install --with linter,typecheck,unittest,tracking,native-runtime
	$(MAKE) materialize-native-runtime

install-frontend: ## Install the editable frontend without an engine (lint, typecheck, and unit tests only)
	poetry install --with linter,typecheck,unittest,tracking

install-complete: install ## Compatibility alias for install

materialize-native-runtime: ## Install a verified CI payload or build the explicit local override
	@if [ -n "$(NATIVE_RUNTIME_SOURCE)" ] && [ -d "$(NATIVE_RUNTIME_SOURCE)" ]; then \
		python_path="$$(poetry run python -c 'import sys; print(sys.executable)')"; \
		$(MAKE) -C "$(NATIVE_RUNTIME_SOURCE)" native-build-deps audit-native-runtime-wheel \
			PYTHON="$$python_path"; \
		"$$python_path" -m pip install --no-index --no-deps --force-reinstall \
			--find-links "$(NATIVE_RUNTIME_SOURCE)/dist" compileiq-native-runtime; \
		"$$python_path" -c "from compileiq.backends import get_backend_registration; \
			registration = get_backend_registration('amber'); \
			assert registration.name == 'amber'" \
			|| { echo "ERROR: the installed native runtime could not register its amber backend." >&2; exit 1; }; \
	elif [ -n "$(strip $(NATIVE_RUNTIME_MANIFESTS))" ]; then \
		poetry run python dev/materialize_native_runtime.py \
			$(if $(NATIVE_RUNTIME_WHEEL),--runtime-wheel "$(NATIVE_RUNTIME_WHEEL)",) \
			--runtime-directory dist/native-runtime \
			--runtime-directory "$(NATIVE_RUNTIME_DIRECTORY)" \
			$(foreach manifest,$(NATIVE_RUNTIME_MANIFESTS),--lane-manifest "$(manifest)"); \
	else \
		echo "ERROR: no verified native payload is staged." >&2; \
		echo "Provide NATIVE_RUNTIME_MANIFEST (and optionally NATIVE_RUNTIME_WHEEL)," >&2; \
		echo "stage a verified manifest/wheel pair, or use the explicit" >&2; \
		echo "NATIVE_RUNTIME_SOURCE=/path/to/native-source override." >&2; \
		exit 1; \
	fi

install-examples: ## Install dev dependencies + examples and materialize the native amber runtime
	poetry install --with examples,linter,typecheck,unittest,tracking,native-runtime
	$(MAKE) materialize-native-runtime

install-examples-complete: install-examples ## Compatibility alias for install-examples

install-docs: ## Install docs dependencies
	poetry install --with docs

install-release: ## Install release-prep helper dependencies
	poetry install --with release,unittest

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

DOC_VERSION ?= latest
DOCS_PORT ?= 8000

docs: ## Build documentation for DOC_VERSION=latest or MAJOR.MINOR
	@version="$(DOC_VERSION)"; \
	folder="latest"; \
	if [ "$$version" != "latest" ]; then folder="v$$version"; fi; \
	DOC_VERSION="$$version" poetry run sphinx-build -E -a docs "public/$$folder"

docs-serve: ## Build and serve local docs preview
	bash dev/view_docs.sh

docs-preview: ## Build and serve live worktree docs locally
	CIQ_DOCS_LOCAL_PREVIEW=1 CIQ_DOCS_BASE_URL=http://localhost:$(DOCS_PORT) DOC_VERSION=latest poetry run sphinx-build -E -a docs public/latest
	@echo "Serving docs at http://localhost:$(DOCS_PORT)/latest/"
	poetry run python -m http.server $(DOCS_PORT) --directory public

build: ## Build wheel and sdist
	poetry build

FRONTEND_WHEEL ?= $(firstword $(wildcard dist/compileiq-[0-9]*.whl))
COMPOSE_NATIVE_RUNTIME_WHEEL ?= $(if \
	$(NATIVE_RUNTIME_WHEEL), \
	$(NATIVE_RUNTIME_WHEEL), \
	$(firstword $(wildcard dist/native-runtime/compileiq_native_runtime-*.whl)))
LANE_MANIFEST ?= dist/native-runtime/native-runtime-lane-manifest.json
COMPLETE_WHEEL_DIR ?= dist/complete
COMPLETE_WHEEL ?= $(firstword $(wildcard $(COMPLETE_WHEEL_DIR)/compileiq-*.whl))
COMPLETE_WHEEL_MANIFEST ?= $(COMPLETE_WHEEL_DIR)/composition-manifest.json

compose-complete-wheel: ## Compose one publishable wheel from manifested frontend/runtime inputs
	@test -n "$(FRONTEND_WHEEL)" || (echo "ERROR: set FRONTEND_WHEEL=/path/to/compileiq.whl" >&2; exit 1)
	@test -n "$(COMPOSE_NATIVE_RUNTIME_WHEEL)" || (echo "ERROR: set NATIVE_RUNTIME_WHEEL=/path/to/runtime.whl" >&2; exit 1)
	@test -f "$(LANE_MANIFEST)" || (echo "ERROR: set LANE_MANIFEST=/path/to/native-runtime-lane-manifest.json" >&2; exit 1)
	poetry run python dev/compose_compileiq_wheel.py \
		--frontend-wheel "$(FRONTEND_WHEEL)" \
		--runtime-wheel "$(COMPOSE_NATIVE_RUNTIME_WHEEL)" \
		--lane-manifest "$(LANE_MANIFEST)" \
		--output-dir "$(COMPLETE_WHEEL_DIR)" \
		--output-manifest "$(COMPLETE_WHEEL_MANIFEST)"

verify-complete-wheel: ## Audit a composed wheel as the single public distribution
	@test -n "$(COMPLETE_WHEEL)" || (echo "ERROR: set COMPLETE_WHEEL=/path/to/complete.whl" >&2; exit 1)
	poetry run python dev/verify_complete_wheel.py "$(COMPLETE_WHEEL)" \
		--composition-manifest "$(COMPLETE_WHEEL_MANIFEST)"

test-complete-wheel-install: ## Clean-install only the complete public wheel
	@test -n "$(COMPLETE_WHEEL)" || (echo "ERROR: set COMPLETE_WHEEL=/path/to/complete.whl" >&2; exit 1)
	poetry run python dev/test_complete_wheel_install.py --wheel "$(COMPLETE_WHEEL)"

clean: ## Remove build artifacts and caches
	rm -rf dist/ public/ .coverage .testmondata
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find . -name '.cache' -exec rm -rf {} + 2>/dev/null || true
	find . -name '.testmondata' -exec rm -rf {} + 2>/dev/null || true

validate: lint typecheck test-unit ## Quick validation (lint + typecheck + unit tests)

# CompileIQ package release targets.

RELEASE_VERSION ?=
PACKAGE_RELEASE_ENV_FILE ?= dist/compileiq-package-release/current.env

setup-compileiq-package-release: ## Write sourceable env vars for a CompileIQ package release
	@test -n "$(RELEASE_VERSION)" || (echo "ERROR: set RELEASE_VERSION=MAJOR.MINOR.PATCH[SUFFIX], e.g. RELEASE_VERSION=1.0.0" >&2; exit 1)
	@test "$(RELEASE_VERSION)" != "MAJOR.MINOR.PATCH" || (echo "ERROR: replace MAJOR.MINOR.PATCH with the actual release version" >&2; exit 1)
	@mkdir -p "$$(dirname "$(PACKAGE_RELEASE_ENV_FILE)")"
	@if ! printf '%s\n' "$(RELEASE_VERSION)" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc|dev)[0-9]+)?$$'; then \
		echo "ERROR: RELEASE_VERSION must match MAJOR.MINOR.PATCH with optional aN, bN, rcN, or devN suffix, e.g. 1.0.0 or 1.0.0rc5" >&2; \
		exit 1; \
	fi; \
	release_major_minor="$$(printf '%s\n' "$(RELEASE_VERSION)" | sed -E 's/^([0-9]+\.[0-9]+)\..*$$/\1/')"; \
	{ \
		echo 'export RELEASE_VERSION="$(RELEASE_VERSION)"'; \
		echo 'export RELEASE_TAG="v$(RELEASE_VERSION)"'; \
		echo "export RELEASE_MAJOR_MINOR=\"$$release_major_minor\""; \
		echo "export RELEASE_BRANCH=\"release-$$release_major_minor\""; \
	} > "$(PACKAGE_RELEASE_ENV_FILE)"; \
	echo "Wrote $(PACKAGE_RELEASE_ENV_FILE):"; \
	cat "$(PACKAGE_RELEASE_ENV_FILE)"; \
	echo; \
	echo "Run to set environment variables for release process:"; \
	echo "  source $(PACKAGE_RELEASE_ENV_FILE)"; \

# Search Space release targets.

# Compatibility alias for older local commands. Prefer SS_RELEASE_TAG.
SS_TAG ?=

# Search Space release tag being prepared. Advanced override only for backfills
# or repairs, e.g. SS_RELEASE_TAG=search-spaces-2026.05.21-rev1.
SS_RELEASE_TAG ?= $(if $(SS_TAG),$(SS_TAG),search-spaces-$(shell date +%Y.%m.%d))

# Prior Search Space release tag used to seed an incremental catalog update.
# Defaults to the newest published search-spaces-* GitHub release. Empty means
# first-release bootstrap.
SS_PRIOR_RELEASE_TAG ?= $(shell gh release list --repo NVIDIA/CompileIQ --exclude-drafts --exclude-pre-releases --limit 100 --json tagName --jq '.[] | select(.tagName | startswith("search-spaces-")) | .tagName' 2>/dev/null | head -n 1)

# Checked-in launch metadata used only when no prior search-spaces-* release exists.
SS_MANIFEST_SOURCE ?= release/search-spaces/manifest-source.yaml

# Repo-local working directory for Search Space release preparation. This is
# ignored through the repo's existing **/dist ignore rule.
SS_RELEASE_ROOT ?= dist/search-space-release/$(SS_RELEASE_TAG)

# Convenience env file written by setup-search-space-release for this shell.
SS_ENV_FILE ?= dist/search-space-release/current.env

# Required input directory containing manifest-source.yaml plus approved .bin files.
SS_INPUT_DIR ?= $(SS_RELEASE_ROOT)/release-inputs

# Directory for generated manifest, .bin assets, checksums, and release body.
SS_OUTPUT_DIR ?= $(SS_RELEASE_ROOT)/staged-release

# Stable public docs URL written into generated release-body.md.
SS_DOCS_URL ?= https://nvidia.github.io/CompileIQ/stable/compilers_overview.html

# Compatibility paths for older local commands.
SS_ARTIFACTS_DIR ?= $(SS_INPUT_DIR)
SS_MANIFEST_OUT ?= $(SS_OUTPUT_DIR)/manifest.json
SS_RELEASE_NOTES_OUT ?= $(SS_OUTPUT_DIR)/release-body.md

setup-search-space-release: ## Seed Search Space release inputs from a previous GitHub release or bootstrap source
	poetry run python dev/setup_search_space_release.py \
		--input-dir "$(SS_INPUT_DIR)" \
		--env-file "$(SS_ENV_FILE)" \
		--release-tag "$(SS_RELEASE_TAG)" \
		--manifest-source "$(SS_MANIFEST_SOURCE)" \
		--output-dir "$(SS_OUTPUT_DIR)" \
		$(if $(SS_PRIOR_RELEASE_TAG),--prior-tag "$(SS_PRIOR_RELEASE_TAG)",)

update-search-space-catalog: ## Reconcile Search Space manifest source with staged .bin files
	@test -n "$(SS_INPUT_DIR)" || (echo "ERROR: set SS_INPUT_DIR=/path/to/search-space-release-inputs" >&2; exit 1)
	@test -d "$(SS_INPUT_DIR)" || (echo "ERROR: SS_INPUT_DIR does not exist: $(SS_INPUT_DIR)" >&2; exit 1)
	poetry run python dev/update_search_space_catalog.py "$(SS_INPUT_DIR)" \
		--tag "$(SS_RELEASE_TAG)"

build-search-space-release: ## Build Search Space manifest + .bin assets + checksums + release body
	@test -n "$(SS_INPUT_DIR)" || (echo "ERROR: set SS_INPUT_DIR=/path/to/search-space-release-inputs" >&2; exit 1)
	@test -d "$(SS_INPUT_DIR)" || (echo "ERROR: SS_INPUT_DIR does not exist: $(SS_INPUT_DIR)" >&2; exit 1)
	poetry run python dev/build_search_space_release.py \
		--source-dir "$(SS_INPUT_DIR)" \
		--output-dir "$(SS_OUTPUT_DIR)" \
		--tag "$(SS_RELEASE_TAG)" \
		--docs-url "$(SS_DOCS_URL)" \
		--clean-output

# Validate local staged output before upload. Requires release-body.md because it
# becomes the GitHub Release notes.
check-search-space-staging: ## Validate local staged Search Space release before upload
	poetry run python dev/verify_search_space_release.py \
		"$(SS_OUTPUT_DIR)" \
		--tag "$(SS_RELEASE_TAG)" \
		--extra-ok release-body.md \
		--docs-url "$(SS_DOCS_URL)" \
		--require-release-body

# Validate assets downloaded from GitHub. release-body.md is not uploaded as an
# asset; GitHub stores it as the release notes.
check-search-space-assets: ## Validate Search Space assets downloaded from GitHub
	poetry run python dev/verify_search_space_release.py \
		"$(SS_OUTPUT_DIR)" \
		--tag "$(SS_RELEASE_TAG)" \
		--extra-ok release-body.md \
		--docs-url "$(SS_DOCS_URL)"

create-search-space-draft-release: ## Create draft Search Space release from staged assets
	gh release create "$(SS_RELEASE_TAG)" \
		"$(SS_OUTPUT_DIR)/manifest.json" \
		"$(SS_OUTPUT_DIR)"/*.bin \
		"$(SS_OUTPUT_DIR)/SHA256SUMS.txt" \
		--repo NVIDIA/CompileIQ \
		--target main \
		--title "Search Space Catalog Release $(SS_RELEASE_TAG)" \
		--latest=false \
		--draft \
		--notes-file "$(SS_OUTPUT_DIR)/release-body.md"

# Show draft release metadata for a final read-only human sanity check.
inspect-search-space-release: ## Inspect draft Search Space release before publishing
	gh release view "$(SS_RELEASE_TAG)" \
		--repo NVIDIA/CompileIQ \
		--json tagName,name,isDraft,isPrerelease,targetCommitish,assets,body \
		--template 'tag: {{.tagName}}{{"\n"}}title: {{.name}}{{"\n"}}draft: {{.isDraft}}{{"\n"}}prerelease: {{.isPrerelease}}{{"\n"}}target: {{.targetCommitish}}{{"\n"}}assets:{{"\n"}}{{range .assets}}  - {{.name}}{{"\n"}}{{end}}{{"\n"}}body:{{"\n"}}{{.body}}{{"\n"}}'

# Publish the validated draft without changing GitHub "Latest". Repositories
# with immutable releases seal a release as it is published, so both fields
# must be updated in the same API call.
publish-search-space-release: ## Publish draft Search Space release with make_latest=false
	@set -eu; \
	if [ "$(CONFIRM_PUBLISH_RELEASE)" != "false" ]; then \
		printf 'Type %s to publish: ' "$(SS_RELEASE_TAG)"; \
		read confirmation; \
		if [ "$$confirmation" != "$(SS_RELEASE_TAG)" ]; then \
			echo "ERROR: confirmation did not match $(SS_RELEASE_TAG); release was not published." >&2; \
			exit 1; \
		fi; \
	fi; \
	release_id="$$(gh release view "$(SS_RELEASE_TAG)" --repo NVIDIA/CompileIQ --json databaseId --jq .databaseId)"; \
	test -n "$$release_id" || (echo "ERROR: could not find release $(SS_RELEASE_TAG)" >&2; exit 1); \
	gh api --method PATCH "repos/NVIDIA/CompileIQ/releases/$$release_id" \
		-F draft=false \
		-f make_latest=false \
		--silent
	@$(MAKE) --no-print-directory check-search-space-published SS_RELEASE_TAG="$(SS_RELEASE_TAG)"
	@echo "PASS: Published $(SS_RELEASE_TAG) with make_latest=false."

clear-search-space-latest: ## Clear GitHub Latest marker from a Search Space release
	@set -eu; \
	release_id="$$(gh release view "$(SS_RELEASE_TAG)" --repo NVIDIA/CompileIQ --json databaseId --jq .databaseId)"; \
	test -n "$$release_id" || (echo "ERROR: could not find release $(SS_RELEASE_TAG)" >&2; exit 1); \
	gh api --method PATCH "repos/NVIDIA/CompileIQ/releases/$$release_id" \
		-f make_latest=false \
		--silent
	@$(MAKE) --no-print-directory check-search-space-published SS_RELEASE_TAG="$(SS_RELEASE_TAG)"
	@echo "PASS: Cleared GitHub Latest marker for $(SS_RELEASE_TAG)."

# Confirm the published release is no longer a draft and is not GitHub "Latest".
check-search-space-published: ## Confirm published Search Space release is not Latest
	@status="$$(gh release list --repo NVIDIA/CompileIQ --limit 100 --json tagName,isDraft,isLatest --jq '.[] | select(.tagName == "$(SS_RELEASE_TAG)") | [.isDraft, .isLatest] | @tsv')"; \
	test -n "$$status" || (echo "ERROR: could not find release $(SS_RELEASE_TAG)" >&2; exit 1); \
	set -- $$status; \
	if [ "$$1" = "false" ] && [ "$$2" = "false" ]; then \
		echo "PASS: $(SS_RELEASE_TAG) is published and is not marked Latest."; \
	else \
		echo "FAIL: $(SS_RELEASE_TAG) has isDraft=$$1 isLatest=$$2." >&2; \
		exit 1; \
	fi

build-search-space-manifest: ## Build legacy search-space manifest.json from SS_MANIFEST_SOURCE + SS_ARTIFACTS_DIR
	@test -n "$(SS_ARTIFACTS_DIR)" || (echo "ERROR: set SS_ARTIFACTS_DIR=/path/to/search-space-bins" >&2; exit 1)
	@test -d "$(SS_ARTIFACTS_DIR)" || (echo "ERROR: SS_ARTIFACTS_DIR does not exist: $(SS_ARTIFACTS_DIR)" >&2; exit 1)
	poetry run python dev/build_manifest.py --source "$(SS_MANIFEST_SOURCE)" --artifacts-dir "$(SS_ARTIFACTS_DIR)" --tag "$(SS_RELEASE_TAG)" --out "$(SS_MANIFEST_OUT)"
	@echo "Wrote $(SS_MANIFEST_OUT) for $(SS_RELEASE_TAG)."

build-search-space-release-notes: ## Build legacy Markdown release notes from generated search-space manifest
	@test -f "$(SS_MANIFEST_OUT)" || (echo "ERROR: missing $(SS_MANIFEST_OUT); run build-search-space-manifest first" >&2; exit 1)
	poetry run python dev/build_search_space_release_notes.py --manifest "$(SS_MANIFEST_OUT)" --out "$(SS_RELEASE_NOTES_OUT)"
	@echo "Wrote $(SS_RELEASE_NOTES_OUT)."

build-search-space-manifest-schema: ## Regenerate the JSON Schema that validates release catalog manifests
	poetry run python dev/generate_manifest_schema.py

check-search-space-manifest-schema: ## Verify the checked-in search-space manifest JSON Schema is current
	poetry run python dev/generate_manifest_schema.py --check

clean-search-cache: ## Clear the local resolver cache (~/.cache/compileiq/)
	rm -rf $(HOME)/.cache/compileiq/

# Booster Pack release targets.

# Booster Pack release tag being prepared. Advanced override only for backfills
# or repairs, e.g. BOOSTER_RELEASE_TAG=booster-packs-2026.05.21.
BOOSTER_RELEASE_TAG ?= booster-packs-$(shell date +%Y.%m.%d)

# Prior Booster Pack release tag used to seed an incremental catalog update.
# Defaults to the newest published booster-packs-* GitHub release. Advanced
# override only for backfills or repairs, e.g. BOOSTER_PRIOR_RELEASE_TAG=booster-packs-2026.05.21.
BOOSTER_PRIOR_RELEASE_TAG ?= $(shell gh release list --repo NVIDIA/CompileIQ --exclude-drafts --exclude-pre-releases --limit 100 --json tagName --jq '.[] | select(.tagName | startswith("booster-packs-")) | .tagName' 2>/dev/null | head -n 1)

# Repo-local working directory for Booster Pack release preparation. This is
# ignored through the repo's existing **/dist ignore rule.
BOOSTER_RELEASE_ROOT ?= dist/booster-pack-release/$(BOOSTER_RELEASE_TAG)

# Convenience env file written by setup-booster-pack-release for this shell.
BOOSTER_ENV_FILE ?= dist/booster-pack-release/current.env

# Required input directory containing a booster-pack-catalog.json plus
# the pack zip assets named by that catalog.
BOOSTER_INPUT_DIR ?= $(BOOSTER_RELEASE_ROOT)/release-inputs

# Directory for generated catalog, pack zips, checksums, and release body.
BOOSTER_OUTPUT_DIR ?= $(BOOSTER_RELEASE_ROOT)/staged-release

# Stable public docs URL written into generated release-body.md.
BOOSTER_DOCS_URL ?= https://nvidia.github.io/CompileIQ/stable/booster_packs.html

# Prompt for exact-tag confirmation before publishing by default. Set
# CONFIRM_PUBLISH_RELEASE=false only for controlled noninteractive automation.
CONFIRM_PUBLISH_RELEASE ?= true

setup-booster-pack-release: ## Seed Booster Pack release inputs from a previous GitHub release
	@test -n "$(BOOSTER_PRIOR_RELEASE_TAG)" || (echo "ERROR: could not infer BOOSTER_PRIOR_RELEASE_TAG; set BOOSTER_PRIOR_RELEASE_TAG=booster-packs-YYYY.MM.DD[-suffix]" >&2; exit 1)
	@test "$(BOOSTER_PRIOR_RELEASE_TAG)" != "$(BOOSTER_RELEASE_TAG)" || (echo "ERROR: BOOSTER_RELEASE_TAG matches BOOSTER_PRIOR_RELEASE_TAG; choose a new release tag" >&2; exit 1)
	@mkdir -p "$(BOOSTER_INPUT_DIR)"
	@test -z "$$(find "$(BOOSTER_INPUT_DIR)" -mindepth 1 -maxdepth 1 -print -quit)" || (echo "ERROR: BOOSTER_INPUT_DIR is not empty: $(BOOSTER_INPUT_DIR)" >&2; exit 1)
	@{ \
		echo 'export BOOSTER_RELEASE_TAG="$(BOOSTER_RELEASE_TAG)"'; \
		echo 'export BOOSTER_PRIOR_RELEASE_TAG="$(BOOSTER_PRIOR_RELEASE_TAG)"'; \
		echo 'export BOOSTER_RELEASE_ROOT="$(BOOSTER_RELEASE_ROOT)"'; \
		echo 'export BOOSTER_INPUT_DIR="$(BOOSTER_INPUT_DIR)"'; \
		echo 'export BOOSTER_OUTPUT_DIR="$(BOOSTER_OUTPUT_DIR)"'; \
	} > "$(BOOSTER_ENV_FILE)"
	gh release download "$(BOOSTER_PRIOR_RELEASE_TAG)" \
		--repo NVIDIA/CompileIQ \
		--dir "$(BOOSTER_INPUT_DIR)" \
		--pattern "booster-pack-catalog.json" \
		--pattern "booster-pack-*.zip"
	cp "$(BOOSTER_INPUT_DIR)/booster-pack-catalog.json" "$(BOOSTER_INPUT_DIR)/.booster-pack-catalog.prior-release.json"
	@echo "Seeded $(BOOSTER_INPUT_DIR) from $(BOOSTER_PRIOR_RELEASE_TAG)."
	@echo "Saved prior-release comparison catalog in $(BOOSTER_INPUT_DIR)/.booster-pack-catalog.prior-release.json."
	@echo "Wrote $(BOOSTER_ENV_FILE). Run: source $(BOOSTER_ENV_FILE)"
	@echo "Review release-inputs as the full catalog contents before building $(BOOSTER_RELEASE_TAG)."

update-booster-pack-catalog: ## Reconcile Booster Pack input catalog with staged zip files
	@test -n "$(BOOSTER_INPUT_DIR)" || (echo "ERROR: set BOOSTER_INPUT_DIR=/path/to/booster-release-inputs" >&2; exit 1)
	@test -d "$(BOOSTER_INPUT_DIR)" || (echo "ERROR: BOOSTER_INPUT_DIR does not exist: $(BOOSTER_INPUT_DIR)" >&2; exit 1)
	poetry run python dev/update_booster_pack_catalog.py "$(BOOSTER_INPUT_DIR)" \
		--tag "$(BOOSTER_RELEASE_TAG)"

build-booster-pack-release: ## Build Booster Pack catalog + zip assets + checksums + release body
	@test -n "$(BOOSTER_INPUT_DIR)" || (echo "ERROR: set BOOSTER_INPUT_DIR=/path/to/booster-release-inputs" >&2; exit 1)
	@test -d "$(BOOSTER_INPUT_DIR)" || (echo "ERROR: BOOSTER_INPUT_DIR does not exist: $(BOOSTER_INPUT_DIR)" >&2; exit 1)
	poetry run python dev/build_booster_pack_release.py \
		--source-dir "$(BOOSTER_INPUT_DIR)" \
		--output-dir "$(BOOSTER_OUTPUT_DIR)" \
		--tag "$(BOOSTER_RELEASE_TAG)" \
		--docs-url "$(BOOSTER_DOCS_URL)" \
		--clean-output

# Validate local staged output before upload. Requires release-body.md because it
# becomes the GitHub Release notes.
check-booster-pack-staging: ## Validate local staged Booster Pack release before upload
	poetry run python dev/verify_booster_pack_release.py \
		"$(BOOSTER_OUTPUT_DIR)" \
		--tag "$(BOOSTER_RELEASE_TAG)" \
		--extra-ok release-body.md \
		--docs-url "$(BOOSTER_DOCS_URL)" \
		--require-validation-passed \
		--require-release-body

# Validate assets downloaded from GitHub. release-body.md is not uploaded as an
# asset; GitHub stores it as the release notes.
check-booster-pack-assets: ## Validate Booster Pack assets downloaded from GitHub
	poetry run python dev/verify_booster_pack_release.py \
		"$(BOOSTER_OUTPUT_DIR)" \
		--tag "$(BOOSTER_RELEASE_TAG)" \
		--extra-ok release-body.md \
		--docs-url "$(BOOSTER_DOCS_URL)" \
		--require-validation-passed

# Show draft release metadata for a final read-only human sanity check.
inspect-booster-pack-release: ## Inspect draft Booster Pack release before publishing
	gh release view "$(BOOSTER_RELEASE_TAG)" \
		--repo NVIDIA/CompileIQ \
		--json tagName,name,isDraft,isPrerelease,targetCommitish,assets,body \
		--template 'tag: {{.tagName}}{{"\n"}}title: {{.name}}{{"\n"}}draft: {{.isDraft}}{{"\n"}}prerelease: {{.isPrerelease}}{{"\n"}}target: {{.targetCommitish}}{{"\n"}}assets:{{"\n"}}{{range .assets}}  - {{.name}}{{"\n"}}{{end}}{{"\n"}}body:{{"\n"}}{{.body}}{{"\n"}}'

# Publish the validated draft without changing GitHub "Latest". Repositories
# with immutable releases seal a release as it is published, so both fields
# must be updated in the same API call.
publish-booster-pack-release: ## Publish draft Booster Pack release with make_latest=false
	@set -eu; \
	if [ "$(CONFIRM_PUBLISH_RELEASE)" != "false" ]; then \
		printf 'Type %s to publish: ' "$(BOOSTER_RELEASE_TAG)"; \
		read confirmation; \
		if [ "$$confirmation" != "$(BOOSTER_RELEASE_TAG)" ]; then \
			echo "ERROR: confirmation did not match $(BOOSTER_RELEASE_TAG); release was not published." >&2; \
			exit 1; \
		fi; \
	fi; \
	release_id="$$(gh release view "$(BOOSTER_RELEASE_TAG)" --repo NVIDIA/CompileIQ --json databaseId --jq .databaseId)"; \
	test -n "$$release_id" || (echo "ERROR: could not find release $(BOOSTER_RELEASE_TAG)" >&2; exit 1); \
	gh api --method PATCH "repos/NVIDIA/CompileIQ/releases/$$release_id" \
		-F draft=false \
		-f make_latest=false \
		--silent
	@$(MAKE) --no-print-directory check-booster-pack-published BOOSTER_RELEASE_TAG="$(BOOSTER_RELEASE_TAG)"
	@echo "PASS: Published $(BOOSTER_RELEASE_TAG) with make_latest=false."

clear-booster-pack-latest: ## Clear GitHub Latest marker from a Booster Pack release
	@set -eu; \
	release_id="$$(gh release view "$(BOOSTER_RELEASE_TAG)" --repo NVIDIA/CompileIQ --json databaseId --jq .databaseId)"; \
	test -n "$$release_id" || (echo "ERROR: could not find release $(BOOSTER_RELEASE_TAG)" >&2; exit 1); \
	gh api --method PATCH "repos/NVIDIA/CompileIQ/releases/$$release_id" \
		-f make_latest=false \
		--silent
	@$(MAKE) --no-print-directory check-booster-pack-published BOOSTER_RELEASE_TAG="$(BOOSTER_RELEASE_TAG)"
	@echo "PASS: Cleared GitHub Latest marker for $(BOOSTER_RELEASE_TAG)."

# Confirm the published release is no longer a draft and is not GitHub "Latest".
check-booster-pack-published: ## Confirm published Booster Pack release is not Latest
	@status="$$(gh release list --repo NVIDIA/CompileIQ --limit 100 --json tagName,isDraft,isLatest --jq '.[] | select(.tagName == "$(BOOSTER_RELEASE_TAG)") | [.isDraft, .isLatest] | @tsv')"; \
	test -n "$$status" || (echo "ERROR: could not find release $(BOOSTER_RELEASE_TAG)" >&2; exit 1); \
	set -- $$status; \
	if [ "$$1" = "false" ] && [ "$$2" = "false" ]; then \
		echo "PASS: $(BOOSTER_RELEASE_TAG) is published and is not marked Latest."; \
	else \
		echo "FAIL: $(BOOSTER_RELEASE_TAG) has isDraft=$$1 isLatest=$$2." >&2; \
		exit 1; \
	fi
