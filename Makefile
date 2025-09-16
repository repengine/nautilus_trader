# Variables
# -----------------------------------------------------------------------------
PROJECT?=nautechsystems/nautilus_trader
REGISTRY?=ghcr.io/
IMAGE?=$(REGISTRY)$(PROJECT)
GIT_TAG:=$(shell git rev-parse --abbrev-ref HEAD)
IMAGE_FULL?=$(IMAGE):$(GIT_TAG)

V = 0  # 0 / 1 - verbose mode
Q = $(if $(filter 1,$V),,@) # Quiet mode, suppress command output
M = $(shell printf "$(BLUE)>$(RESET)") # Message prefix for commands

# Verbose options for specific targets (defaults to true, can be overridden)
VERBOSE ?= true

# FAIL_FAST controls whether `cargo nextest` should stop after the first test
# failure. When set to `true` the `--no-fail-fast` flag is omitted so tests
# abort on the first failure. When `false` (the default) the flag is included
# allowing the full test suite to run.
FAIL_FAST ?= false

# Select the appropriate flag for `cargo nextest` depending on FAIL_FAST.
ifeq ($(FAIL_FAST),true)
FAIL_FAST_FLAG :=
else
FAIL_FAST_FLAG := --no-fail-fast
endif

# > Colors
RED    := $(shell tput -Txterm setaf 1)
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
BLUE   := $(shell tput -Txterm setaf 4)
PURPLE := $(shell tput -Txterm setaf 5)
CYAN   := $(shell tput -Txterm setaf 6)
GRAY   := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)

.DEFAULT_GOAL := help

#== Installation

.PHONY: install
install:  #-- Install in release mode with all dependencies and extras
	$(info $(M) Installing Nautilus Trader in release mode with all dependencies and extras...)
	$Q BUILD_MODE=release uv sync --active --all-groups --all-extras --verbose

.PHONY: install-debug
install-debug:  #-- Install in debug mode for development
	$(info $(M) Installing Nautilus Trader in debug mode for development...)
	$Q BUILD_MODE=debug uv sync --active --all-groups --all-extras --verbose

.PHONY: install-just-deps
install-just-deps:  #-- Install dependencies only without building the package
	$(info $(M) Installing dependencies only without building the package...)
	$Q uv sync --active --all-groups --all-extras --no-install-package nautilus_trader

# == Poetry-backed installs (use Poetry as the source of truth for groups)
.PHONY: install-poetry-tests
install-poetry-tests:  #-- Install dev+test groups via Poetry into .venv (preferred for full pytest)
	$(info $(M) Installing dev+test groups with Poetry into .venv...)
	$Q POETRY_VIRTUALENVS_IN_PROJECT=true poetry install --with dev,test

.PHONY: install-poetry-all
install-poetry-all:  #-- Install main+dev+test (and extras) via Poetry into .venv
	$(info $(M) Installing all groups with Poetry into .venv...)
	$Q POETRY_VIRTUALENVS_IN_PROJECT=true poetry install --with dev,test --all-extras

#== Build

.PHONY: build
build:  #-- Build the package in release mode
	BUILD_MODE=release uv run --active --no-sync build.py

.PHONY: build-debug
build-debug:  #-- Build the package in debug mode (recommended for development)
ifeq ($(VERBOSE),true)
	$(info $(M) Building in debug mode with verbose output...)
	BUILD_MODE=debug uv run --active --no-sync build.py
else
	$(info $(M) Building in debug mode (errors will still be shown)...)
	BUILD_MODE=debug uv run --active --no-sync build.py 2>&1 | grep -E "(Error|error|ERROR|Failed|failed|FAILED|Warning|warning|WARNING|Build completed|Build time:|Traceback)" || true
endif

.PHONY: build-debug-pyo3
build-debug-pyo3:  #-- Build the package with PyO3 debug symbols (for debugging Rust code)
ifeq ($(VERBOSE),true)
	$(info $(M) Building in debug mode with PyO3 debug symbols...)
	BUILD_MODE=debug-pyo3 uv run --active --no-sync build.py
else
	$(info $(M) Building in debug mode with PyO3 debug symbols (errors will still be shown)...)
	BUILD_MODE=debug-pyo3 uv run --active --no-sync build.py 2>&1 | grep -E "(Error|error|ERROR|Failed|failed|FAILED|Warning|warning|WARNING|Build completed|Build time:|Traceback)" || true
endif

.PHONY: build-wheel
build-wheel:  #-- Build wheel distribution in release mode
	BUILD_MODE=release uv build --wheel

.PHONY: build-wheel-debug
build-wheel-debug:  #-- Build wheel distribution in debug mode
	BUILD_MODE=debug uv build --wheel

.PHONY: build-dry-run
build-dry-run:  #-- Show build commands without executing them
	DRY_RUN=true uv run --active --no-sync build.py

#== Clean

.PHONY: clean
clean: clean-build-artifacts clean-caches clean-builds  #-- Clean all build artifacts, caches, and builds

.PHONY: clean-builds
clean-builds:  #-- Clean distribution and target directories
	$Q rm -rf dist target 2>/dev/null || true

.PHONY: clean-build-artifacts
clean-build-artifacts:  #-- Clean compiled artifacts (.so, .dll, .pyc files)
	@echo "Cleaning build artifacts..."
	# Clean Rust build artifacts (keep final libraries)
	find target -name "*.rlib" -delete 2>/dev/null || true
	find target -name "*.rmeta" -delete 2>/dev/null || true
	rm -rf target/*/build target/*/deps 2>/dev/null || true
	# Clean Python build artifacts
	rm -rf build/ 2>/dev/null || true
	find . -type d -name "__pycache__" -not -path "./.venv*" -print0 | xargs -0 rm -rf
	find . -type f -a \( -name "*.pyc" -o -name "*.pyo" \) -not -path "./.venv*" -print0 | xargs -0 rm -f
	find . -type f -a \( -name "*.so" -o -name "*.dll" -o -name "*.dylib" \) -not -path "./.venv*" -print0 | xargs -0 rm -f
	# Clean test artifacts
	rm -rf .coverage .benchmarks 2>/dev/null || true

.PHONY: clean-caches
clean-caches:  #-- Clean pytest, mypy, ruff, uv, and cargo caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache 2>/dev/null || true
	-uv cache prune
	-cargo clean

.PHONY: distclean
distclean: clean  #-- Nuclear clean - remove all untracked files (requires FORCE=1)
	@[ "$$FORCE" = 1 ] || { echo "Pass FORCE=1 to really nuke"; exit 1; }
	@echo "⚠️  nuking working tree (git clean -fxd)…"
	git clean -fxd -e tests/test_data/large/ -e .venv

#== Code Quality

.PHONY: format
format:  #-- Format Rust code using nightly formatter
	cargo +nightly fmt

.PHONY: pre-commit
pre-commit:  #-- Run all pre-commit hooks on all files
	uv run --active --no-sync pre-commit run --all-files

#== Database Migrations (ML)

.PHONY: db-preflight-dedupe
db-preflight-dedupe:  #-- Check duplicates in feature values before adding unique index
	$(info $(M) Running preflight duplicate detection for ml_feature_values...)
	@[ -n "$$DATABASE_URL" ] || (echo "ERROR: DATABASE_URL is not set" && exit 1)
	psql $$DATABASE_URL -f ml/stores/migrations/005a_feature_values_dedupe.sql
	$(info $(YELLOW)Review results before enabling delete block in the SQL file.$(RESET))

.PHONY: db-migrate-hardening
db-migrate-hardening:  #-- Apply schema hardening migrations (unique keys, created_at types, views)
	$(info $(M) Applying schema hardening migrations...)
	@[ -n "$$DATABASE_URL" ] || (echo "ERROR: DATABASE_URL is not set" && exit 1)
	psql $$DATABASE_URL -f ml/stores/migrations/005_schema_hardening.sql
	psql $$DATABASE_URL -f ml/stores/migrations/005_views.sql
	$(info $(GREEN)DB hardening migrations applied.$(RESET))

.PHONY: db-migrate-cli
db-migrate-cli:  #-- Apply ML DB migrations via CLI runner (use DATABASE_URL, optional FULL=1, SCHEMA=stores|registry|both)
	$(info $(M) Applying ML DB migrations via CLI runner...)
	$Q uv run --active --no-sync python -m ml.scripts.apply_migrations $(if $(DATABASE_URL),--db-url $(DATABASE_URL)) $(if $(FULL),--full,) $(if $(SCHEMA),--schema $(SCHEMA),)

.PHONY: db-migrate-cli-print
db-migrate-cli-print:  #-- Show planned migration files without executing
	$Q uv run --active --no-sync python -m ml.scripts.apply_migrations --print-only

.PHONY: db-migrate-cli-dry-run
db-migrate-cli-dry-run:  #-- Dry-run migrations (no execution)
	$Q uv run --active --no-sync python -m ml.scripts.apply_migrations $(if $(DATABASE_URL),--db-url $(DATABASE_URL)) --dry-run $(if $(FULL),--full,) $(if $(SCHEMA),--schema $(SCHEMA),)

.PHONY: db-convert-stores-to-partitioned
db-convert-stores-to-partitioned:  #-- Convert non-partitioned ML store tables to partitioned parents (one-time)
	@[ -n "$$DATABASE_URL" ] || { echo "Provide DATABASE_URL=postgresql://..."; exit 1; }
	$(info $(M) Converting ML store tables to partitioned parents...)
	uv run --active --no-sync python -m ml.scripts.convert_stores_to_partitioned \
	  --db-url $(DATABASE_URL) $(if $(TABLES),--tables $(TABLES),) $(if $(AHEAD),--ahead $(AHEAD),)

.PHONY: ruff
ruff:  #-- Run ruff linter with automatic fixes (ML package only)
	uv run --active --no-sync ruff check ml --fix

.PHONY: validate-metrics
validate-metrics:  #-- Validate metrics bootstrap usage (no direct prometheus collectors)
	uv run --active --no-sync python tools/validate_metrics_bootstrap.py

.PHONY: validate-events
validate-events:  #-- Validate canonical event stage constants usage
	uv run --active --no-sync python tools/validate_event_constants.py

.PHONY: validate-nautilus-patterns
validate-nautilus-patterns:  #-- Run extended ML validation suite (patterns, semgrep, import-linter, duplication, xenon)
	$(info $(M) Running ML validation suite...)
	uv run --active --no-sync python .pre-commit-hooks/check_nautilus_patterns.py $$(find ml -name "*.py" -not -path "ml/tests/*" -not -name "test_*.py") || true
	uv run --active --no-sync semgrep --config tools/semgrep/ml-rules.yml --error || true
	uv run --active --no-sync python tools/duplication/check_duplication.py || true
	uv run --active --no-sync lint-imports --config importlinter.ini || true
	uv run --active --no-sync xenon --max-absolute B --max-modules B --max-average B ml/ || true
	uv run --active --no-sync bandit -q -r ml -x ml/tests || true
	uv run --active --no-sync vulture ml --min-confidence 90 --exclude ml/tests || true
	@echo "Validation suite complete (non-blocking). To enforce, run via pre-commit hooks."

#== ML Orchestrator

.PHONY: ml-pipeline-orchestrator
ml-pipeline-orchestrator:  #-- Run cold-path pipeline: optional ingestion + dataset build + optional HPO + teacher train
	$(info $(M) Running ML pipeline orchestrator...)
	$Q uv run --active --no-sync python -m ml.cli.pipeline_orchestrator \
		$(if $(INGEST),--ingest,) \
		$(if $(DATASET_ID),--dataset_id $(DATASET_ID),) \
		$(if $(SCHEMA),--schema $(SCHEMA),) \
		$(if $(INSTRUMENTS),--instruments $(INSTRUMENTS),) \
		$(if $(LOOKBACK_DAYS),--lookback_days $(LOOKBACK_DAYS),) \
		$(if $(COVERAGE_MODE),--coverage_mode $(COVERAGE_MODE),) \
		$(if $(CATALOG_PATH),--catalog_path $(CATALOG_PATH),) \
		$(if $(DB),--db $(DB),) \
		$(if $(WRITE_MODE),--write_mode $(WRITE_MODE),) \
		$(if $(DATA_DIR),--data_dir $(DATA_DIR),) \
		$(if $(SYMBOLS),--symbols $(SYMBOLS),) \
		$(if $(OUT_DIR),--out_dir $(OUT_DIR),) \
		$(if $(INCLUDE_MACRO),--include_macro,) \
		$(if $(MACRO_LAG_DAYS),--macro_lag_days $(MACRO_LAG_DAYS),) \
		$(if $(INCLUDE_MICRO),--include_micro,) \
		$(if $(INCLUDE_L2),--include_l2,) \
		$(if $(HORIZON_MINUTES),--horizon_minutes $(HORIZON_MINUTES),) \
		$(if $(THRESHOLD),--threshold $(THRESHOLD),) \
		$(if $(LOOKBACK_PERIODS),--lookback_periods $(LOOKBACK_PERIODS),) \
		$(if $(HPO),--hpo,) \
		$(if $(HPO_EPOCHS),--hpo_epochs $(HPO_EPOCHS),) \
		$(if $(HPO_BATCH_SIZE),--hpo_batch_size $(HPO_BATCH_SIZE),) \
		$(if $(HPO_TAIL_ROWS),--hpo_tail_rows $(HPO_TAIL_ROWS),) \
		$(if $(HPO_LIMIT_GROUPS),--hpo_limit_groups $(HPO_LIMIT_GROUPS),) \
		$(if $(TRAIN),--train,) \
		$(if $(TEACHER_MODEL_ID),--teacher_model_id $(TEACHER_MODEL_ID),) \
		$(if $(FEATURE_REGISTRY_DIR),--feature_registry_dir $(FEATURE_REGISTRY_DIR),) \
		$(if $(FEATURE_SET_ID),--feature_set_id $(FEATURE_SET_ID),) \
		$(if $(MAX_EPOCHS),--max_epochs $(MAX_EPOCHS),)

.PHONY: ml-pipeline-scheduler
ml-pipeline-scheduler:  #-- Run the scheduler for the cold-path orchestrator (env-driven)
	$(info $(M) Starting ML pipeline scheduler...)
	$Q uv run --active --no-sync python -m ml.cli.pipeline_scheduler \
		$(if $(SCHEDULE_TIME),--schedule-time $(SCHEDULE_TIME),) \
		$(if $(INTERVAL_MIN),--interval-min $(INTERVAL_MIN),) \
		$(if $(ORCH_CONFIG),--config $(ORCH_CONFIG),) \
		$(if $(DRY_RUN),--dry-run,) \
		$(if $(FORCE),--force,)

.PHONY: ml-pipeline-scheduler-example
ml-pipeline-scheduler-example:  #-- Run scheduler with example TOML and 24h interval (dry run via DRY_RUN=1)
	$(info $(M) Running scheduler with example config...)
	$Q $(MAKE) ml-pipeline-scheduler \
		ORCH_CONFIG=ml/config/pipeline_scheduler_example.toml \
		INTERVAL_MIN=1440 $(if $(DRY_RUN),DRY_RUN=$(DRY_RUN),DRY_RUN=1)

.PHONY: ml-scheduler-smoke
ml-scheduler-smoke:  #-- CI one-shot smoke run (forces dummy integration)
	$(info $(M) Running scheduler smoke...)
	$Q ML_ALLOW_DUMMY=$(if $(ML_ALLOW_DUMMY),$(ML_ALLOW_DUMMY),1) \
		uv run --active --no-sync python -m ml.cli.scheduler_smoke \
		$(if $(ORCH_CONFIG),--config $(ORCH_CONFIG),) \
		$(if $(DRY_RUN),--dry-run,)

.PHONY: clippy
clippy:  #-- Run Rust clippy linter with fixes
	cargo clippy --fix --all-targets --all-features -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::unwrap_used -W clippy::expect_used

.PHONY: clippy-nightly
clippy-nightly:  #-- Run Rust clippy linter with nightly toolchain
	cargo +nightly clippy --fix --all-targets --all-features --allow-dirty --allow-staged -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::unwrap_used -W clippy::expect_used

.PHONY: clippy-crate-%
clippy-crate-%:  #-- Run clippy for a specific Rust crate (usage: make clippy-crate-<crate_name>)
	cargo clippy --all-targets --all-features -p $* -- -D warnings

#== Dependencies

.PHONY: outdated
outdated:  #-- Check for outdated Rust dependencies
	cargo outdated

.PHONY: update cargo-update
update: cargo-update  #-- Update all dependencies (uv and cargo)
	uv self update
	uv lock --upgrade

#== Documentation

.PHONY: docs
docs: docs-python docs-rust  #-- Build all documentation (Python and Rust)

.PHONY: docs-python
docs-python:  #-- Build Python documentation with Sphinx
	BUILD_MODE=debug uv run --active sphinx-build -M markdown ./docs/api_reference ./api_reference

.PHONY: docs-rust
docs-rust:  #-- Build Rust documentation with cargo doc
	RUSTDOCFLAGS="--enable-index-page -Zunstable-options" cargo +nightly doc --all-features --no-deps --workspace

.PHONY: docsrs-check
docsrs-check: check-hack-installed #-- Check documentation builds for docs.rs compatibility
	RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo hack --workspace doc --no-deps --all-features

#== Rust Development

.PHONY: cargo-build
cargo-build:  #-- Build Rust crates in release mode
	cargo build --release --all-features

.PHONY: cargo-update
cargo-update:  #-- Update Rust dependencies and install test tools
	cargo update \
	&& cargo install cargo-nextest \
	&& cargo install cargo-llvm-cov

.PHONY: cargo-check
cargo-check:  #-- Check Rust code without building
	cargo check --workspace --all-features

.PHONY: check-nextest-installed
check-nextest-installed:  #-- Verify cargo-nextest is installed
	@if ! cargo nextest --version >/dev/null 2>&1; then \
		echo "cargo-nextest is not installed. You can install it using 'cargo install cargo-nextest'"; \
		exit 1; \
	fi

.PHONY: check-llvm-cov-installed
check-llvm-cov-installed:  #-- Verify cargo-llvm-cov is installed
	@if ! cargo llvm-cov --version >/dev/null 2>&1; then \
		echo "cargo-llvm-cov is not installed. You can install it using 'cargo install cargo-llvm-cov'"; \
		exit 1; \
	fi

.PHONY: check-hack-installed
check-hack-installed:  #-- Verify cargo-hack is installed
	@if ! cargo hack --version >/dev/null 2>&1; then \
		echo "cargo-hack is not installed. You can install it using 'cargo install cargo-hack'"; \
		exit 1; \
	fi

.PHONY: check-features  #-- Verify crate feature combinations compile correctly
check-features: check-hack-installed
	cargo hack check --each-feature

#== Rust Testing

.PHONY: cargo-test
cargo-test: RUST_BACKTRACE=1
cargo-test: HIGH_PRECISION=true
cargo-test: check-nextest-installed
cargo-test:  #-- Run all Rust tests with ffi,python,high-precision,defi features
ifeq ($(VERBOSE),true)
	$(info $(M) Running Rust tests with verbose output...)
	cargo nextest run --workspace --features "ffi,python,high-precision,defi" $(FAIL_FAST_FLAG) --cargo-profile nextest --verbose
else
	$(info $(M) Running Rust tests (showing summary and failures only)...)
	cargo nextest run --workspace --features "ffi,python,high-precision,defi" $(FAIL_FAST_FLAG) --cargo-profile nextest --status-level fail --final-status-level flaky
endif

.PHONY: cargo-test-lib
cargo-test-lib: RUST_BACKTRACE=1
cargo-test-lib: HIGH_PRECISION=true
cargo-test-lib: check-nextest-installed
cargo-test-lib:  #-- Run Rust library tests only with high precision
	cargo nextest run --lib --workspace --no-default-features --features "ffi,python,high-precision,defi,stubs" $(FAIL_FAST_FLAG) --cargo-profile nextest

.PHONY: cargo-test-standard-precision
cargo-test-standard-precision: RUST_BACKTRACE=1
cargo-test-standard-precision: HIGH_PRECISION=false
cargo-test-standard-precision: check-nextest-installed
cargo-test-standard-precision:  #-- Run Rust tests with standard precision (64-bit)
	cargo nextest run --workspace --features "ffi,python" $(FAIL_FAST_FLAG) --cargo-profile nextest

.PHONY: cargo-test-debug
cargo-test-debug: RUST_BACKTRACE=1
cargo-test-debug: HIGH_PRECISION=true
cargo-test-debug: check-nextest-installed
cargo-test-debug:  #-- Run Rust tests in debug mode with high precision
	cargo nextest run --workspace --features "ffi,python,high-precision,defi" $(FAIL_FAST_FLAG)

.PHONY: cargo-test-standard-precision-debug
cargo-test-standard-precision-debug: RUST_BACKTRACE=1
cargo-test-standard-precision-debug: HIGH_PRECISION=false
cargo-test-standard-precision-debug: check-nextest-installed
cargo-test-standard-precision-debug:  #-- Run Rust tests in debug mode with standard precision
	cargo nextest run --workspace --features "ffi,python"

.PHONY: cargo-test-coverage
cargo-test-coverage: check-nextest-installed check-llvm-cov-installed
cargo-test-coverage:  #-- Run Rust tests with coverage reporting
	cargo llvm-cov nextest run --workspace

# -----------------------------------------------------------------------------
# Library tests for a single crate
# -----------------------------------------------------------------------------
# Invoke as:
#   make cargo-test-crate-<crate_name>
# Examples:
#   make cargo-test-crate-nautilus-model
#   make cargo-test-crate-nautilus-core FEATURES="python,ffi"
#
# This reuses the same flags as `cargo-test-lib` but targets only the specified
# crate by replacing `--workspace` with `-p <crate>`.
# To include specific features, use the FEATURES variable with comma-separated values.
# -----------------------------------------------------------------------------

.PHONY: cargo-test-crate-%
cargo-test-crate-%: RUST_BACKTRACE=1
cargo-test-crate-%: HIGH_PRECISION=true
cargo-test-crate-%: check-nextest-installed
cargo-test-crate-%:  #-- Run Rust tests for a specific crate (usage: make cargo-test-crate-<crate_name>)
	cargo nextest run --lib $(FAIL_FAST_FLAG) --cargo-profile nextest -p $* $(if $(FEATURES),--features "$(FEATURES)")

.PHONY: cargo-test-coverage-crate-%
cargo-test-coverage-crate-%: RUST_BACKTRACE=1
cargo-test-coverage-crate-%: HIGH_PRECISION=true
cargo-test-coverage-crate-%: check-nextest-installed check-llvm-cov-installed
cargo-test-coverage-crate-%:  #-- Run Rust tests with coverage reporting for a specific crate (usage: make cargo-test-coverage-crate-<crate_name>)
	cargo llvm-cov nextest --lib $(FAIL_FAST_FLAG) --cargo-profile nextest -p $* $(if $(FEATURES),--features "$(FEATURES)")

#------------------------------------------------------------------------------
# Benchmarks
#------------------------------------------------------------------------------

# List of crates whose criterion/iai benches run in the performance workflow
CI_BENCH_CRATES := nautilus-core nautilus-model nautilus-common nautilus-live

# NOTE:
# - We invoke `cargo bench` *once per crate* to avoid the well-known
#   "mixed panic strategy" linker error that appears when crates which specify
#   different `panic` strategies (e.g. `abort` for cdylib/staticlib targets vs
#   `unwind` for Criterion) are linked into the *same* benchmark binary.
# - Cargo will still reuse compiled artifacts between iterations, so the cost
#   of the extra invocations is marginal while the linker remains happy.

.PHONY: cargo-ci-benches
cargo-ci-benches:  #-- Run Rust benches for the crates included in the CI performance workflow
	@for crate in $(CI_BENCH_CRATES); do \
	  echo "Running benches for $$crate"; \
	  cargo bench -p $$crate --profile bench --benches --no-fail-fast; \
	done

#== Docker

.PHONY: docker-build
docker-build: clean  #-- Build Docker image for NautilusTrader
	docker pull $(IMAGE_FULL) || docker pull $(IMAGE):nightly || true
	docker build -f .docker/nautilus_trader.dockerfile --platform linux/x86_64 -t $(IMAGE_FULL) .

.PHONY: docker-build-force
docker-build-force:  #-- Force rebuild Docker image without cache
	docker build --no-cache -f .docker/nautilus_trader.dockerfile -t $(IMAGE_FULL) .

.PHONY: docker-push
docker-push:  #-- Push Docker image to registry
	docker push $(IMAGE_FULL)

.PHONY: docker-build-jupyter
docker-build-jupyter:  #-- Build JupyterLab Docker image
	docker build --build-arg GIT_TAG=$(GIT_TAG) -f .docker/jupyterlab.dockerfile --platform linux/x86_64 -t $(IMAGE):jupyter .

.PHONY: docker-push-jupyter
docker-push-jupyter:  #-- Push JupyterLab Docker image to registry
	docker push $(IMAGE):jupyter

.PHONY: init-services
init-services:  #-- Initialize development services eg. for integration tests (start containers and setup database)
	$(info $(M) Initializing development services...)
	@$(MAKE) start-services
	@echo "${PURPLE}Waiting for PostgreSQL to be ready...${RESET}"
	@sleep 10
	@$(MAKE) init-db

.PHONY: start-services
start-services:  #-- Start development services (without reinitializing database)
	$(info $(M) Starting development services...)
	docker compose -f .docker/docker-compose.yml up -d

.PHONY: stop-services
stop-services:  #-- Stop development services (preserves data)
	$(info $(M) Stopping development services...)
	docker compose -f .docker/docker-compose.yml down

.PHONY: purge-services
purge-services:  #-- Purge all development services (stop containers and remove volumes)
	$(info $(M) Purging integration test services...)
	docker compose -f .docker/docker-compose.yml down -v

.PHONY: init-db
init-db:  #-- Initialize PostgreSQL database schema
	$(info $(M) Initializing PostgreSQL database schema...)
	cat schema/sql/*.sql | docker exec -i nautilus-database psql -U nautilus -d nautilus

#== Test DB (PostgreSQL) helpers

.PHONY: docker-up-test
docker-up-test:  #-- Start PostgreSQL for tests (defaults match StrategyStore)
	$(info $(M) Starting PostgreSQL for tests...)
	POSTGRES_USER=postgres POSTGRES_PASSWORD=postgres POSTGRES_DB=nautilus \
		docker compose -f .docker/docker-compose.yml up -d postgres
	$(info $(M) Waiting for PostgreSQL to be ready...)
	DATABASE_URL=postgresql://postgres:postgres@localhost:5432/nautilus \
		uv run --active --no-sync python tools/wait_for_postgres.py

.PHONY: check-db
check-db:  #-- Check DATABASE_URL is reachable (waits up to DB_WAIT_TIMEOUT)
	$(info $(M) Checking PostgreSQL readiness via $$DATABASE_URL...)
	uv run --active --no-sync python tools/wait_for_postgres.py

.PHONY: docker-down-test
docker-down-test:  #-- Stop PostgreSQL test container and remove volumes
	$(info $(M) Stopping PostgreSQL test container...)
	docker compose -f .docker/docker-compose.yml down -v

.PHONY: pytest-ml-db
pytest-ml-db:  #-- Run ML tests requiring DB with coverage (FAST=1 skips heavy/legacy)
	$(info $(M) Running ML DB tests with coverage...)
	DATABASE_URL=postgresql://postgres:postgres@localhost:5432/nautilus \
		uv run --active --no-sync pytest -n logical --dist=loadgroup \
		--cov=ml --cov=nautilus_trader --cov-report=term-missing \
		$(if $(FAST),-k "not tft and not stores_concurrency and not stores_integration",-k "not tft") \
		-v ml/tests

.PHONY: pytest-ml-fast
pytest-ml-fast:  #-- Quick ML test run (smoke + unit + core + actors + features)
	$(info $(M) Running quick ML smoke/unit tests...)
	uv run --active --no-sync pytest -q \
		ml/tests/unit \
		ml/tests/features \
		-k "not integration and not performance and not slow" || exit $$?
	$(info $(M) Running fast ML test subset...)
	uv run --active --no-sync pytest -n auto --dist=loadfile \
		-k "smoke or unit or actors or features or core or EngineManagerIntegration" -v ml/tests

.PHONY: pytest-ml-coverage
pytest-ml-coverage:  #-- Run ML tests with coverage (exclude perf), then guardrails baseline
	# Coverage run (exclude performance/prototype to keep stable and fast); continue to guardrails
	@status=0; \
	PYTHONWARNINGS="ignore:pkg_resources is deprecated as an API.*:UserWarning${PYTHONWARNINGS:+,$(PYTHONWARNINGS)}" \
	uv run --active --no-sync pytest -n logical --dist=loadgroup \
		--cov=ml --cov=nautilus_trader --cov-report=term-missing \
		-m "not performance and not prototype and not slow" -k "not tft" -v ml/tests || status=$$?; \
	echo "Coverage run exit status: $$status"; \
	ML_REPORT_FILE=ml/tests/validation_reports/performance-guardrails.json $(MAKE) pytest-ml-guardrails || status=$$?; \
	echo "Guardrails run exit status: $$status"; \
	exit $$status

#== Python Testing

.PHONY: pytest
pytest:  #-- Run Python tests with pytest in parallel with immediate failure reporting
	$(info $(M) Running Python tests in parallel with immediate failure reporting...)
	uv run --active --no-sync pytest --new-first --failed-first --tb=line -n logical --dist=loadgroup --maxfail=50 --durations=0 --durations-min=10.0 $(if $(filter true,$(VERBOSE)),-v,)

.PHONY: pytest-green
pytest-green:  #-- Run correctness lane only (no perf/real API), with coverage gate
	$(info $(M) Running green lane (correctness only) with coverage gate...)
	uv run --active --no-sync pytest -q ml/tests \
		-m "not integration and not performance and not real_api and not prototype" \
		-n auto --dist=loadscope \
		-k "not microbench and not performance and not integration and not real_api and not strategies" \
		--cov=ml --cov-report=term-missing:skip-covered --cov-fail-under=43

.PHONY: pytest-perf
pytest-perf:  #-- Run microbench performance lane (non-blocking)
	$(info $(M) Running performance microbench lane...)
	ML_BENCH_RELAX?=3 uv run --active --no-sync pytest -q ml/tests/performance -k microbench --benchmark-only || true

.PHONY: pytest-real-api
pytest-real-api:  #-- Run real API tests (Databento/FRED) when keys are present
	$(info $(M) Running real API tests when keys are set...)
	@[ -n "$$DATABENTO_API_KEY" ] || echo "Warning: DATABENTO_API_KEY not set; tests may skip";
	@[ -n "$$FRED_API_KEY" ] || echo "Warning: FRED_API_KEY not set; tests may skip";
	uv run --active --no-sync pytest -q ml -m real_api

.PHONY: pytest-memory-tracking
pytest-memory-tracking:  #-- Run Python tests with memory tracking enabled
	$(info $(M) Running Python tests with memory tracking enabled...)
	MEMORY_TRACKING_ENABLED_PY=true uv run --active --no-sync pytest --new-first --failed-first -v -n logical --dist=loadgroup

.PHONY: test-performance
test-performance:  #-- Run performance tests with codspeed benchmarking
	uv run --active --no-sync pytest tests/performance_tests --benchmark-disable-gc --codspeed

#== ML Development

.PHONY: update-ml-baseline
update-ml-baseline:  #-- Update ML performance baseline after optimization
	@echo "Updating ML performance baseline..."
	@python .pre-commit-hooks/check_ml_performance.py --update-baseline
	@echo "Baseline updated in .ml_performance_baseline.json"

.PHONY: test-feature-parity
test-feature-parity:  #-- Run ML feature parity tests manually
	@python -m pytest tests/test_feature_parity.py -v

.PHONY: benchmark-ml
benchmark-ml:  #-- Run ML performance benchmarks manually
	@python benchmarks/ml_performance.py --report

#== ML Deployment (Docker Compose)

.PHONY: _compose_args
_compose_args:
	@true

# Compose files (include override if present)
COMPOSE_BASE=ml/deployment/docker-compose.yml
COMPOSE_OVERRIDE=ml/deployment/docker-compose.override.yml
ifneq (,$(wildcard $(COMPOSE_OVERRIDE)))
COMPOSE_ARGS=-f $(COMPOSE_BASE) -f $(COMPOSE_OVERRIDE)
else
COMPOSE_ARGS=-f $(COMPOSE_BASE)
endif

.PHONY: ml-up
ml-up:  #-- Bring up ML stack (postgres, redis, ml_pipeline, grafana, prometheus)
	$(info $(M) Starting ML stack with Docker Compose (project name 'ml')...)
	docker compose $(COMPOSE_ARGS) up -d

.PHONY: ml-up-core
ml-up-core:  #-- Bring up core ML services without building optional images
	$(info $(M) Starting core ML services (postgres, redis, ml_pipeline, grafana, prometheus) -- no build...)
	docker compose $(COMPOSE_ARGS) up -d --no-build postgres redis ml_pipeline prometheus grafana

.PHONY: ml-down
ml-down:  #-- Bring down ML stack and remove volumes
	$(info $(M) Stopping ML stack and removing volumes...)
	docker compose $(COMPOSE_ARGS) down -v

.PHONY: ml-logs
ml-logs:  #-- Tail logs for ml_pipeline service (CTRL-C to exit)
	$(info $(M) Tailing ml_pipeline logs...)
	docker compose $(COMPOSE_ARGS) logs -f ml_pipeline

.PHONY: ml-ps
ml-ps:  #-- Show status of ML services
	docker compose $(COMPOSE_ARGS) ps

.PHONY: ml-migrate
ml-migrate:  #-- Apply ML database migrations via docker compose exec
	$(info $(M) Applying ML DB migrations via docker compose exec postgres...)
	uv run --active --no-sync python -m ml.deployment.migrations --apply --compose-file $(COMPOSE_BASE)

#== CLI Tools

.PHONY: install-cli
install-cli:  #-- Install Nautilus CLI tool from source
	cargo install --path crates/cli --bin nautilus --force

.PHONY: ml-build-runner
ml-build-runner:  #-- Run per-symbol dataset builds from a JSON/TOML config (CONFIG=path)
	@[ -n "$(CONFIG)" ] || { echo "Provide CONFIG=/path/to/config.{json,toml}"; exit 1; }
	uv run --active --no-sync python -m ml.pipelines.build_runner --config $(CONFIG)

.PHONY: ml-dataset-report
ml-dataset-report:  #-- Generate dataset report (DATASET=parquet|csv, OUT_JSON, OUT_MD optional)
	@[ -n "$(DATASET)" ] || { echo "Provide DATASET=/path/to/dataset.parquet"; exit 1; }
	uv run --active --no-sync python -m ml.scripts.dataset_report --dataset $(DATASET) $(if $(OUT_JSON),--out_json $(OUT_JSON),) $(if $(OUT_MD),--out_md $(OUT_MD),)

.PHONY: ml-promote-features
ml-promote-features:  #-- Promote features via gates (FEATURE_REGISTRY_DIR, FEATURE_SET_ID, METRICS_JSON, GATES or GATES_JSON)
	@[ -n "$(FEATURE_REGISTRY_DIR)" ] || { echo "FEATURE_REGISTRY_DIR is required"; exit 1; }
	@[ -n "$(FEATURE_SET_ID)" ] || { echo "FEATURE_SET_ID is required"; exit 1; }
	@[ -n "$(METRICS_JSON)" ] || { echo "METRICS_JSON is required"; exit 1; }
	uv run --active --no-sync python -m ml.scripts.promote_features \
	  --feature_registry_dir $(FEATURE_REGISTRY_DIR) \
	  --feature_set_id $(FEATURE_SET_ID) \
	  --metrics_json $(METRICS_JSON) \
	  $(if $(GATES_JSON),--gates_json $(GATES_JSON),) \
	  $(if $(GATE1),--gate $(GATE1) $(GATE2) $(GATE3) $(GATE4),)

#== Internal

.PHONY: help
help:  #-- Show this help message and exit
	@printf "Nautilus Trader Makefile\n\n"
	@printf "$(GREEN)Usage:$(RESET) make $(CYAN)<target>$(RESET)\n\n"
	@printf "$(GRAY)Tips: Use $(CYAN)make <target> V=1$(GRAY) for verbose output$(RESET)\n"
	@printf "$(GRAY)      Use $(CYAN)make <target> VERBOSE=false$(GRAY) to disable verbose output for build-debug, cargo-test, and pytest$(RESET)\n"
	@printf "$(GRAY)      Use $(CYAN)make pytest VERBOSE=true$(GRAY) to run tests with verbose output$(RESET)\n\n"

	@printf "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣴⣶⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
	@printf "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣾⣿⣿⣿⠀⢸⣿⣿⣿⣿⣶⣶⣤⣀⠀⠀⠀⠀⠀\n"
	@printf "⠀⠀⠀⠀⠀⠀⢀⣴⡇⢀⣾⣿⣿⣿⣿⣿⠀⣾⣿⣿⣿⣿⣿⣿⣿⠿⠓⠀⠀⠀⠀\n"
	@printf "⠀⠀⠀⠀⠀⣰⣿⣿⡀⢸⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⠟⠁⣠⣄⠀⠀⠀⠀\n"
	@printf "⠀⠀⠀⠀⢠⣿⣿⣿⣇⠀⢿⣿⣿⣿⣿⣿⠀⢻⣿⣿⣿⡿⢃⣠⣾⣿⣿⣧⡀⠀⠀\n"
	@printf "⠀⠀⠀⠠⣾⣿⣿⣿⣿⣿⣧⠈⠋⢀⣴⣧⠀⣿⡏⢠⡀⢸⣿⣿⣿⣿⣿⣿⣿⡇⠀\n"
	@printf "⠀⠀⠀⣀⠙⢿⣿⣿⣿⣿⣿⠇⢠⣿⣿⣿⡄⠹⠃⠼⠃⠈⠉⠛⠛⠛⠛⠛⠻⠇⠀\n"
	@printf "⠀⠀⢸⡟⢠⣤⠉⠛⠿⢿⣿⠀⢸⣿⡿⠋⣠⣤⣄⠀⣾⣿⣿⣶⣶⣶⣦⡄⠀⠀⠀\n"
	@printf "⠀⠀⠸⠀⣾⠏⣸⣷⠂⣠⣤⠀⠘⢁⣴⣾⣿⣿⣿⡆⠘⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀\n"
	@printf "⠀⠀⠀⠀⠛⠀⣿⡟⠀⢻⣿⡄⠸⣿⣿⣿⣿⣿⣿⣿⡀⠘⣿⣿⣿⣿⠟⠀⠀⠀⠀\n"
	@printf "⠀⠀⠀⠀⠀⠀⣿⠇⠀⠀⢻⡿⠀⠈⠻⣿⣿⣿⣿⣿⡇⠀⢹⣿⠿⠋⠀⠀⠀⠀⠀\n"
	@printf "⠀⠀⠀⠀⠀⠀⠋⠀⠀⠀⡘⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀\n"

	@awk '\
	BEGIN { FS = ":.*#--"; target_maxlen = 0 } \
	/^[$$()% a-zA-Z_-]+:.*?#--/ { \
		if (length($$1) > target_maxlen) target_maxlen = length($$1); \
		targets[NR] = $$1; descriptions[NR] = $$2; \
	} \
	/^#==/ { \
		groups[NR] = substr($$0, 5); \
	} \
	END { \
		for (i = 1; i <= NR; i++) { \
			if (groups[i]) { \
				printf "\n$(GREEN)%s:$(RESET)\n", groups[i]; \
			} else if (targets[i]) { \
				printf "  $(CYAN)%-*s$(RESET) %s\n", target_maxlen, targets[i], descriptions[i]; \
			} \
		} \
	}' $(MAKEFILE_LIST)

#== ML Property Testing

.PHONY: test-ml-properties
test-ml-properties:  #-- Run ML property-based tests with Hypothesis
	$(info $(M) Running ML property tests with Hypothesis...)
	HYPOTHESIS_PROFILE=ci uv run --active --no-sync pytest ml/tests/unit/*hypothesis*.py \
		--hypothesis-show-statistics \
		--tb=short \
		-v

.PHONY: test-ml-properties-debug
test-ml-properties-debug:  #-- Run ML property tests in debug mode (more examples)
	$(info $(M) Running ML property tests in debug mode...)
	HYPOTHESIS_PROFILE=debug uv run --active --no-sync pytest ml/tests/unit/*hypothesis*.py \
		--hypothesis-show-statistics \
		--hypothesis-seed=0 \
		-vv

.PHONY: test-ml-invariants
test-ml-invariants:  #-- Test critical ML invariants (RSI bounds, feature parity, etc)
	$(info $(M) Testing critical ML invariants...)
	uv run --active --no-sync pytest \
		ml/tests/unit/features/test_feature_engineering_hypothesis.py::TestFeatureEngineerProperties::test_rsi_bounds_property \
		ml/tests/unit/features/test_feature_engineering_hypothesis.py::TestFeatureEngineerProperties::test_feature_count_consistency \
		-v

.PHONY: ml-coverage
ml-coverage:  #-- Generate ML module coverage report with property tests
	$(info $(M) Generating ML coverage report...)
	uv run --active --no-sync pytest ml/tests/ \
		--cov=ml \
		--cov-report=term-missing \
		--cov-report=html:htmlcov/ml \
		--cov-fail-under=75
	@echo "Coverage report generated in htmlcov/ml/index.html"
sanity:
	@echo "Running ML codebase sanity sweep (advisory)..."
	@python ml/cli/sanity_check.py || true

.PHONY: pytest-ml

pytest-ml:  #-- Run ML tests optimized: parallel non-integration (no perf/real API), then serial integration (no real API)
	$(info $(M) Running ML tests: parallel non-integration (excl. perf/real_api), then serial integration (excl. real_api) ...)
	uv run --active --no-sync pytest -c ml/pytest.ini ml -m "not integration and not performance and not real_api" -q -n auto --dist=loadscope || exit $$?
	uv run --active --no-sync pytest -c ml/pytest.ini ml -m "integration and not real_api" -q -n 1 || exit $$?
	@echo "$(GREEN)ML tests completed$(RESET)"

.PHONY: pytest-ml-perf
pytest-ml-perf:  #-- Run ML performance tests with optional relax factor (ML_BENCH_RELAX)
	$(info $(M) Running ML performance tests...)
	@echo "Relax factor: $${ML_BENCH_RELAX:-1.0}"
	ML_BENCH_RELAX=$${ML_BENCH_RELAX:-1.0} uv run --active --no-sync pytest ml/tests/performance -q

.PHONY: pytest-ml-guardrails
pytest-ml-guardrails:  #-- Run ML performance guardrails (FAILS CI if regressions detected)
	$(info $(M) Running ML performance guardrails...)
	@echo "Guardrails mode: strict=$${ML_GUARDRAILS_STRICT:-false}"
	uv run --active --no-sync python ml/tests/performance/ci_performance_guardrails.py \
		$(if $(ML_GUARDRAILS_STRICT),--strict) \
		$(if $(ML_REPORT_FILE),--report-file $(ML_REPORT_FILE))

.PHONY: pytest-ml-guardrails-strict
pytest-ml-guardrails-strict:  #-- Run ML performance guardrails in strict mode
	$(info $(M) Running ML performance guardrails in STRICT mode...)
	ML_GUARDRAILS_STRICT=true $(MAKE) pytest-ml-guardrails

.PHONY: pytest-ml-zero-allocation
pytest-ml-zero-allocation:  #-- Run zero-allocation validation tests only
	$(info $(M) Running ML zero-allocation validation...)
	uv run --active --no-sync python ml/tests/performance/ci_performance_guardrails.py --zero-allocation-only

# ML test profiles (fast/stores/integration)
.PHONY: test-fast
test-fast:  #-- Fast dev profile: unit + property + contracts (parallel; no integration/performance)
	$(info $(M) Running ML fast profile: unit + property + contracts...)
	HYPOTHESIS_PROFILE=ci ML_DISABLE_METRICS_SERVER=1 TEST_DB_SKIP_TRUNCATE=1 \
		uv run --active --no-sync pytest -c ml/pytest.ini -q -n auto --dist=loadscope \
		-m "not integration and not performance and not prototype" \
		ml/tests/unit ml/tests/property ml/tests/contracts \
		--junitxml=ml/tests/validation_reports/junit-dev-fast.xml \
		--cov=ml --cov-report=xml:ml/tests/validation_reports/coverage-dev-fast.xml

.PHONY: test-stores
test-stores:  #-- Stores-focused subset (all store tests excluding performance)
	$(info $(M) Running ML store-focused tests...)
	uv run --active --no-sync pytest -c ml/pytest.ini -q -n auto --dist=loadscope \
		-k "stores and not performance" ml/tests \
		--junitxml=ml/tests/validation_reports/junit-stores.xml

.PHONY: test-integration
test-integration:  #-- Integration tests only (real Postgres; serial where needed)
	$(info $(M) Running ML integration tests...)
	uv run --active --no-sync pytest -c ml/pytest.ini -q -m integration -n 1 \
		ml/tests/integration \
		--junitxml=ml/tests/validation_reports/junit-integration.xml
