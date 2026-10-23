# Common commands for the COMP 536 prefix-sharing project.
# Run `make` or `make help` to list them. Works with the GNU Make 3.81 that
# ships with macOS. CI calls the same targets (specs/001-ci-cd).

PYTHON_VERSION ?= 3.12
VENV ?= .venv
# Branch PRs merge into (the default branch on GitHub); `lint` and
# `check-pr` compare against it. Update it when the default branch changes.
BASE ?= origin/milestone-2
# Extra pytest arguments, e.g. `make test-m1 ARGS="-k prefix -x"`.
ARGS ?=
# PR description file for `check-pr`; without it the description is skipped.
BODY ?=

PY := $(VENV)/bin/python
# Puts the virtualenv first on PATH, so nobody has to activate it.
WITH_VENV := PATH="$(abspath $(VENV))/bin:$$PATH"
SIM_ENV := VLLM_USE_SIMULATOR=1 VLLM_NO_USAGE_STATS=1
UNAME_S := $(shell uname -s)
PYTORCH_CPU_INDEX := https://download.pytorch.org/whl/cpu

.DEFAULT_GOAL := help
.PHONY: help setup test test-m1 test-m2 test-m3 test-core check-markers \
	test-pr-check ci lint check-pr

help: ## List the available commands
	@echo "Usage: make <target> [ARGS=\"pytest args\"] [BASE=$(BASE)] [BODY=pr.md]"
	@echo
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  make %-15s %s\n", $$1, $$2}'

setup: ## Create .venv and install the project, CPU only (macOS or Linux)
	@command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/"; exit 1; }
	test -x $(PY) || uv venv --python $(PYTHON_VERSION) $(VENV)
ifeq ($(UNAME_S),Darwin)
	VLLM_USE_PRECOMPILED=1 uv pip install --python $(PY) --editable . \
		-c constraints-simulator.txt
else
	VLLM_TARGET_DEVICE=empty uv pip install --python $(PY) --editable . \
		-c constraints-simulator.txt \
		--extra-index-url $(PYTORCH_CPU_INDEX) --index-strategy unsafe-best-match
endif
	uv pip install --python $(PY) pytest==8.3.3 pytest-asyncio==0.24.0 \
		matplotlib==3.9.2 -c constraints-simulator.txt
	$(SIM_ENV) $(PY) -c "from vllm.platforms import current_platform; \
		assert current_platform.is_cpu(); print('Setup OK:', current_platform)"

test: ## Run every milestone's tests
	$(WITH_VENV) tools/run_project_tests.sh all $(ARGS)

test-m1: ## Run only the Milestone 1 tests
	$(WITH_VENV) tools/run_project_tests.sh m1 $(ARGS)

test-m2: ## Run only the Milestone 2 tests
	$(WITH_VENV) tools/run_project_tests.sh m2 $(ARGS)

test-m3: ## Run only the Milestone 3 tests
	$(WITH_VENV) tools/run_project_tests.sh m3 $(ARGS)

check-markers: ## Fail if a project test has no m1/m2/m3 marker
	$(WITH_VENV) tools/run_project_tests.sh check-markers

test-core: ## Run vLLM's scheduler and block-manager tests (CONTRIBUTING 6.2)
	$(SIM_ENV) $(PY) -m pytest tests/core -q \
		--ignore=tests/core/block/e2e \
		--ignore=tests/core/test_num_computed_tokens_update.py $(ARGS)

test-pr-check: ## Run the PR checker's own tests
	$(PY) -m pytest tools/pr_check -q -p no:cacheprovider

ci: check-markers test-m1 test-m2 test-m3 test-core test-pr-check ## Run all the tests CI runs, in order

lint: ## Lint files changed since BASE with the repo's pinned linters
	@base=$$(git merge-base $(BASE) HEAD) || exit 1; \
	changed=$$( { git diff --name-only --diff-filter=ACMR $$base; \
		git ls-files --others --exclude-standard; } | sort -u); \
	py=$$(printf '%s\n' $$changed | grep '\.py$$' || true); \
	if [ -n "$$py" ]; then \
		echo "ruff, isort, yapf on:" $$py; \
		uvx ruff@0.6.5 check $$py && \
		uvx --python $(PYTHON_VERSION) isort@5.13.2 --check-only $$py && \
		uvx --python $(PYTHON_VERSION) --from yapf==0.32.0 --with toml==0.10.2 \
			yapf --diff $$py || exit 1; \
	else echo "No changed Python files."; fi; \
	text=$$(printf '%s\n' $$changed | grep -E '\.(py|md|rst)$$' || true); \
	if [ -n "$$text" ]; then \
		uvx --from codespell==2.3.0 --with tomli codespell --toml pyproject.toml $$text \
			&& echo "codespell clean"; \
	fi

check-pr: ## Check this branch against CONTRIBUTING.md (BODY=pr.md adds the description)
	$(PY) tools/pr_check/check_pr.py --base $(BASE) --head HEAD \
		$(if $(BODY),--body-file $(BODY),--no-description)
