PYTHON := .venv/bin/python
PYTEST := $(PYTHON) -m pytest
BLACK  := .venv/bin/black
SRC    := src
TESTS  := tests

.PHONY: help install install-dev test test-api format format-check clean build

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  install       Install runtime dependencies into .venv - install-dev   Install runtime + test dependencies into .venv"
	@echo "  test          Run unit tests (no API calls)"
	@echo "  test-api      Run all tests including live API calls"
	@echo "  format        Auto-format source and tests with black"
	@echo "  format-check  Check formatting without modifying files"
	@echo "  clean         Remove build artifacts and caches"
	@echo "  build         Build distribution packages"

install:
	uv sync

install-dev:
	uv sync --extra test

test:
	$(PYTEST) $(TESTS) -v

test-api:
	$(PYTEST) $(TESTS) -v --run-api

format:
	$(BLACK) $(SRC) $(TESTS)

format-check:
	$(BLACK) --check $(SRC) $(TESTS)

clean:
	rm -rf dist/ build/ src/*.egg-info
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +

build: clean
	uv build
