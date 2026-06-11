# Makefile for Call Me Maybe

# Variables
PYTHON_MODULE = src
PYTHON = uv run python
FLAKE8 = uv run flake8
MYPY = uv run mypy

export UV_CACHE_DIR=/sgoinfre/students/lfleitas/uv_cache
export UV_PROJECT_ENVIRONMENT=/sgoinfre/students/lfleitas/call_me_maybe_venv

install:
	@echo "Installing project dependencies..."
	uv sync

run:
	@echo "Running the main script..."
	$(PYTHON) -m $(PYTHON_MODULE)

debug:
	@echo "Running in debug mode..."
	$(PYTHON) -m pdb -m $(PYTHON_MODULE)

clean:
	@echo "Cleaning temporary files and caches..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +

	@rm -rf .venv
	
	@echo "[*] Clean complete!"

lint:
	@echo "Running standard linting (flake8 & mypy)..."
	$(FLAKE8) $(PYTHON_MODULE)
	$(MYPY) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs $(PYTHON_MODULE)

lint-strict:
	@echo "Running strict linting..."
	$(FLAKE8) $(PYTHON_MODULE)
	$(MYPY) --strict $(PYTHON_MODULE)

.PHONY: install run debug clean lint lint-strict