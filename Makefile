PYTHON ?= python

.PHONY: test lint format datasets-list datasets-public validate-datasets

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

datasets-list:
	$(PYTHON) scripts/datasets/download_all.py --list

datasets-public:
	$(PYTHON) scripts/datasets/download_all.py --dataset public

validate-datasets:
	$(PYTHON) scripts/datasets/validate_datasets.py
