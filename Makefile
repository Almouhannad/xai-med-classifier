.PHONY: setup lint format test train-smoke explain-smoke report

setup:
	pip install -e .
	pip install -r requirements-dev.txt

lint:
	ruff check .
	mypy src

format:
	black .
	ruff check --fix .

test:
	pytest tests

train-smoke:
	python -m xaimed.cli train --config configs/experiments/quick_smoke.yaml

explain-smoke:
	python -m xaimed.cli explain --config configs/experiments/quick_smoke.yaml

report:
	python -m xaimed.cli report --config configs/default.yaml
