.PHONY: setup lint format test download-data train-smoke explain-smoke report

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
	PYTHONPATH=src pytest tests

download-data:
	PYTHONPATH=src python scripts/download_data.py --dataset pathmnist --data-dir data

train-smoke:
	python -m xaimed.cli train --config configs/experiments/quick_smoke.yaml

explain-smoke:
	python -m xaimed.cli explain --config configs/experiments/quick_smoke.yaml

report:
	python -m xaimed.cli report --config configs/default.yaml
