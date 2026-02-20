.PHONY: setup lint format test download-data train-smoke eval-smoke explain-smoke report

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
	PYTHONPATH=src python -m xaimed.cli --config configs/experiments/quick_smoke.yaml train

eval-smoke: train-smoke
	PYTHONPATH=src python -m xaimed.cli --config configs/experiments/quick_smoke.yaml eval

explain-smoke:
	PYTHONPATH=src python -m xaimed.cli --config configs/experiments/quick_smoke.yaml explain

report:
	PYTHONPATH=src python -m xaimed.cli --config configs/default.yaml report
