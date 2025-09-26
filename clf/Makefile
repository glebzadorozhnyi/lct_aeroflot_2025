check: format lint pytest clean

.PHONY: format
format: pyformat yamlfix mdformat

.PHONY: pyformat
pyformat:
	poetry run ruff check --fix src tests notebooks/*.ipynb && poetry run ruff format src tests notebooks/*.ipynb

.PHONY: yamlfix
yamlfix:
	poetry run yamlfix configs

.PHONY: mdformat
mdformat:
	poetry run mdformat .

.PHONY: lint
lint: pylint

.PHONY: pylint
pylint:
	poetry run ruff check src tests

.PHONY: pytest
pytest:
	poetry run pytest tests

.PHONY: clean
clean:
	PYTHONPATH=. poetry run pyclean .
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf **/**/.ipynb_checkpoints

.PHONY: train
train:
	PYTHONPATH=. HYDRA_FULL_ERROR=1 poetry run python src/train.py $(args)

.PHONY: export
export:
	PYTHONPATH=. HYDRA_FULL_ERROR=1 poetry run python src/export.py $(args)
