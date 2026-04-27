.PHONY: help install install-dev lint test coverage assets deploy ver clean

.DEFAULT_GOAL := help

help:
	@echo "Comparativa Regresion - Comandos disponibles"
	@echo "=============================================="
	@echo "install          Instalar dependencias del proyecto"
	@echo "install-dev      Instalar dependencias de desarrollo + pre-commit hooks"
	@echo "lint             Ejecutar linting y formateo con ruff"
	@echo "test             Ejecutar tests con pytest"
	@echo "coverage         Ejecutar tests con reporte de cobertura"
	@echo "assets           Generar archivos JSON de datos"
	@echo "ver              Abrir visualización en navegador"
	@echo "deploy           Copiar assets al repo Jekyll"
	@echo "clean            Eliminar cachés y artefactos de build"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src tests
	ruff format --check src tests

test:
	pytest tests/ -v

coverage:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "\nReporte de cobertura generado en htmlcov/index.html"

assets:
	python run.py etl

ver:
	python run.py ver

deploy:
	python run.py deploy

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf build dist *.egg-info
