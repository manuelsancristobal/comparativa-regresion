# Changelog

En este archivo puedes encontrar todos los cambios notables de este proyecto.
Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/).

## [Unreleased]

## [1.0.0] - 2025-01-01

### Añadido
- Expansión a 3 tipos de regresión: lineal, polinómica y logística.
- Dataset California Housing (20,640 filas) reemplazando al dataset Diabetes.
- Pipeline ETL modular (extract → transform → load).
- Visualización D3.js con 3 secciones y controles interactivos (scroll vertical).
- 19 tests unitarios validando carga, convergencia y estructura JSON.
- CI con GitHub Actions y deploy automatizado a Jekyll.
- Configuración de `pyproject.toml`, `Makefile`, `run.py`, ruff y pre-commit.

### Cambiado
- Los paneles de visualización ahora están sincronizados: visualización principal, métrica de convergencia e información adicional.

## [0.1.0] - 2024-01-01

### Añadido
- Prototipo inicial en Jupyter Notebook (Google Colab).
- Comparativa de 3 métodos de regresión lineal simple (analítica, gradiente descendente, scikit-learn).
- Dataset Diabetes de scikit-learn.
- Animaciones matplotlib exportadas como GIF.
