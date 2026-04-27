# Evolución del Proyecto

## v1 — Pipeline ETL + Visualización D3.js interactiva

### Problema
El notebook original solo cubría regresión lineal simple con 3 métodos sobre el dataset Diabetes (442 filas). No tenía estructura de proyecto, tests, ni despliegue automatizado.

### Decisiones de diseño
- Expansión del notebook original (solo regresión lineal) a 3 tipos de regresión: lineal, polinómica y logística
- Migración de dataset Diabetes (442 filas, BMI → target) a California Housing (20,640 filas, 8 features)
- Pipeline ETL modular (extract → transform → load) siguiendo el patrón de barchart-race
- Visualización D3.js con 3 secciones y controles interactivos (scroll vertical)
- Cada sección con 3 paneles sincronizados: visualización principal, métrica de convergencia, información adicional

### Cambios técnicos
- `src/etl/extract.py`: carga California Housing via scikit-learn, prepara datos por sección con train/test split
- `src/etl/transform.py`: cómputo de regresiones (analítica, GD, Newton-Raphson, sklearn) + generación de frames
- `src/etl/load.py`: genera 3 JSONs (linear, polynomial, logistic) con muestreo estratificado para D3.js
- `viz/`: página única con scroll vertical, 3 motores JS independientes + utilidades compartidas
- 19 tests unitarios validando carga, convergencia, estructura JSON y serialización
- CI con GitHub Actions, deploy automatizado a Jekyll
- `pyproject.toml`, `Makefile`, `run.py`, ruff, pre-commit

## v0 — Prototipo en Jupyter Notebook (Google Colab)
- Notebook único comparando 3 métodos de regresión lineal simple (analítica, gradiente descendente, scikit-learn)
- Dataset Diabetes de scikit-learn (442 filas, BMI → target)
- Animaciones matplotlib exportadas como GIF
- Sin tests, sin estructura de proyecto, sin deploy automatizado
