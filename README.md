# Comparativa de Regresión - Visualización Interactiva

## Impacto y Valor del Proyecto
Este proyecto proporciona una herramienta educativa e interactiva para comprender profundamente el comportamiento de los algoritmos fundamentales de aprendizaje supervisado. Al visualizar en tiempo real cómo convergen diferentes métodos de optimización (Analítico, Gradiente Descendente, Newton-Raphson), se facilita la intuición técnica sobre conceptos críticos como el *overfitting*, la regularización y la eficiencia computacional. Es una pieza clave para demostrar la capacidad de traducir modelos matemáticos complejos en interfaces visuales accionables y comprensibles para perfiles no técnicos.

## Stack Tecnológico
- **Lenguaje**: Python 3.10+
- **Librerías Clave**: `Pandas`, `Numpy`, `Scikit-learn`.
- **Frontend**: D3.js v7 (Visualización interactiva), HTML5/CSS3.
- **Calidad de Código**: `Ruff` (Linting), `Pytest` (Testing), `Pre-commit`.
- **CI/CD**: GitHub Actions.

## Arquitectura de Datos y Metodología
El proyecto sigue un patrón ETL (Extract, Transform, Load) desacoplado de la visualización:
1. **Extracción**: Carga del dataset *California Housing* mediante Scikit-learn.
2. **Transformación**: 
   - Limpieza y escalado de datos (StandardScaler).
   - Entrenamiento de modelos de Regresión Lineal, Polinómica y Logística.
   - Generación de "frames" de entrenamiento (captura de estados intermedios del modelo para animación).
3. **Carga**: Exportación de estados del modelo a archivos JSON optimizados para D3.js.
4. **Visualización**: Renderizado dinámico en el navegador con sincronización de paneles de métricas y trayectorias.

## Quick Start (Reproducibilidad)
1. `git clone https://github.com/manuelsancristobal/comparativa-regresion`
2. `make install` (Instala dependencias y el paquete en modo editable)
3. `make test` (Valida la integridad de los algoritmos)
4. `make etl` (Genera los archivos de datos necesarios)
5. `make ver` (Abre la visualización en el navegador)

## Estructura del Proyecto
- `src/`: Código fuente (ETL, Configuración, Deploy).
- `data/`: Estructura estándar (`raw/`, `processed/`, `external/`).
- `viz/`: Activos de la visualización web (D3.js).
- `tests/`: Suite de pruebas unitarias y de integración.
- `jekyll/`: Configuración para integración con el sitio del portafolio.

---
**Autor**: Manuel San Cristóbal Opazo 
**Licencia**: MIT
