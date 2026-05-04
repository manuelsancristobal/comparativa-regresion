---
layout: project
title: Comparativa de Métodos de Regresión
category: Machine Learning
description: Visualización interactiva D3.js comparando 3 tipos de regresión (lineal, polinómica, logística) con animaciones de convergencia
github_url: https://github.com/manuelsancristobal/comparativa-regresion
tech_stack:
  - Python
  - NumPy
  - scikit-learn
  - D3.js
  - Pandas
---

## Conclusión Central: Un Solo Problema, Diferentes Espacios

**Los métodos de regresión lineal y polinómica resuelven el mismo problema de mínimos cuadrados ordinarios (OLS) en distintos espacios de características.** Aunque la regresión logística cambia la función de costo hacia la estimación de máxima verosimilitud (MLE), los motores de optimización como el gradiente descendente o Newton-Raphson operan bajo principios idénticos.

Este proyecto utiliza el dataset **California Housing** (~20,640 viviendas) para demostrar que las 3 técnicas convergen al mismo punto óptimo. Como se observa en las comparativas de tiempo, el método de Newton-Raphson requiere de 5 a 10 iteraciones, mientras que el gradiente requiere miles para alcanzar la misma precisión.

---

## Sección 1: Regresión Lineal

<div class="viz-container" style="position: relative; width: 100%; padding-bottom: 100%; overflow: hidden; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem;">
  <iframe
    src="/proyectos/comparativa-regresion/linear.html"
    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;"
    loading="lazy"
    title="Regresión Lineal"
    allowfullscreen>
  </iframe>
</div>

[Ver en pantalla completa](/proyectos/comparativa-regresion/linear.html){: .btn .btn--primary }

### El Insight de la Convergencia

**La solución analítica es al instante, mientras que el gradiente busca el camino.** Como se aprecia en la animación principal, la **línea azul (analítica)** se posiciona desde el inicio, marcando el objetivo final que la **línea roja (gradiente)** alcanzará tras múltiples pasos.

### La Solución Analítica (Panel A)

La ecuación normal resuelve el problema de forma directa mediante álgebra lineal:

```
β = (X^T X)^-1 X^T y
```

No existen iteraciones en este cálculo. Se trata de una inversión de matriz que posiciona el punto azul de forma inmediata en el gráfico.

### Trayectoria de Parámetros (Panel C)

Como se vio en el espacio de parámetros del **Panel C**, la línea roja traza un descenso continuo hacia el valle de error mínimo. Cada punto representa una combinación de intercepto y pendiente, donde el gradiente se desplaza de forma optimizada por la geometría del error cuadrático.

### Convergencia Única

Los tres métodos colapsan en uno porque **OLS posee una solución única**. No existen mínimos locales; hay un único punto óptimo al que todos los caminos conducen de forma inevitable.

---

## Sección 2: Regresión Polinómica

<div class="viz-container" style="position: relative; width: 100%; padding-bottom: 100%; overflow: hidden; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem;">
  <iframe
    src="/proyectos/comparativa-regresion/polynomial.html"
    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;"
    loading="lazy"
    title="Regresión Polinómica"
    allowfullscreen>
  </iframe>
</div>

[Ver en pantalla completa](/proyectos/comparativa-regresion/polynomial.html){: .btn .btn--primary }

### Transformación de Espacio, No de Paradigma

**La regresión polinómica es OLS aplicado a una matriz transformada.** La matriz de Vandermonde convierte el valor x en un vector [1, x, x², x³], permitiendo que el mismo motor de la sección anterior resuelva problemas no lineales en apariencia.

### Overfitting y Estabilidad (Panel B)

Como se observa en el **Panel B**, al utilizar pocos puntos el modelo de **grado 4 oscila de forma errática**. Este es un ejemplo de sobreajuste en tiempo real que se estabiliza de forma progresiva al incrementar el volumen de datos.

### El Costo de la Complejidad (Panel C)

El **Panel C** revela la causa de la inestabilidad: **los coeficientes de grado 4 crecen en exceso**. Estos valores elevados provocan las oscilaciones visualizadas, un problema que métodos como Ridge o Lasso corrigen al penalizar la magnitud de estos parámetros.

---

## Sección 3: Regresión Logística

<div class="viz-container" style="position: relative; width: 100%; padding-bottom: 100%; overflow: hidden; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem;">
  <iframe
    src="/proyectos/comparativa-regresion/logistic.html"
    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;"
    loading="lazy"
    title="Regresión Logística"
    allowfullscreen>
  </iframe>
</div>

[Ver en pantalla completa](/proyectos/comparativa-regresion/logistic.html){: .btn .btn--primary }

### De Números a Probabilidades

**El objetivo cambia hacia la predicción de una probabilidad binaria entre 0 y 1.** La función sigmoide comprime los valores reales, situando la frontera de decisión en el punto donde la probabilidad cruza el umbral de 0.5.

### Optimización: Log-Loss vs Newton-Raphson

Como se vio en la comparativa del **Panel B**, el uso de Log-loss evita que los gradientes se aplanen en los extremos, manteniendo una fuerza de mejora constante. Por otro lado, la eficiencia de **Newton-Raphson** destaca al converger en menos de 10 iteraciones gracias al uso de la Hessiana.

### Eficiencia Topográfica

Newton-Raphson emplea la curvatura del terreno para "saltar" al óptimo. Mientras el gradiente desciende de forma ciega, la Hessiana actúa como un mapa topográfico. En este entorno de 3 características, Newton resulta con una rapidez increíble, aunque su costo computacional crece de forma cúbica con el número de variables.

---

## Tabla Comparativa

| Característica | Lineal | Polinómica | Logística |
|---|---|---|---|
| **Tipo de Problema** | Predicción continua | Predicción continua (no lineal) | Clasificación binaria |
| **Función de Costo** | MSE | MSE | Log-loss (cross-entropy) |
| **R² / Accuracy** | 0.47 | ~0.55-0.65 | ~80% |
| **Convergencia** | Al instante (analítica) | Al instante | Iterativa (GD ms, Newton ns) |
| **Supuestos Clave** | Linealidad, homocedasticidad | Linealidad (espacio transformado) | Independencia, log-odds lineales |

---

## Síntesis Técnica

Entender la optimización es la base del aprendizaje automático. No es necesario memorizar algoritmos aislados, sino comprender:
1. **Definición del error** (función de costo)
2. **Búsqueda del mínimo** (gradiente, Newton, ecuación normal)
3. **Control de complejidad** (regularización)

Todo lo demás constituye variaciones sobre estos pilares fundamentales.

### Recursos

- **Notebook original**: Código fuente del análisis.
- **Dataset**: California Housing (scikit-learn) — censo 1990.
- **Visualización**: D3.js v7 con animaciones sincronizadas.

---

**Utiliza los controles de la visualización superior para observar la convergencia de forma directa.**


<div class="methodology-box" style="margin-top: 2rem; padding: 1.5rem; background: var(--bg-light); border-radius: 8px; border-left: 4px solid var(--secondary);">
    <p style="margin: 0; font-size: 0.95rem;">📌 <strong>Sobre este proyecto:</strong> Esta es la segunda versión del análisis, rediseñada como aplicación interactiva. El proyecto original está disponible en <a href="https://colab.research.google.com/drive/1wrEKoYuQHJkn9a3lEFVlo8ElW89SFcTg?usp=sharing" target="_blank" style="color: var(--secondary); font-weight: 600;">Google Colab</a>.</p>
</div>
