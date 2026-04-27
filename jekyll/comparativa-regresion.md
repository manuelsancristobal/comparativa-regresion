---
layout: project
title: Comparativa de Métodos de Regresión
category: Machine Learning
description: Visualización interactiva D3.js comparando 3 tipos de regresión (lineal, polinómica, logística) con animaciones de convergencia
github_url: https://github.com/manuelsancristobal
tech_stack:
  - Python
  - NumPy
  - scikit-learn
  - D3.js
  - Pandas
---

## Introducción

¿Qué diferencia hay entre regresión lineal, polinómica y logística? No es lo que probablemente crees.

Este proyecto es una **visualización interactiva** que demuestra que lineal y polinómica resuelven el **mismo problema OLS en distintos espacios**. Logística cambia la función de costo (MLE en vez de suma de cuadrados), pero los métodos de optimización (gradiente descendente, Newton-Raphson) son fundamentalmente los mismos.

Usa el dataset **California Housing** (~20,640 viviendas) para mostrar cómo 3 técnicas distintas convergen a la misma solución, o por qué Newton-Raphson necesita 5-10 iteraciones mientras el gradiente necesita miles.

## Visualización Interactiva

<div style="position: relative; width: 100%; padding-bottom: 75%; overflow: hidden; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem;">
  <iframe
    src="/proyectos/comparativa-regresion/viz.html"
    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;"
    loading="lazy"
    title="Comparativa de Métodos de Regresión"
    allowfullscreen>
  </iframe>
</div>

---

## Sección 1: Regresión Lineal

### La Historia

Al observar la animación, lo primero que salta a la vista es que la **línea roja (gradiente) se mueve erráticamente al inicio** pero termina exactamente donde la **línea azul (analítica) ya estaba desde el frame 1**.

Esto no es coincidencia.

### Por qué la solución analítica es instantánea

La ecuación normal resuelve el problema **en un solo paso de álgebra lineal**:

```
β = (X^T X)^-1 X^T y
```

No hay iteraciones. No hay cálculos iterativos. Solo una inversión de matriz y multiplicación. Eso es lo que ves: el punto azul aparece desde el principio porque el problema ya está resuelto.

### La trayectoria en el espacio de parámetros

El Panel C muestra algo hipnótico: la línea roja trazando un camino "cuesta abajo" en el espacio de (intercept, slope). Cada punto es una "apuesta" de parámetros, y el gradiente recorre un camino optimizado por la geometría del error cuadrático.

### Por qué los 3 métodos colapsan en uno

La diferencia entre MSE train y MSE test es pequeña pero importante: el modelo **generaliza**. Los 3 métodos convergen exactamente al mismo punto porque **OLS tiene solucion única**. No hay múltiples minimos. Hay un único punto óptimo, y todos los caminos llevan a él.

### Supuestos y limitaciones

Para que estos resultados sean confiables, OLS asume:
1. **Relación lineal** entre MedInc y MedHouseVal
2. **Errores con media cero**
3. **Homocedasticidad** (varianza constante de residuos)
4. **No autocorrelación** entre errores

Pero si graficas los residuos, notas algo: **la varianza crece con MedInc alto**. Eso es **heterocedasticidad**. Además, MedHouseVal está **capped en 5.0**, lo que aplana la relación en el extremo superior.

El modelo funciona, pero estas violaciones explican por qué R² no pasa de 0.47. No es debilidad del método, es la realidad de los datos.

---

## Sección 2: Regresión Polinómica

### La Revelación

Lo que revela esta animación es que **regresión polinómica no es un modelo nuevo**.

La matriz de Vandermonde transforma x en [1, x, x², x³], y a partir de ahí es el **mismo OLS de la sección anterior**. Solo que en un espacio de features diferente.

### Vandermonde = Transformación, no Magia

```
[1, x, x²]  →  (ecuación normal OLS)  →  coeficientes
```

Compara la línea Vandermonde manual (rojo) con sklearn Pipeline (verde punteado). **Son exactamente iguales**. Porque sklearn PolynomialFeatures + LinearRegression hace exactamente eso: construye la matriz y resuelve OLS.

### Overfitting en Tiempo Real

Aquí es donde se pone interesante. En el Panel B, al agregar pocos puntos, **grado 4 oscila salvajemente**. Es overfitting en vivo. Pero a medida que agregas puntos, la curva se estabiliza.

El Panel C muestra por qué: **los coeficientes de grado 4 crecen desproporcionadamente**. Son números enormes, lo que causa esas oscilaciones. Es exactamente lo que **Ridge/Lasso penalizan**: esos coeficientes grandes.

### La Conexión

Si fijas grado=1 en esta sección, obtienes exactamente la regresión lineal de arriba. Mismo problema, mismo OLS, diferente espacio de features.

Ese es el insight real: **no hay múltiples paradigmas aquí, solo transformaciones de espacio**.

### Supuestos y el Bias-Variance Tradeoff

Regresión polinómica hereda los mismos supuestos de OLS (es OLS en espacio transformado), pero agrega un riesgo nuevo: **a mayor grado, mayor varianza del modelo**.

Grado 2 tiene alto sesgo (curva rígida, no captura los picos en latitudes ~34° y ~37° que corresponden a ciudades importantes).

Grado 4 tiene alta varianza (oscila con pocos datos).

Grado 3 equilibra ambos.

**Eso es exactamente lo que Ridge resuelve**: en vez de elegir un grado arbitrariamente, penaliza coeficientes grandes. Permite que el modelo encuentre el equilibrio por sí solo.

---

## Sección 3: Regresión Logística

### El Cambio de Paradigma

Lo interesante aquí es que **ya no estamos prediciendo un número, sino una probabilidad**.

La función sigmoid comprime cualquier valor real al rango [0,1]:

```
P(y=1) = 1 / (1 + exp(-z))
```

La frontera de decisión es donde esa probabilidad cruza 0.5.

### Por qué Log-Loss, no MSE

Aquí es donde muchas explicaciones online fallan. Podrías usar MSE para logística, pero verías algo extraño: **los gradientes se aplanan en los extremos**. Cuando el modelo es muy seguro de su predicción (z muy alto o muy bajo), el gradiente de sigmoid + MSE se vuelve pequeñísimo. Es como empujar un objeto cuesta arriba en una pendiente cada vez más plana.

Log-loss (cross-entropy) evita esto:

```
Log-loss = -y * log(ŷ) - (1-y) * log(1-ŷ)
```

Los gradientes **nunca se aplanan**. Siempre hay fuerza para mejorar, aunque el modelo sea muy seguro. Ves esto en el Panel B: GD avanza constantemente, sin "estancarse".

### Newton-Raphson vs Gradiente Descendente

Aquí ocurre algo mágico. Newton-Raphson converge en ~5-10 iteraciones. Gradiente necesita miles.

¿Por qué?

**Newton-Raphson usa la Hessiana** — la matriz de segundas derivadas. Es como la diferencia entre:
- Caminar cuesta abajo a ciegas, esperando no tropezar (GD)
- Tener un mapa topográfico que te muestra exactamente dónde está el valle (Newton)

La Hessiana te dice **cómo es la curvatura del terreno**, no solo si vas en la dirección correcta. Por eso "salta" al óptimo en casi nada.

Pero tiene un costo: invertir una matriz de n×n es O(n³), mientras que un paso de gradiente es O(n). Para problemas pequeños (como aquí, 3 features), Newton es increíblemente rápido. Para millones de features, es impracticable.

### Multicolinealidad

Aquí usamos MedInc, HouseAge y AveRooms. Pero hay un problema: ingresos altos **correlacionan con casas más grandes**. No es causalidad, es confusión estadística.

Esto no afecta la **predicción** (el modelo sigue funcionando), pero sí la **interpretación de coeficientes individuales**. No puedes decir "MedInc aporta X" si está confundido con AveRooms.

Sklearn agrega regularización L2 por defecto (C=1.0), por eso sus coeficientes pueden diferir levemente de las implementaciones manuales.

---

## Tabla Comparativa

| Característica | Lineal | Polinómica | Logística |
|---|---|---|---|
| **Tipo de Problema** | Predicción continua | Predicción continua (no lineal) | Clasificación binaria |
| **Función de Costo** | MSE | MSE | Log-loss (cross-entropy) |
| **R² / Accuracy** | 0.47 | ~0.55-0.65 | ~80% |
| **Convergencia** | Instantánea (analítica) | Instantánea | Iterativa (GD ms, Newton ns) |
| **Supuestos Clave** | Linealidad, homocedasticidad | Linealidad (en espacio transformado) | Independencia, log-odds lineales |

---

## Conclusión

Lineal y polinómica son el **mismo OLS en distintos espacios**. Logística cambia la función de costo pero los métodos de optimización son los mismos.

Lo que este proyecto realmente enseña es que **entender optimización es la base de todo**.

No necesitas memorizar 50 algoritmos distintos. Necesitas entender:
1. **Cómo defines el error** (función de costo)
2. **Cómo encuentras el mínimo** (gradiente descendente, Newton, ecuación normal)
3. **Cómo evitas sobreaustarte** (regularización)

Todo lo demás son variaciones sobre el mismo tema.

### Recursos

- **Notebook original**: Contiene el código que generó este proyecto
- **Dataset**: California Housing (scikit-learn) — 20,640 viviendas de California, censo 1990
- **Visualización**: D3.js v7 con animaciones sincronizadas

---

**Explora la visualización interactiva arriba. Juega con los controles. Observa, no leas — eso es aprender.**


<div class="methodology-box" style="margin-top: 2rem; padding: 1.5rem; background: var(--bg-light); border-radius: 8px; border-left: 4px solid var(--secondary);">
    <p style="margin: 0; font-size: 0.95rem;">📌 <strong>Sobre este proyecto:</strong> Esta es la segunda versión del análisis, rediseñada como aplicación interactiva. El proyecto original, desarrollado como notebook exploratorio, está disponible en
    <a href="https://colab.research.google.com/drive/1wrEKoYuQHJkn9a3lEFVlo8ElW89SFcTg?usp=sharing" target="_blank" style="color: var(--secondary); font-weight: 600;">Google Colab</a>.</p>
</div>
