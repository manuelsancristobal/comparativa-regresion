---
layout: project
title: El Viaje de la Regresión
category: Machine Learning
description: Un recorrido visual y narrativo por la regresión lineal, polinómica y logística, descubriendo cómo los datos encuentran su forma.
github_url: https://github.com/manuelsancristobal/comparativa-regresion
tech_stack:
  - Python
  - NumPy
  - scikit-learn
  - D3.js
  - Pandas
---

## Un mismo destino, diferentes caminos

Cuando empecé a explorar los modelos de regresión, me di cuenta de algo fascinante: aunque las fórmulas cambien, casi siempre estamos intentando resolver el mismo rompecabezas. Es como intentar encontrar el camino más corto en un mapa; puedes usar una brújula o un GPS, pero el destino es el mismo.

En este proyecto, quise ver con mis propios ojos cómo aprenden estos modelos usando el dataset de **viviendas de California**. No se trata solo de números, sino de ver cómo una línea "aprende" a seguir a los datos.

---

## 1. La Línea Recta: ¿Por dónde se empieza?

Lo primero que aprendemos es que una línea recta es la forma más simple de entender una relación. Pero, ¿cómo decide esa línea dónde ponerse?

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

### El baile entre la teoría y la práctica

Si miras el **Panel A**, verás dos líneas. La **azul** es la "sabia": usa una fórmula matemática directa y se coloca en su sitio al instante. La **roja**, en cambio, es la que "aprende" paso a paso (el famoso gradiente descendente). Es increíble ver cómo la línea roja va tanteando, cometiendo errores y ajustándose hasta que, finalmente, termina abrazando a la línea azul.

En el **Panel C**, podemos seguir ese rastro. Es como ver las huellas de alguien que baja una montaña buscando el punto más bajo. No importa de dónde parta la línea roja, siempre acaba encontrando el mismo valle de error mínimo.

---

## 2. Cuando la realidad tiene curvas

A veces, una línea recta se queda corta. La realidad es más compleja y los datos suelen tener curvas que una simple regla no puede seguir. Aquí es donde descubrí el truco de la **regresión polinómica**.

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

### El peligro de querer ser perfecto

Lo más curioso que descubrí aquí está en el **Panel B**. Cuando intentamos que la curva sea demasiado "flexible" (usando un grado alto), el modelo se vuelve un poco loco. Con pocos datos, la curva salta de un lado a otro intentando tocar cada punto, lo que llamamos *overfitting*.

Solo cuando le damos suficientes datos, esa curva nerviosa empieza a calmarse. En el **Panel C**, verás cómo los coeficientes (la "fuerza" de la curva) crecen muchísimo cuando el modelo está confundido, un recordatorio de que, a veces, menos es más.

---

## 3. De predecir precios a tomar decisiones

Finalmente, me pregunté: ¿y si no quiero predecir un valor exacto, sino simplemente clasificar algo? ¿Es una casa cara o barata? Aquí la línea recta se transforma en una "S" suave llamada **regresión logística**.

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

### La elegancia de la probabilidad

Lo que me gusta de esta visualización es cómo la frontera de decisión separa los dos mundos. En el **Panel B**, comparamos dos formas de aprender: una que camina despacio (Gradiente) y otra que parece dar saltos gigantes hacia la solución (Newton-Raphson).

Es como si uno fuera bajando la montaña a pie y el otro tuviera un mapa topográfico tan perfecto que puede saltar de una cima a otra. Al final, ambos llegan al mismo punto, pero la elegancia matemática de Newton es difícil de ignorar.

---

## Un breve resumen de lo aprendido

| ¿Qué buscamos? | Lo más simple (Lineal) | Con curvas (Polinómica) | Sí o No (Logística) |
| :--- | :--- | :--- | :--- |
| **El objetivo** | Una línea que cruce los datos | Una curva que se adapte | Una frontera que separe |
| **La sorpresa** | El gradiente siempre llega | Cuidado con pasarse de listo | Hay métodos que "saltan" al éxito |
| **Resultado** | Funciona bien casi siempre | Captura detalles complejos | Ideal para clasificar |

---

### Reflexión final

Al final del día, estos modelos son solo herramientas para ayudarnos a ver patrones donde solo parece haber ruido. No hace falta memorizar cada ecuación, sino entender la intención detrás de ellas: **escuchar lo que los datos intentan decirnos.**

### Recursos para curiosear
- **Código fuente**: Todo el motor que mueve estas animaciones está en GitHub.
- **Datos**: Usamos el censo de California de 1990.
- **Visualización**: Hecho con mucho cariño usando D3.js.

<div class="methodology-box" style="margin-top: 2rem; padding: 1.5rem; background: var(--bg-light); border-radius: 8px; border-left: 4px solid var(--secondary);">
    <p style="margin: 0; font-size: 0.95rem;">📌 <strong>Nota:</strong> Este proyecto nació como una curiosidad personal para entender qué pasa "bajo el capó" de los algoritmos que usamos todos los días en Machine Learning.</p>
</div>
