---
layout: project
title: "Reaprendiendo Regresiones: Un Refresco Matemático"
category: Machine Learning
description: Una excusa personal para volver a las bases, desempolvar el papel milimetrado y entender qué sucede realmente detrás de una sola línea de código.
github_url: https://github.com/manuelsancristobal/comparativa-regresion
tech_stack:
  - Python
  - NumPy
  - scikit-learn
  - D3.js
  - Pandas
---

## Volviendo a aprender

Cuando estudié regresiones en la universidad, el panorama era distinto. Éramos "pobres" de herramientas: los más afortunados teníamos una calculadora científica Casio estándar, pero nadie soñaba con una que pudiera graficar. En ese entonces, las regresiones se hacían a pulso, con lápiz y papel milimetrado. Fue ahí donde aprendí la metodología de los **Mínimos Cuadrados Ordinarios (MCO)** y supe que existían las regresiones lineales, polinómicas y logísticas; aunque, afortunadamente, nunca tuve que calcular estas últimas a mano.

A veces veía con envidia a compañeros de otras facultades, más pudientes, con esas hermosas calculadoras que hacían sendos gráficos, pero incluso para ellos el proceso seguía siendo lento.

Sin embargo, con el aumento explosivo de la potencia de cálculo, la regresión se transformó en una simple línea de código. Al ver esto, mi primera reacción fue casi de indignación: *"¡Los jóvenes de ahora nunca sabrán lo que es calcular una regresión!"*. Me sentía el guardián de un conocimiento antiguo.

Pero todo cambió durante mi posgrado. Un profesor nos contó que, en sus tiempos, tenían que mandar los datos impresos al extranjero y esperar meses a que volviera el resultado, simplemente porque en Chile no existía la capacidad de cómputo para procesarlos.

Fue en ese momento cuando me asaltó la duda: **¿Realmente sé yo cómo se hace una regresión?** El tiempo y la comodidad me habían borrado las matemáticas de la memoria por falta de uso. Así que decidí emprender este viaje para explorar cómo se construyen estos modelos desde cero. Y lo primero que descubrí es que, para resolver el mismo rompecabezas, existen caminos muy diferentes.

---

## 1. La Línea Recta: ¿Por dónde se empieza?

Mi reencuentro formal con las matemáticas fue en un curso de Machine Learning. Allí conocí la librería **scikit-learn**, una maravilla que hace el mismo cálculo de mínimos cuadrados a una velocidad impresionante. Para alguien que creció esperando meses por un resultado o peleando con papel milimetrado, ver una regresión resuelta en milisegundos parece magia negra.

Pero, como buen "viejo lobo" del papel, quise entender qué pasaba bajo el capó. Recordaba que el método de **Mínimos Cuadrados Ordinarios (MCO)** busca la distancia mínima entre los puntos y la recta, elevada al cuadrado para evitar peleas entre signos positivos y negativos. Pero para que esa recta sea válida, el mundo debe ser ideal y cumplir supuestos que solemos ignorar: **linealidad**, **independencia de los errores**, **homocedasticidad** (que el error sea constante) y **normalidad**.

### La chispa durante una licencia médica

Un día, navegando por internet, vi una animación hermosa de una recta armándose punto a punto. Mi ego me hizo despreciarla: *"Bah, eso lo hago yo en Python"*. Así que, durante una licencia médica y con más dificultad de la que estoy dispuesto a admitir, programé mi propia animación de un OLS que recalculaba la pendiente cada vez que un nuevo punto aparecía desde el origen (0,0). Me sentía orgulloso.

Sin embargo, al meter mi código en **Claude** para pulirlo, la IA me devolvió algo que no esperaba: un método llamado **Gradiente Descendente**. Me sentí pequeño; o no lo conocía, o mi memoria lo había borrado por completo.

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

### Entendiendo los tres paneles de mi duda

Esta visualización es el resultado de ese choque entre lo que sabía y lo que descubrí:

*   **Panel A (El duelo):** Aquí puedes ver mi lógica original frente a la de la IA. La **línea azul** es mi MCO "recalculado" (la solución analítica): salta a la posición perfecta de inmediato. La **línea roja** es el Gradiente Descendente: no sabe la respuesta, pero usa la **derivada de la función de coste** (el gradiente) para dar pasos pequeños hacia el éxito. Este método fue diseñado para la **optimización a gran escala**, allí donde las matrices son tan gigantes que la matemática directa simplemente no puede procesarlas.
*   **Panel B (La Carrera MSE):** Es el termómetro del error. Verás cómo, a pesar de que la línea roja parece "tonta" al principio, su error (MSE) baja de forma dramática hasta casi calcar el de la línea azul.
*   **Panel C (La huella del aprendizaje):** Muestra la historia de la pendiente. La línea roja dibuja una curva de aprendizaje que busca estabilizarse. Es el rastro de un modelo que, a diferencia de nosotros con el papel milimetrado, tiene la paciencia de aprender de sus errores paso a paso.

---

## 2. Cuando la realidad tiene curvas (y el ego se desborda)

Pronto me di cuenta de que la línea recta es una herramienta noble pero limitada. La realidad rara vez es una regla perfecta; suele tener curvas, ciclos y caprichos que una simple pendiente no puede capturar. Para solucionar esto, la matemática nos ofrece un truco: la **Regresión Polinómica**.

Básicamente, "engañamos" al modelo lineal agregando potencias de las variables (como $x^2$ o $x^3$). Al principio, parece una idea brillante: a más grados, más flexibilidad. Pero aquí es donde tropecé con uno de los conceptos más peligrosos y fascinantes del Machine Learning: el **Overfitting**.

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

### El drama de memorizar en lugar de entender

El **Overfitting** es como aquel estudiante que, en lugar de entender la materia, se aprende las preguntas del examen de memoria. Si el examen es exactamente igual, saca un 7.0; pero si le cambias una coma, no sabe qué responder.

En esta visualización puedes ver ese drama en tiempo real:

*   **Panel A (La danza de las curvas):** Verás que si usamos un grado bajo (grado 1 o 2), el modelo se queda corto (**Underfitting**). Pero si subimos el grado al máximo, la curva se vuelve "nerviosa": intenta tocar cada punto del dataset, retorciéndose de formas absurdas.
*   **Panel B (La curva de la complejidad):** Este es el gráfico que me hubiera ahorrado muchas dudas en la universidad. Muestra el error de entrenamiento (azul) bajando sin parar mientras subimos el grado. Pero fíjate en el error de prueba (rojo): llega un punto en que **empieza a subir**. Esa "U" es la marca del sobreajuste; el modelo es perfecto para lo que conoce, pero inútil para el mundo real.
*   **Panel C (La explosión de los coeficientes):** Aquí está el síntoma físico del problema. Cuando un modelo intenta "memorizar", los coeficientes matemático (los pesos) crecen hasta el infinito. Es el modelo gritando para intentar dar sentido al ruido.

Aprendí que en los datos, como en la vida, **la simplicidad es una virtud**. Solo cuando agregamos miles de datos, estas curvas nerviosas logran calmarse y encontrar la verdadera forma de la realidad.

---

## 3. De predecir precios a tomar decisiones

Al llegar aquí, mi cabeza ya estaba procesando datos de forma distinta, pero me faltaba un salto: ¿y si no me interesa saber el precio exacto de una casa, sino simplemente saber si es cara o barata? En la universidad esto parecía un mundo aparte, pero ahora entendí que es la misma lógica de antes, solo que "aplastada" por una función matemática llamada **Sigmoide**.

Esta función es la que crea esa curva en forma de "S" que ves en la visualización. Lo que hace es tomar cualquier número y forzarlo a vivir entre el 0 y el 1, transformando una simple recta en una **probabilidad**.

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

### La elegancia de la velocidad: Newton vs. el resto

En esta etapa de mi reaprendizaje, me encontré con un viejo conocido que había olvidado: el método de **Newton-Raphson**. Si el Gradiente Descendente era ese caminante que bajaba la montaña tanteando el terreno, Newton es alguien con un mapa topográfico de alta resolución.

*   **Panel A (La Frontera):** Aquí ves cómo la frontera de decisión intenta separar las casas "baratas" de las "caras". Es el momento de la verdad donde el modelo deja de predecir y empieza a clasificar.
*   **Panel B (El duelo de optimizadores):** Esta es la parte que más me voló la cabeza. Verás la línea roja (Gradiente) avanzando paso a paso, de forma segura pero lenta. Y luego verás la línea azul (**Newton-Raphson**) dando saltos gigantes.

¿El truco? Newton no solo usa la pendiente (primera derivada), sino también la curvatura del terreno (segunda derivada o matriz Hessiana). Es matemáticamente más pesado, pero increíblemente más eficiente para llegar al destino. Es como si uno bajara a pie y el otro tuviera un teleférico que sabe exactamente dónde está el valle.

Al final, ambos métodos coinciden, demostrándome que en el Machine Learning, como en la vida, hay quienes prefieren la seguridad de los pasos cortos y quienes se lanzan con la precisión de la geometría.

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
