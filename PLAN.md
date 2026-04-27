# Plan: Proyecto "Comparativa Regresion" - Visualizacion Interactiva D3.js

## Contexto

El portafolio tiene 3 proyectos completos (Atractivos, Barchart Race, Dashboard) y un cuarto proyecto "Comparativa regresion" que solo existe como notebook Jupyter. El objetivo es expandir el prototipo para comparar **3 tipos de regresion** (lineal, polinomica, logistica) usando el dataset California Housing de scikit-learn (~20,640 viviendas, 8 features), con una animacion interactiva D3.js en scroll vertical, siguiendo la estructura de los otros proyectos del portafolio.

---

## Dataset: California Housing (scikit-learn)

**Fuente**: `sklearn.datasets.fetch_california_housing()` (no requiere CSV externo)
**Contenido**: 20,640 bloques de viviendas en California, censo 1990
**Features**: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
**Target**: MedHouseVal (valor mediano de la vivienda, en cientos de miles USD)

**Nota de rendimiento**: Los computos (regresiones, metricas) usan las 20,640 filas completas. Los scatters de D3 usan una muestra estratificada de ~2,000 puntos para evitar saturar el navegador con 20K circulos SVG. El muestreo se hace en `load.py` al generar los JSONs, preservando la distribucion visual del dataset completo.

### Uso por seccion:

| Seccion | Variable X | Variable Y | Logica |
|---------|-----------|-----------|--------|
| **Lineal** | MedInc (ingreso mediano) | MedHouseVal | Relacion clara ingreso-precio, R² ~0.47 — ni trivial ni ruidosa |
| **Polinomica** | Latitude | MedHouseVal | Patron no lineal: precios pico en latitudes de SF (~37.7°) y LA (~34°), valles entre ambas. No es efecto costa (eso requeriria Longitude) sino correlacion con centros urbanos |
| **Logistica** | Features (MedInc, HouseAge, AveRooms) | Cara/Barata (1/0, umbral = mediana) | Clasificacion binaria, accuracy ~80%, curva ROC visualmente interesante |

---

## Diseno de la Visualizacion (Scroll Vertical)

La pagina tiene 3 secciones independientes, cada una con su propia animacion y controles.

### Seccion 1: Regresion Lineal (MedInc vs MedHouseVal)

**Metodos comparados** (portar del notebook, adaptar a California Housing):
1. **Analitica** (azul) - Ecuacion normal, solucion instantanea
2. **Gradiente Descendente** (rojo) - Iterativo, converge visualmente
3. **Scikit-learn** (verde) - LinearRegression, solucion instantanea

**3 paneles sincronizados:**
- **Panel A (principal)**: Scatter MedInc vs MedHouseVal + 3 lineas de regresion. Analitica (azul) y sklearn (verde) aparecen fijas desde el frame 1; solo GD (rojo) se anima iteracion a iteracion hasta converger a las otras dos
- **Panel B (inferior izq)**: Carrera MSE - gradiente desciende hasta alcanzar analitica/sklearn. Incluye linea punteada horizontal de MSE test (referencia de generalizacion)
- **Panel C (inferior der)**: Trayectoria en espacio de parametros (intercept vs slope)

**Controles**: Play/Pause, slider de iteracion, velocidad (1x/2x/5x), toggles de metodo

**Nota narrativa**: Texto breve entre el titulo y los paneles explicando que los 3 metodos resuelven el mismo problema OLS — la diferencia es *como* llegan ahi (cerrada vs iterativa vs optimizada)

### Seccion 2: Regresion Polinomica (Latitude vs MedHouseVal)

**Idea clave a demostrar**: regresion polinomica ES regresion lineal en un espacio transformado. Se construye la matriz de Vandermonde `[1, x, x², ..., xᵈ]` a mano y se resuelve con la ecuacion normal — el mismo OLS de la seccion 1, solo cambia el espacio de features.

**Metodos comparados:**
1. **Vandermonde + OLS manual** - Construye la matriz a mano, resuelve con ecuacion normal. Colores por grado: grado 2 (azul), grado 3 (naranja), grado 4 (rojo)
2. **Scikit-learn Pipeline** (verde punteado por grado) - PolynomialFeatures + LinearRegression como referencia/benchmark, superpuesto para mostrar equivalencia

**Animacion**: ambos metodos se animan simultaneamente, agregando puntos incrementalmente (similar a la animacion analitica del notebook). A medida que se agregan puntos, las curvas de cada grado se van ajustando y se puede ver como grados altos oscilan con pocos datos pero convergen con mas.

**3 paneles sincronizados:**
- **Panel A (principal)**: Scatter Latitude vs MedHouseVal + curvas polinomicas animandose por grado (Vandermonde manual vs sklearn superpuestos para mostrar equivalencia)
- **Panel B (inferior izq)**: Comparativa MSE train vs MSE test por grado (bar chart animado mostrando overfitting en grado 4)
- **Panel C (inferior der)**: Evolucion de coeficientes por grado — muestra como los coeficientes altos crecen desproporcionadamente (motiva regularizacion)

**Controles**: Play/Pause, slider de puntos incluidos, toggles de grado

**Nota narrativa**: Texto que conecta con la seccion 1: "Si grado=1, esto es exactamente la regresion lineal anterior". Mencion breve de que Ridge/Lasso penalizan los coeficientes grandes que se ven en el Panel C

### Seccion 3: Regresion Logistica (Clasificacion: vivienda cara/barata)

**Target binario**: MedHouseVal >= mediana → cara (1), < mediana → barata (0)
**Features**: MedInc, HouseAge, AveRooms (normalizadas)

**Idea clave a demostrar**: la regresion logistica no predice un valor continuo sino una probabilidad via la funcion sigmoid. La optimizacion minimiza log-loss (cross-entropy), no MSE. Esto cambia el paisaje de optimizacion y permite comparar velocidades de convergencia.

**Metodos comparados:**
1. **Gradiente Descendente manual** (rojo) - Implementacion con sigmoid + cross-entropy, converge en ~miles de iteraciones
2. **Newton-Raphson** (azul) - Usa la Hessiana (matriz de segundas derivadas) para convergencia cuadratica: ~5-10 iteraciones. La animacion hace evidente este salto
3. **Scikit-learn LogisticRegression** (verde) - Referencia optimizada (usa L-BFGS internamente, con regularizacion L2 por defecto C=1.0)

**3 paneles sincronizados:**
- **Panel A (principal)**: Scatter de MedInc vs HouseAge coloreado por clase (cara=verde, barata=rojo) + frontera de decision animada. La frontera se calcula fijando AveRooms en su media, proyectando el plano de decision 3D a una linea 2D. Anotacion en vivo: cuando Newton-Raphson converge en ~5 iteraciones mientras GD sigue ajustandose
- **Panel B (inferior izq)**: Carrera de Log-Loss (cross-entropy) - los 3 metodos convergiendo. Hace visualmente obvio que Newton-Raphson "salta" al optimo
- **Panel C (inferior der)**: Curva ROC animada construyendose frame a frame

**Controles**: Play/Pause, slider de iteracion, velocidad, toggles de metodo

**Nota narrativa**: Texto explicando por que se usa log-loss en vez de MSE (gradientes de sigmoid + MSE se aplanan, log-loss no). Mencion de que sklearn agrega regularizacion L2 por defecto — por eso sus coeficientes pueden diferir levemente de las implementaciones manuales

### Seccion Final: Tabla Comparativa Resumen
- Tabla estatica con metricas finales de los 3 tipos
- Complejidad computacional O-notation
- Fila de **supuestos** por tipo: linealidad, homocedasticidad, normalidad de residuos (lineal); mismos + riesgo de overfitting por grado (polinomica); independencia de observaciones, no multicolinealidad severa (logistica)
- Fila de **limitaciones detectadas**: ej. MedHouseVal esta capped en 5.0 (viola linealidad en el extremo superior), Latitude como unica feature no captura la geografia real (longitud importa)
- Mencion de **regularizacion**: Ridge/Lasso para polinomica, parametro C para logistica — y por que importa cuando los coeficientes crecen (conecta con Panel C de seccion 2)
- Conclusiones (adaptadas del notebook)

---

## Estructura del Proyecto

```
Comparativa regresion/
├── .github/workflows/ci.yml
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
├── pyproject.toml
├── Makefile
├── run.py                              # CLI con subcomandos: etl, ver, deploy, test (sys.argv, patron Barchart Race)
├── src/
│   ├── __init__.py
│   ├── config.py                       # Rutas, constantes, colores
│   ├── deploy.py                       # Copia a Jekyll
│   ├── main.py                         # Orquestador ETL
│   └── etl/
│       ├── __init__.py
│       ├── extract.py                  # Carga dataset sklearn + prepara arrays por seccion
│       ├── transform.py                # Computo de regresiones + muestreo de frames
│       └── load.py                     # Genera JSONs para D3
├── data/
│   └── processed/                      # JSONs generados
│       ├── linear_frames.json
│       ├── polynomial_frames.json
│       └── logistic_frames.json
├── tests/
│   ├── conftest.py
│   ├── test_extract.py
│   ├── test_transform.py
│   └── test_load.py
├── jekyll/
│   └── comparativa-regresion.md
└── viz/
    ├── index.html                      # Pagina unica con scroll vertical
    └── assets/
        ├── css/regression.css
        ├── js/
        │   ├── common.js               # Utilidades compartidas (escalas, controles, playback)
        │   ├── linear-race.js           # Motor seccion lineal
        │   ├── polynomial-race.js       # Motor seccion polinomica
        │   └── logistic-race.js         # Motor seccion logistica
        └── data/                        # JSONs copiados
```

---

## Makefile (CLI del proyecto)

Patron comun con Barchart Race, Atractivos y Dashboard. Targets:

```makefile
help          # Lista de comandos disponibles (default goal)
install       # pip install -e .
install-dev   # pip install -e ".[dev]" && pre-commit install
lint          # ruff check + ruff format --check sobre src/ tests/
test          # pytest tests/ -v
coverage      # pytest con --cov=src, genera htmlcov/
etl           # python run.py etl (genera los 3 JSONs)
ver           # python run.py ver (abre viz en navegador)
deploy        # python run.py deploy (copia assets a Jekyll)
clean         # rm caches, __pycache__, build/, htmlcov/
```

---

## Pipeline ETL

### `src/config.py`
- Rutas internas: PROJECT_ROOT, DATA_PROCESSED, VIZ_DIR
- Rutas Jekyll: JEKYLL_REPO (default: `~/OneDrive/Documentos/manuelsancristobal.github.io`), JEKYLL_BASE (`proyectos/comparativa-regresion`), subdirs para data, css, js, page
- Colores por metodo y por seccion
- Constantes: GD_LEARNING_RATE, GD_MAX_ITERS, POLY_DEGREES, N_FRAMES
- Train/test split ratio (80/20)

### `src/etl/extract.py`

Solo carga y prepara datos (sin computo de modelos). Cada funcion `prepare_*` aplica train/test split (80/20, seed fijo para reproducibilidad) y retorna `{x_train, x_test, y_train, y_test}`:
- `load_california_housing()` -> DataFrame con 8 features + MedHouseVal (via sklearn)
- `prepare_linear_data(df)` -> x=MedInc, y=MedHouseVal (arrays normalizados con StandardScaler fit en train)
- `prepare_polynomial_data(df)` -> x=Latitude (normalizado), y=MedHouseVal
- `prepare_logistic_data(df)` -> X=matrix(MedInc, HouseAge, AveRooms normalizadas con StandardScaler fit en train), y=cara/barata (1/0, umbral=mediana)
- `extract()` -> dict con los 3 bloques de datos preparados (cada uno con train/test)

### `src/etl/transform.py`

Computo de regresiones + generacion de frames para animacion:

**Regresion lineal** (portada del notebook):
- `compute_analytical(x_train, y_train, x_test, y_test)` -> {"intercept", "slope", "mse_train", "mse_test"}
- `compute_gradient_descent(x_train, y_train, x_test, y_test, lr, n_iters)` -> {"intercept", "slope", "mse_train", "mse_test", "history": [{slope, intercept, mse_train}, ...]}
- `compute_sklearn_linear(x_train, y_train, x_test, y_test)` -> {"intercept", "slope", "mse_train", "mse_test"}

**Regresion polinomica:**
- `build_vandermonde(x, degree)` -> matriz [1, x, x², ..., xᵈ]
- `compute_polynomial_ols(x, y, degree, n_points_sequence)` -> {"coefficients", "history_by_points": [{n_points, coeffs, mse_train, mse_test}, ...]} (ecuacion normal sobre Vandermonde)
- `compute_sklearn_polynomial(x, y, degree)` -> {"coefficients", "mse_train", "mse_test"}

**Regresion logistica:**
- `sigmoid(z)` -> array
- `compute_logistic_gd(X, y, lr, n_iters)` -> {"weights", "history": [{weights, log_loss}, ...]}
- `compute_logistic_newton(X, y, n_iters)` -> {"weights", "history": [{weights, log_loss}, ...]}
- `compute_sklearn_logistic(X, y)` -> {"weights", "log_loss", "accuracy"}

**Generacion de frames:**
- `sample_history_log(history, n_frames)` -> muestreo logaritmico de historiales de iteraciones (mas frames al inicio). Usado por lineal (GD) y logistica (GD + Newton). NO aplica a polinomica (que anima por cantidad de puntos, no por iteraciones)
- `sample_points_sequence(n_total, n_frames)` -> secuencia creciente de cantidad de puntos para animacion polinomica
- `build_linear_frames(data)` -> frames con estado de 3 metodos + trayectoria
- `build_polynomial_frames(data)` -> frames con curvas por grado + MSE train/test
- `build_logistic_frames(data)` -> frames con frontera de decision (AveRooms fijado en media) + log-loss + ROC
- `compute_roc_curve(y_true, y_prob)` -> puntos (fpr, tpr) para curva ROC
- `transform(extract_result)` -> dict con los 3 conjuntos de frames

### `src/etl/load.py`

Genera 3 JSONs:
1. **`linear_frames.json`**: scatter, 200 frames (3 metodos), trayectoria parametros, colores
2. **`polynomial_frames.json`**: scatter, frames por grado, MSE train/test por grado, colores
3. **`logistic_frames.json`**: scatter con clases, frames (3 metodos), log-loss, curva ROC, colores

---

## Visualizacion D3.js

### `viz/assets/js/common.js`
Utilidades compartidas (evitar duplicacion):
- `createPlaybackControls(container, callbacks)` - genera controles Play/Pause/Slider/Speed
- `createMethodToggles(container, methods, colors, callback)` - checkboxes de metodos
- `createAnnotationBanner(container)` - banner de anotaciones
- `createScales(domain, range)` - factory de escalas D3
- `formatNumber(n)` - formato numerico consistente

### `viz/assets/js/linear-race.js`
Patron IIFE. 3 paneles sincronizados:
- Scatter MedInc vs MedHouseVal + lineas animadas
- MSE race (eje Y log)
- Trayectoria espacio parametros

### `viz/assets/js/polynomial-race.js`
Patron IIFE. 3 paneles:
- Scatter Latitude vs MedHouseVal + curvas polinomicas animadas (agregan puntos incrementalmente)
- Bar chart MSE train vs test por grado (muestra overfitting)
- Panel de coeficientes

### `viz/assets/js/logistic-race.js`
Patron IIFE. 3 paneles:
- Scatter MedInc vs HouseAge coloreado por clase + frontera de decision animada (AveRooms fijado en media)
- Log-loss race (3 metodos convergiendo)
- Curva ROC construyendose

### `viz/index.html`
Pagina unica con:
- Header del proyecto (titulo, descripcion breve)
- Seccion 1: Regresion Lineal (con sus controles y 3 paneles)
- Seccion 2: Regresion Polinomica (con sus controles y 3 paneles)
- Seccion 3: Regresion Logistica (con sus controles y 3 paneles)
- Seccion 4: Tabla comparativa resumen + conclusiones
- Footer

### `viz/assets/css/regression.css`
Adaptar `barchart.css`:
- CSS custom properties para colores por seccion
- Layout flex para cada seccion de 3 paneles
- Estilos de controles reutilizados
- Separadores visuales entre secciones
- Scroll suave entre secciones

---

## Secuencia de Implementacion

### Fase 1: Scaffolding + Data
- Crear estructura de directorios completa
- `pyproject.toml` (deps: numpy, scikit-learn, pandas; dev: pytest, pytest-cov, ruff, pre-commit)
- `Makefile`, `run.py`, `.gitignore`, `.pre-commit-config.yaml`
- Notebook original documentado en CHANGELOG v0
- Validar carga de California Housing con `fetch_california_housing()` (no requiere CSV externo)
- **Done**: `python -c "from src.etl.extract import load_california_housing; print(load_california_housing().shape)"` imprime `(20640, 9)`

### Fase 2: ETL Pipeline
- `src/config.py`
- `src/etl/extract.py` - carga California Housing + preparacion de arrays por seccion
- `src/etl/transform.py` - computo de regresiones (lineal del notebook, polinomica y logistica nuevas) + muestreo de frames
- `src/etl/load.py` - generacion de 3 JSONs
- `src/main.py` - orquestador
- **Done**: `python run.py etl` genera 3 JSONs validos en `data/processed/`, cada uno con estructura de frames verificable

### Fase 3: Tests
- `test_extract.py` - carga de datos, shapes correctos, normalizacion
- `test_transform.py` - coeficientes, convergencia GD, clasificacion, estructura de frames, ROC
- `test_load.py` - estructura JSON valida, campos requeridos presentes
- **Done**: `python run.py test` pasa todos los tests + `ruff check` limpio

### Fase 4: Visualizacion D3.js
- `viz/assets/js/common.js` - utilidades compartidas
- `viz/assets/js/linear-race.js` - seccion lineal
- `viz/assets/js/polynomial-race.js` - seccion polinomica
- `viz/assets/js/logistic-race.js` - seccion logistica
- `viz/index.html` - layout scroll vertical
- `viz/assets/css/regression.css` - estilos
- **Done**: las 3 secciones renderizan con datos reales, controles Play/Pause/Slider funcionan, sin errores en consola del navegador

### Fase 5: Deploy + Documentacion
- `README.md`
- `jekyll/comparativa-regresion.md` — pagina del proyecto en Jekyll.
  Estructura: intro breve → (iframe + insights) × 3 secciones → cierre.
  Cada seccion sigue el ciclo: **ver la animacion → descubrir que paso**.
  Tono de redaccion: descubrimiento personal / guia de estudio ("al observar...", "lo que revela esto es...", "lo interesante aqui es que...").

  **Frontmatter**: layout, title, category ("Machine Learning"), description, github_url, tech_stack (Python, D3.js, scikit-learn, NumPy)

  **Intro** (2-3 parrafos):
  - Que hace el proyecto y por que existe
  - El hilo conductor: lineal y polinomica comparten OLS como fundamento, logistica introduce MLE (Maximum Likelihood), pero las tres comparten los mismos metodos de optimizacion (ecuacion normal, gradiente descendente, Newton-Raphson) — esa es la conexion real

  **Seccion 1 — Regresion Lineal**:
  - Iframe unico que carga `viz.html` completo (patron Barchart Race). El usuario hace scroll dentro del iframe para navegar las 3 secciones. No se usan hash anchors separados
  - Insights de descubrimiento, ejemplo de tono:
    - "Al observar la animacion, lo primero que salta a la vista es que la linea roja (gradiente) se mueve erráticamente al inicio pero termina exactamente donde la azul (analitica) ya estaba desde el frame 1. Esto no es coincidencia..."
    - Por que la solucion analitica es instantanea (ecuacion normal resuelve en un paso)
    - Que significa la trayectoria en el espacio de parametros: cada punto es una "apuesta" de (intercepto, pendiente) y GD recorre un camino cuesta abajo por la superficie de error
    - Que las 3 lineas colapsan en una sola al final demuestra que OLS tiene solucion unica
    - La diferencia entre MSE train y MSE test: el modelo no memoriza, generaliza
  - Supuestos y diagnostico:
    - "Para que estos resultados sean confiables, OLS asume: (1) relacion lineal entre MedInc y MedHouseVal, (2) errores con media cero, (3) homocedasticidad (varianza constante de los residuos), (4) no autocorrelacion entre errores"
    - "Al graficar los residuos se nota que la varianza crece con MedInc alto — eso es heterocedasticidad. Ademas, MedHouseVal esta capped en 5.0, lo que aplana la relacion en el extremo superior. El modelo funciona, pero estas violaciones explican por que R² no pasa de 0.47"

  **Seccion 2 — Regresion Polinomica**:
  - (continua en el mismo iframe, el usuario llega haciendo scroll)
  - Insights de descubrimiento:
    - "Lo que revela esta animacion es que regresion polinomica no es un modelo nuevo — es regresion lineal disfrazada. La matriz de Vandermonde transforma x en [1, x, x², x³] y a partir de ahi es el mismo OLS de la seccion anterior"
    - Por que Vandermonde manual y sklearn Pipeline dan exactamente los mismos coeficientes
    - Que al agregar pocos puntos, grado 4 oscila salvajemente — eso es overfitting en tiempo real
    - El Panel C muestra coeficientes que crecen desproporcionadamente con el grado — eso es lo que Ridge/Lasso penalizan
    - Conexion: "si fijo grado=1 en esta seccion, obtengo exactamente la regresion lineal de arriba"
  - Supuestos y diagnostico:
    - "Regresion polinomica hereda los mismos supuestos de OLS (es OLS en un espacio transformado), pero agrega un riesgo nuevo: a mayor grado, mayor varianza del modelo"
    - "Lo que la animacion revela es el bias-variance tradeoff en vivo: grado 2 tiene alto sesgo (curva rigida, no captura los picos en latitudes ~34° y ~37°), grado 4 tiene alta varianza (oscila con pocos puntos). El grado 'justo' depende de cuantos datos tengas — y eso es exactamente lo que Ridge resuelve penalizando coeficientes grandes en vez de elegir un grado a ciegas"

  **Seccion 3 — Regresion Logistica**:
  - (continua en el mismo iframe)
  - Insights de descubrimiento:
    - "Lo interesante aqui es que ya no estamos prediciendo un numero, sino una probabilidad. La funcion sigmoid comprime cualquier valor real al rango [0,1], y la frontera de decision es donde esa probabilidad cruza 0.5"
    - Por que se usa log-loss y no MSE (gradientes de sigmoid + MSE se aplanan, log-loss no — se puede ver en el Panel B como GD avanza sin estancarse)
    - El momento "aha" de Newton-Raphson: converge en ~5 iteraciones vs miles de GD. La Hessiana le da informacion de curvatura que GD no tiene — es como la diferencia entre caminar a ciegas cuesta abajo vs tener un mapa topografico
    - Por que los coeficientes de sklearn difieren levemente: regularizacion L2 con C=1.0 por defecto
  - Supuestos y diagnostico:
    - "Regresion logistica no asume normalidad ni homocedasticidad (no es OLS), pero si asume: (1) independencia de observaciones, (2) relacion lineal entre features y log-odds, (3) ausencia de multicolinealidad severa entre features"
    - "Al usar MedInc, HouseAge y AveRooms juntas, hay riesgo de multicolinealidad — ingresos altos correlacionan con casas mas grandes. Esto no afecta la prediccion pero si la interpretabilidad de los coeficientes individuales: no puedes decir 'MedInc aporta X' si esta confundido con AveRooms"
    - "El accuracy de ~80% confirma que la separacion entre cara/barata no es perfectamente lineal en el espacio de features elegido — la frontera de decision animada lo hace visible"

  **Cierre**:
  - Tabla comparativa resumen: metricas finales, O-notation, supuestos clave (referenciando las secciones donde se diagnosticaron)
  - Limitaciones del dataset (MedHouseVal capped en 5.0, Latitude como unica feature geografica)
  - Reflexion final: lineal y polinomica son el mismo OLS en distintos espacios; logistica cambia la funcion de costo (MLE), pero los metodos de optimizacion (GD, Newton) son los mismos — entender optimizacion es la base de todo lo demas
  - Link al notebook original (patron Atractivos)
- `src/deploy.py` — copia al repo Jekyll local (patron Barchart Race / Atractivos):
  - `data/processed/*.json` → `JEKYLL_BASE/assets/data/`
  - `viz/assets/css/regression.css` → `JEKYLL_BASE/assets/css/`
  - `viz/assets/js/*.js` (common + 3 race) → `JEKYLL_BASE/assets/js/`
  - `viz/index.html` → `JEKYLL_BASE/viz.html`
  - `jekyll/comparativa-regresion.md` → `_projects/`
  - Push manual (consistente con los otros proyectos)
- `.github/workflows/ci.yml`
- **Done**: `python run.py deploy` copia todos los archivos a Jekyll, `ls` confirma presencia, CI pasa en verde

---

## Verificacion

1. `python run.py etl` -> genera 3 JSONs en `data/processed/`
2. `python run.py test` -> todos los tests pasan + ruff limpio
3. `python run.py ver` -> abre navegador con la pagina scroll
4. Verificar seccion Lineal: gradiente converge a los mismos valores que analitica/sklearn, MSE test visible como referencia
5. Verificar seccion Polinomica: Vandermonde manual coincide con sklearn por grado, grado 4 muestra overfitting visible (MSE test sube), coeficientes crecen
6. Verificar seccion Logistica: Newton-Raphson converge en ~5-10 iters vs miles de GD (diferencia visible), frontera de decision se estabiliza, ROC se construye correctamente
7. Verificar que cada seccion tiene controles independientes funcionando

---

## Archivos Criticos de Referencia

- **Notebook fuente**: documentado en CHANGELOG v0 (notebook original ya no vive en el repo)
- **Patron ETL**: `Barchart race/src/etl/` (extract.py, transform.py, load.py)
- **Patron D3.js**: `Barchart race/viz/assets/js/barchart-race.js`
- **Patron HTML**: `Barchart race/viz/index.html`
- **Patron CSS**: `Barchart race/viz/assets/css/barchart.css`
- **Patron run.py**: `Barchart race/run.py`
- **Patron Makefile**: `Barchart race/Makefile`
