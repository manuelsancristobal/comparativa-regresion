# Comparativa de Regresión - Visualización Interactiva

Proyecto interactivo que visualiza y compara **3 tipos de regresión** (lineal, polinómica, logística) usando D3.js y el dataset California Housing de scikit-learn.

## ¿Qué es este proyecto?

Una visualización educativa que muestra:

1. **Regresión Lineal**: 3 métodos resolviendo el mismo problema OLS
   - Solución analítica (ecuación normal)
   - Gradiente descendente (iterativo)
   - Scikit-learn (optimizado)

2. **Regresión Polinómica**: Regresión lineal en espacio transformado
   - Matriz de Vandermonde manual vs scikit-learn
   - Observa overfitting en tiempo real (grado 4 con pocos datos)

3. **Regresión Logística**: Clasificación binaria
   - Gradiente descendente
   - Newton-Raphson (convergencia cuadrática)
   - Scikit-learn

Cada sección tiene **3 paneles sincronizados**:
- Panel A: Visualización principal (scatter + regresión)
- Panel B: Métrica de convergencia (MSE, log-loss)
- Panel C: Información adicional (trayectoria, coeficientes, ROC)

## Instalación

```bash
# Clone o descarga el proyecto
cd Comparativa\ regresion

# Instala dependencias
make install

# O manualmente:
pip install -e .
```

## Uso

### Generar datos (ETL)
```bash
make etl
# O: python run.py etl
```

Genera 3 archivos JSON en `data/processed/`:
- `linear_frames.json`
- `polynomial_frames.json`
- `logistic_frames.json`

### Ver visualización
```bash
make ver
# O: python run.py ver
```

Abre el navegador con `viz/index.html`.

### Ejecutar tests
```bash
make test
# O: pytest tests/ -v
```

19 tests unitarios validando:
- Carga de datos (shapes, normalización)
- Convergencia de algoritmos
- Estructura de JSONs
- Serialización correcta

### Lint y formato
```bash
make lint      # Verificar código
make format    # Formatear código
```

## Estructura del Proyecto

```
Comparativa regresion/
├── src/
│   ├── config.py              # Configuración (rutas, constantes, colores)
│   ├── main.py                # Orquestador ETL
│   ├── deploy.py              # Deployer a Jekyll
│   └── etl/
│       ├── extract.py         # Carga dataset + prepara datos
│       ├── transform.py       # Calcula regresiones + genera frames
│       └── load.py            # Exporta JSONs
├── data/processed/            # JSONs generados
├── viz/
│   ├── index.html             # Página principal
│   └── assets/
│       ├── css/regression.css
│       ├── js/
│       │   ├── common.js      # Utilidades D3.js
│       │   ├── linear-race.js
│       │   ├── polynomial-race.js
│       │   └── logistic-race.js
│       └── data/              # JSONs copiados aquí
├── tests/                     # Suite de tests
├── jekyll/                    # Assets para Jekyll
├── Makefile                   # CLI del proyecto
├── pyproject.toml            # Configuración Python
└── README.md
```

## Dataset

**California Housing** (scikit-learn):
- 20,640 bloques de viviendas en California (censo 1990)
- 8 features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
- Target: MedHouseVal (valor mediano, en cientos de miles USD)

**Nota sobre visualización**: Los scatters D3.js usan una muestra estratificada de ~2,000 puntos para evitar saturar el navegador, pero los cálculos de regresión usan las 20,640 filas completas.

## Dependencias

### Runtime
- `numpy >= 1.21.0`
- `pandas >= 1.3.0`
- `scikit-learn >= 1.0.0`

### Development
- `pytest >= 7.0.0`
- `pytest-cov >= 3.0.0`
- `ruff >= 0.0.250`
- `pre-commit >= 2.20.0`

### Visualización
- D3.js v7 (cargado vía CDN en `index.html`)

## Insights Clave

### Sección 1: Lineal
- **Convergencia única**: Los 3 métodos llegan al mismo punto porque OLS tiene solución única
- **Trayectoria**: GD recorre un camino "cuesta abajo" por la superficie de error
- **Trade-off**: MSE train < MSE test (generalización)

### Sección 2: Polinómica
- **OLS disfrazada**: No es modelo nuevo, solo transformación de features
- **Bias-variance**: Grado bajo = alto sesgo (rigidez), grado alto = alta varianza (oscilación)
- **Regularización**: Ridge/Lasso penalizan coeficientes grandes (visibles en Panel C)

### Sección 3: Logística
- **Probabilidades**: Sigmoid transforma cualquier número real a [0,1]
- **Newton vs GD**: Newton-Raphson converge en ~5-10 iteraciones vs miles de GD
- **Log-loss**: Evita el gradiente "plano" de sigmoid+MSE

## Configuración

Edita `src/config.py` para cambiar:
- Tasa de aprendizaje (`GD_LEARNING_RATE`)
- Número máximo de iteraciones (`GD_MAX_ITERS`)
- Colores por método
- Ruta del repo Jekyll (`JEKYLL_REPO`)

## Deploy a Jekyll

```bash
make deploy
# O: python run.py deploy
```

Copia todos los assets a tu repo Jekyll local. Requiere que `JEKYLL_REPO` esté configurado en `src/config.py`.

## Notas Técnicas

### Muestreo de Frames
- **Lineal**: Muestreo logarítmico de iteraciones de GD (más frames al inicio)
- **Polinómica**: Secuencia creciente de cantidad de puntos (0.2n → n)
- **Logística**: Máximo de longitud de historiales GD vs Newton

### Normalización
Todos los features se normalizan con `StandardScaler` ajustado en datos de entrenamiento (80%), aplicado a test (20%).

### Semilla Aleatoria
`RANDOM_SEED = 42` para reproducibilidad.

## Limitaciones Conocidas

1. **MedHouseVal capped**: Capped en 5.0 → viola linealidad en extremo superior
2. **Latitude sola**: No captura geografía real (Longitud importa)
3. **Multicolinealidad**: MedInc y AveRooms correlacionadas (afecta interpretación en logística)
4. **Muestreo visual**: Scatter plots usan ~2,000 puntos en D3.js (los 20,640 en cálculos)

## Recursos

- **Notebook original**: documentado en la sección v0 del [CHANGELOG](CHANGELOG.md)
- **Patrón ETL**: Basado en `Barchart Race` del portafolio
- **Patrón D3.js**: Animaciones scroll vertical inspiradas en race visualizations

## Licencia

Proyecto educativo del portafolio personal.

## Autor

Manuel Sancristobal

---

**Última actualización**: Abril 2026
