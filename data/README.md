# Datos: Comparativa de Regresión

## Origen
- **Dataset**: California Housing Dataset.
- **Fuente**: `sklearn.datasets.fetch_california_housing`.
- **Descripción**: Datos del censo de California de 1990 sobre precios de viviendas y características socioeconómicas.

## Estructura
- `raw/`: Datos originales obtenidos de sklearn (si se guardan en formato plano).
- `processed/`: Archivos JSON (`*_frames.json`) generados para la visualización D3.js, conteniendo los pasos de las animaciones.
- `external/`: Datos de referencia adicionales (si aplica).

## Diccionario de Datos Clave
- `MedInc`: Ingreso medio en el bloque.
- `HouseAge`: Edad media de la casa en el bloque.
- `AveRooms`: Promedio de habitaciones.
- `Target` (MedHouseVal): Valor medio de la vivienda (en cientos de miles de dólares).
