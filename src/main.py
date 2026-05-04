"""Main ETL orchestration for Comparativa Regresion."""

from src.etl.extract import extract
from src.etl.load import load
from src.etl.transform import transform


def etl_pipeline():
    """Run the complete ETL pipeline."""
    print("Paso 1: Extrayendo datos...")
    extracted_data = extract()

    print("Paso 2: Transformando datos...")
    transformed_data = transform(extracted_data)

    print("Paso 3: Cargando datos...")
    loaded_data = load(transformed_data)

    return loaded_data


if __name__ == "__main__":
    result = etl_pipeline()
    print("Pipeline ETL completado!")
    for key, path in result.items():
        print(f"  {key}: {path}")
