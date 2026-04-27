"""Main ETL orchestration for Comparativa Regresion."""

from src.etl.extract import extract
from src.etl.load import load
from src.etl.transform import transform


def etl_pipeline():
    """Run the complete ETL pipeline."""
    print("Step 1: Extracting data...")
    extracted_data = extract()

    print("Step 2: Transforming data...")
    transformed_data = transform(extracted_data)

    print("Step 3: Loading data...")
    loaded_data = load(transformed_data)

    return loaded_data


if __name__ == "__main__":
    result = etl_pipeline()
    print("ETL Pipeline Complete!")
    for key, path in result.items():
        print(f"  {key}: {path}")
