"""Data extraction and preparation for Comparativa Regresion."""

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED, TRAIN_TEST_SPLIT


def load_california_housing():
    """Load California Housing dataset from scikit-learn."""
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["MedHouseVal"] = data.target
    return df


def prepare_linear_data(df):
    """Prepare data for linear regression: MedInc vs MedHouseVal."""
    x = df[["MedInc"]].values
    y = df["MedHouseVal"].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED
    )

    # Normalize features using train data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return {
        "x_train": x_train.ravel(),
        "x_test": x_test.ravel(),
        "y_train": y_train,
        "y_test": y_test,
    }


def prepare_polynomial_data(df):
    """Prepare data for polynomial regression: MedInc vs MedHouseVal."""
    x = df[["MedInc"]].values
    y = df["MedHouseVal"].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED
    )

    # Normalize features using train data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return {
        "x_train": x_train.ravel(),
        "x_test": x_test.ravel(),
        "y_train": y_train,
        "y_test": y_test,
    }


def prepare_logistic_data(df):
    """Prepare data for logistic regression: Classification as cara/barata."""
    # Create binary target: 1 if MedHouseVal >= median, 0 otherwise
    threshold = df["MedHouseVal"].median()
    y = (df["MedHouseVal"] >= threshold).astype(int).values

    # Features: MedInc, HouseAge, AveRooms
    x = df[["MedInc", "HouseAge", "AveRooms"]].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED
    )

    # Normalize features using train data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def extract():
    """Load and prepare all datasets for the project."""
    df = load_california_housing()

    return {
        "linear": prepare_linear_data(df),
        "polynomial": prepare_polynomial_data(df),
        "logistic": prepare_logistic_data(df),
        "df": df,
    }


if __name__ == "__main__":
    data = extract()
    print("Linear data shape:", data["linear"]["x_train"].shape)
    print("Polynomial data shape:", data["polynomial"]["x_train"].shape)
    print("Logistic data shape:", data["logistic"]["x_train"].shape)
