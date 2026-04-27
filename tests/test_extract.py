"""Tests for data extraction."""

import numpy as np

from src.etl.extract import extract, load_california_housing


def test_load_california_housing():
    """Test that California Housing dataset loads correctly."""
    df = load_california_housing()
    assert df.shape == (20640, 9)
    assert "MedHouseVal" in df.columns
    assert "MedInc" in df.columns


def test_extract_returns_dict():
    """Test that extract returns a dictionary with expected keys."""
    data = extract()
    assert isinstance(data, dict)
    assert "linear" in data
    assert "polynomial" in data
    assert "logistic" in data
    assert "df" in data


def test_linear_data_shapes(linear_data):
    """Test that linear data has correct shapes."""
    assert linear_data["x_train"].ndim == 1
    assert linear_data["y_train"].ndim == 1
    assert len(linear_data["x_train"]) == len(linear_data["y_train"])
    assert len(linear_data["x_test"]) == len(linear_data["y_test"])


def test_polynomial_data_shapes(polynomial_data):
    """Test that polynomial data has correct shapes."""
    assert polynomial_data["x_train"].ndim == 1
    assert polynomial_data["y_train"].ndim == 1
    assert len(polynomial_data["x_train"]) == len(polynomial_data["y_train"])
    assert len(polynomial_data["x_test"]) == len(polynomial_data["y_test"])


def test_logistic_data_shapes(logistic_data):
    """Test that logistic data has correct shapes."""
    assert logistic_data["x_train"].ndim == 2
    assert logistic_data["x_train"].shape[1] == 3  # MedInc, HouseAge, AveRooms
    assert logistic_data["y_train"].ndim == 1
    assert set(np.unique(logistic_data["y_train"])) <= {0, 1}
    assert set(np.unique(logistic_data["y_test"])) <= {0, 1}


def test_train_test_split_ratio(linear_data):
    """Test that train/test split is approximately 80/20."""
    n_train = len(linear_data["x_train"])
    n_test = len(linear_data["x_test"])
    ratio = n_train / (n_train + n_test)
    assert 0.78 < ratio < 0.82, f"Expected ~0.8 ratio, got {ratio}"


def test_data_normalization(linear_data):
    """Test that data is normalized (standardized)."""
    x_train = linear_data["x_train"]
    # Normalized data should have mean close to 0
    assert abs(np.mean(x_train)) < 0.1, "x_train should be normalized"
    # Normalized data should have std close to 1
    assert 0.8 < np.std(x_train) < 1.2, "x_train should have std ~1"
