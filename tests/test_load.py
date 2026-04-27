"""Tests for data loading and JSON generation."""

import json
from pathlib import Path

from src.config import DATA_PROCESSED
from src.etl.extract import extract
from src.etl.load import load
from src.etl.transform import transform


def test_load_creates_json_files():
    """Test that load function creates JSON files."""
    data = extract()
    transformed = transform(data)
    result = load(transformed)

    assert "linear" in result
    assert "polynomial" in result
    assert "logistic" in result

    # Check that files were created
    assert Path(result["linear"]).exists()
    assert Path(result["polynomial"]).exists()
    assert Path(result["logistic"]).exists()


def test_linear_json_structure():
    """Test that linear JSON has correct structure."""
    linear_file = DATA_PROCESSED / "linear_frames.json"
    assert linear_file.exists()

    with open(linear_file) as f:
        data = json.load(f)

    assert "scatter" in data
    assert "frames" in data
    assert "analytical" in data
    assert "sklearn" in data

    # Check scatter data
    assert "x" in data["scatter"]
    assert "y" in data["scatter"]
    assert len(data["scatter"]["x"]) > 0
    assert len(data["scatter"]["y"]) > 0

    # Check frames
    assert len(data["frames"]) > 0
    frame = data["frames"][0]
    assert "analytical" in frame
    assert "gradient_descent" in frame
    assert "sklearn" in frame


def test_polynomial_json_structure():
    """Test that polynomial JSON has correct structure."""
    poly_file = DATA_PROCESSED / "polynomial_frames.json"
    assert poly_file.exists()

    with open(poly_file) as f:
        data = json.load(f)

    assert "scatter" in data
    assert "frames" in data
    assert "sklearn" in data

    # Check frames
    assert len(data["frames"]) > 0
    frame = data["frames"][0]
    assert "degree_1" in frame
    assert "degree_2" in frame
    assert "degree_3" in frame
    assert "degree_4" in frame


def test_logistic_json_structure():
    """Test that logistic JSON has correct structure."""
    logistic_file = DATA_PROCESSED / "logistic_frames.json"
    assert logistic_file.exists()

    with open(logistic_file) as f:
        data = json.load(f)

    assert "scatter" in data
    assert "frames" in data
    assert "roc_curve" in data
    assert "sklearn" in data

    # Check scatter data
    assert "x_medinc" in data["scatter"]
    assert "x_houseage" in data["scatter"]
    assert "y" in data["scatter"]

    # Check ROC curve
    assert "fpr" in data["roc_curve"]
    assert "tpr" in data["roc_curve"]
    assert "auc" in data["roc_curve"]

    # Check frames
    assert len(data["frames"]) > 0
    frame = data["frames"][0]
    assert "gradient_descent" in frame
    assert "newton_raphson" in frame
    assert "sklearn" in frame


def test_json_files_serializable():
    """Test that JSON files are valid and serializable."""
    for json_file in DATA_PROCESSED.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
        # Re-serialize to check validity
        json_str = json.dumps(data)
        assert json_str is not None
