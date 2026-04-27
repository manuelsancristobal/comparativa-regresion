"""Load and save JSON data files for D3.js visualization."""

import json

from src.config import DATA_PROCESSED, VIZ_DIR


def load_and_save_linear(transform_result):
    """Load and save linear regression data as JSON."""
    data = transform_result["linear"]

    output = {
        "scatter": data["scatter"],
        "frames": data["frames"],
        "analytical": data["analytical"],
        "sklearn": data["sklearn"],
    }

    filepath = DATA_PROCESSED / "linear_frames.json"
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    # Also copy to viz/assets/data/
    viz_path = VIZ_DIR / "assets" / "data" / "linear_frames.json"
    with open(viz_path, "w") as f:
        json.dump(output, f, indent=2)

    return filepath


def load_and_save_polynomial(transform_result):
    """Load and save polynomial regression data as JSON."""
    data = transform_result["polynomial"]

    output = {
        "scatter": data["scatter"],
        "frames": data["frames"],
        "sklearn": {str(k): v for k, v in data["sklearn"].items()},
    }

    filepath = DATA_PROCESSED / "polynomial_frames.json"
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    # Also copy to viz/assets/data/
    viz_path = VIZ_DIR / "assets" / "data" / "polynomial_frames.json"
    with open(viz_path, "w") as f:
        json.dump(output, f, indent=2)

    return filepath


def load_and_save_logistic(transform_result):
    """Load and save logistic regression data as JSON."""
    data = transform_result["logistic"]

    output = {
        "scatter": data["scatter"],
        "frames": data["frames"],
        "roc_curve": data["roc_curve"],
        "sklearn": data["sklearn"],
    }

    filepath = DATA_PROCESSED / "logistic_frames.json"
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    # Also copy to viz/assets/data/
    viz_path = VIZ_DIR / "assets" / "data" / "logistic_frames.json"
    with open(viz_path, "w") as f:
        json.dump(output, f, indent=2)

    return filepath


def load(transform_result):
    """Load all transformed data into JSON files."""
    linear_path = load_and_save_linear(transform_result)
    polynomial_path = load_and_save_polynomial(transform_result)
    logistic_path = load_and_save_logistic(transform_result)

    return {
        "linear": linear_path,
        "polynomial": polynomial_path,
        "logistic": logistic_path,
    }
