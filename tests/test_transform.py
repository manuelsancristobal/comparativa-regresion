"""Tests for data transformation."""

from src.etl.extract import extract
from src.etl.transform import (
    build_linear_frames,
    build_logistic_frames,
    build_polynomial_frames,
    compute_analytical_linear,
    compute_gradient_descent_linear,
    compute_sklearn_linear,
    transform,
)


def test_transform_returns_dict():
    """Test that transform returns expected structure."""
    data = extract()
    result = transform(data)
    assert isinstance(result, dict)
    assert "linear" in result
    assert "polynomial" in result
    assert "logistic" in result


def test_analytical_linear_solution(linear_data):
    """Test analytical linear regression solution."""
    result = compute_analytical_linear(
        linear_data["x_train"],
        linear_data["y_train"],
        linear_data["x_test"],
        linear_data["y_test"],
    )
    assert "intercept" in result
    assert "slope" in result
    assert "mse_train" in result
    assert "mse_test" in result
    assert result["mse_train"] > 0
    assert result["mse_test"] > 0


def test_gradient_descent_convergence(linear_data):
    """Test that gradient descent converges."""
    result = compute_gradient_descent_linear(
        linear_data["x_train"],
        linear_data["y_train"],
        linear_data["x_test"],
        linear_data["y_test"],
    )
    assert "intercept" in result
    assert "slope" in result
    assert "history" in result
    assert len(result["history"]) > 1

    # Check that MSE decreases (mostly)
    history_losses = [h["mse_train"] for h in result["history"]]
    assert history_losses[-1] < history_losses[0]


def test_sklearn_linear_matches_analytical(linear_data):
    """Test that sklearn linear regression is close to analytical."""
    analytical = compute_analytical_linear(
        linear_data["x_train"],
        linear_data["y_train"],
        linear_data["x_test"],
        linear_data["y_test"],
    )
    sklearn = compute_sklearn_linear(
        linear_data["x_train"],
        linear_data["y_train"],
        linear_data["x_test"],
        linear_data["y_test"],
    )

    # Slopes should be very close
    assert abs(analytical["slope"] - sklearn["slope"]) < 0.01
    # Intercepts should be very close
    assert abs(analytical["intercept"] - sklearn["intercept"]) < 0.01


def test_linear_frames_structure(linear_data):
    """Test that linear frames have expected structure."""
    result = build_linear_frames(linear_data)
    assert "scatter" in result
    assert "frames" in result
    assert len(result["frames"]) > 0

    frame = result["frames"][0]
    assert "frame" in frame
    assert "analytical" in frame
    assert "gradient_descent" in frame
    assert "sklearn" in frame


def test_polynomial_frames_structure(polynomial_data):
    """Test that polynomial frames have expected structure."""
    result = build_polynomial_frames(polynomial_data)
    assert "scatter" in result
    assert "frames" in result
    assert len(result["frames"]) > 0

    frame = result["frames"][0]
    assert "frame" in frame
    assert "degree_1" in frame
    assert "degree_2" in frame
    assert "degree_3" in frame
    assert "degree_4" in frame


def test_logistic_frames_structure(logistic_data):
    """Test that logistic frames have expected structure."""
    result = build_logistic_frames(logistic_data)
    assert "scatter" in result
    assert "frames" in result
    assert "roc_curve" in result
    assert len(result["frames"]) > 0

    frame = result["frames"][0]
    assert "frame" in frame
    assert "gradient_descent" in frame
    assert "newton_raphson" in frame
    assert "sklearn" in frame

    # Check ROC curve structure
    assert "fpr" in result["roc_curve"]
    assert "tpr" in result["roc_curve"]
    assert "auc" in result["roc_curve"]
