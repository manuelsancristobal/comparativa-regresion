"""Transform and compute regression models for Comparativa Regresion."""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import PolynomialFeatures

from src.config import (
    GD_LEARNING_RATE,
    GD_MAX_ITERS,
    LOGISTIC_LEARNING_RATE,
    LOGISTIC_MAX_ITERS,
    N_FRAMES,
    NEWTON_MAX_ITERS,
    POLY_DEGREES,
    RANDOM_SEED,
    SCATTER_SAMPLE_SIZE,
)

# ==================== LINEAR REGRESSION ====================


def compute_analytical_linear(x_train, y_train, x_test, y_test):
    """Compute linear regression using analytical solution (normal equation)."""
    # Add intercept column
    X_train = np.column_stack([np.ones_like(x_train), x_train])

    # Normal equation: (X^T X)^-1 X^T y
    coeff = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
    intercept, slope = coeff[0], coeff[1]

    # Predictions
    y_pred_train = intercept + slope * x_train
    y_pred_test = intercept + slope * x_test

    mse_train = np.mean((y_train - y_pred_train) ** 2)
    mse_test = np.mean((y_test - y_pred_test) ** 2)

    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "mse_train": float(mse_train),
        "mse_test": float(mse_test),
    }


def compute_gradient_descent_linear(x_train, y_train, x_test, y_test, lr=None, n_iters=None):
    """Compute linear regression using gradient descent."""
    if lr is None:
        lr = GD_LEARNING_RATE
    if n_iters is None:
        n_iters = GD_MAX_ITERS

    # Initialize parameters
    m = len(y_train)
    intercept = 0.0
    slope = 0.0
    history = []

    for iteration in range(n_iters):
        # Predictions
        y_pred = intercept + slope * x_train

        # Compute gradients
        d_intercept = -2 / m * np.sum(y_train - y_pred)
        d_slope = -2 / m * np.sum((y_train - y_pred) * x_train)

        # Update parameters
        intercept -= lr * d_intercept
        slope -= lr * d_slope

        # Store history (sample logarithmically)
        if iteration % max(1, n_iters // N_FRAMES) == 0 or iteration == n_iters - 1:
            y_pred_train = intercept + slope * x_train
            y_pred_test = intercept + slope * x_test
            mse_train = np.mean((y_train - y_pred_train) ** 2)
            mse_test = np.mean((y_test - y_pred_test) ** 2)

            history.append(
                {
                    "iteration": int(iteration),
                    "intercept": float(intercept),
                    "slope": float(slope),
                    "mse_train": float(mse_train),
                    "mse_test": float(mse_test),
                }
            )

    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "mse_train": float(np.mean((y_train - (intercept + slope * x_train)) ** 2)),
        "mse_test": float(np.mean((y_test - (intercept + slope * x_test)) ** 2)),
        "history": history,
    }


def compute_sklearn_linear(x_train, y_train, x_test, y_test):
    """Compute linear regression using scikit-learn."""
    model = LinearRegression()
    model.fit(x_train.reshape(-1, 1), y_train)

    y_pred_train = model.predict(x_train.reshape(-1, 1))
    y_pred_test = model.predict(x_test.reshape(-1, 1))

    mse_train = np.mean((y_train - y_pred_train) ** 2)
    mse_test = np.mean((y_test - y_pred_test) ** 2)

    return {
        "intercept": float(model.intercept_),
        "slope": float(model.coef_[0]),
        "mse_train": float(mse_train),
        "mse_test": float(mse_test),
    }


# ==================== POLYNOMIAL REGRESSION ====================


def build_vandermonde(x, degree):
    """Build Vandermonde matrix for polynomial features."""
    return np.column_stack([x**d for d in range(degree + 1)])


def compute_polynomial_ols(x_train, y_train, x_test, y_test, degree, n_points_sequence=None):
    """Compute polynomial regression using OLS (Vandermonde + normal equation)."""
    if n_points_sequence is None:
        n_points_sequence = [len(y_train)]

    history = []

    for n_points in n_points_sequence:
        # Use first n_points for training
        x_subset = x_train[:n_points]
        y_subset = y_train[:n_points]

        # Build Vandermonde matrix
        X = build_vandermonde(x_subset, degree)

        # Solve normal equation
        coeff = np.linalg.lstsq(X, y_subset, rcond=None)[0]

        # Predict on full sets
        X_train = build_vandermonde(x_train, degree)
        X_test = build_vandermonde(x_test, degree)

        y_pred_train = X_train @ coeff
        y_pred_test = X_test @ coeff

        mse_train = np.mean((y_train - y_pred_train) ** 2)
        mse_test = np.mean((y_test - y_pred_test) ** 2)

        history.append(
            {
                "n_points": int(n_points),
                "coefficients": coeff.tolist(),
                "mse_train": float(mse_train),
                "mse_test": float(mse_test),
            }
        )

    return history


def compute_sklearn_polynomial(x_train, y_train, x_test, y_test, degree):
    """Compute polynomial regression using scikit-learn."""
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_train = poly.fit_transform(x_train.reshape(-1, 1))
    X_test = poly.transform(x_test.reshape(-1, 1))

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    mse_train = np.mean((y_train - y_pred_train) ** 2)
    mse_test = np.mean((y_test - y_pred_test) ** 2)

    return {
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "mse_train": float(mse_train),
        "mse_test": float(mse_test),
    }


# ==================== LOGISTIC REGRESSION ====================


def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def compute_logistic_gd(X_train, y_train, X_test, y_test, lr=None, n_iters=None):
    """Compute logistic regression using gradient descent."""
    if lr is None:
        lr = LOGISTIC_LEARNING_RATE
    if n_iters is None:
        n_iters = LOGISTIC_MAX_ITERS

    # Add intercept column
    X_train = np.column_stack([np.ones(len(X_train)), X_train])
    X_test = np.column_stack([np.ones(len(X_test)), X_test])

    # Initialize weights (intercept + features)
    n_features = X_train.shape[1]
    weights = np.zeros(n_features)
    history = []

    m_train = len(y_train)

    for iteration in range(n_iters):
        # Predictions
        z_train = X_train @ weights
        y_pred_train = sigmoid(z_train)

        # Gradients
        errors_train = y_pred_train - y_train
        gradients = (1 / m_train) * (X_train.T @ errors_train)

        # Update weights
        weights -= lr * gradients

        # Store history (sample logarithmically)
        if iteration % max(1, n_iters // N_FRAMES) == 0 or iteration == n_iters - 1:
            z_train = X_train @ weights
            y_pred_train = sigmoid(z_train)
            z_test = X_test @ weights
            y_pred_test = sigmoid(z_test)

            # Log loss
            log_loss_train = -np.mean(
                y_train * np.log(y_pred_train + 1e-15) + (1 - y_train) * np.log(1 - y_pred_train + 1e-15)
            )
            log_loss_test = -np.mean(
                y_test * np.log(y_pred_test + 1e-15) + (1 - y_test) * np.log(1 - y_pred_test + 1e-15)
            )

            history.append(
                {
                    "iteration": int(iteration),
                    "intercept": float(weights[0]),
                    "weights": weights[1:].tolist(),
                    "log_loss_train": float(log_loss_train),
                    "log_loss_test": float(log_loss_test),
                }
            )

    return {
        "intercept": float(weights[0]),
        "weights": weights[1:].tolist(),
        "history": history,
    }


def compute_logistic_newton(X_train, y_train, X_test, y_test, n_iters=None):
    """Compute logistic regression using Newton-Raphson."""
    if n_iters is None:
        n_iters = NEWTON_MAX_ITERS

    # Add intercept column
    X_train = np.column_stack([np.ones(len(X_train)), X_train])
    X_test = np.column_stack([np.ones(len(X_test)), X_test])

    # Initialize weights (intercept + features)
    n_features = X_train.shape[1]
    weights = np.zeros(n_features)
    history = []

    m_train = len(y_train)

    for iteration in range(n_iters):
        z_train = X_train @ weights
        y_pred_train = sigmoid(z_train)

        # Gradient
        errors_train = y_pred_train - y_train
        gradient = (1 / m_train) * (X_train.T @ errors_train)

        # Hessian
        D = np.diag(y_pred_train * (1 - y_pred_train))
        hessian = (1 / m_train) * (X_train.T @ D @ X_train)

        # Newton update
        try:
            hessian_inv = np.linalg.inv(hessian + 1e-6 * np.eye(n_features))
            weights -= hessian_inv @ gradient
        except np.linalg.LinAlgError:
            # Fallback to gradient descent if Hessian is singular
            weights -= 0.1 * gradient

        # Store history
        z_train = X_train @ weights
        y_pred_train = sigmoid(z_train)
        z_test = X_test @ weights
        y_pred_test = sigmoid(z_test)

        log_loss_train = -np.mean(
            y_train * np.log(y_pred_train + 1e-15) + (1 - y_train) * np.log(1 - y_pred_train + 1e-15)
        )
        log_loss_test = -np.mean(y_test * np.log(y_pred_test + 1e-15) + (1 - y_test) * np.log(1 - y_pred_test + 1e-15))

        history.append(
            {
                "iteration": int(iteration),
                "intercept": float(weights[0]),
                "weights": weights[1:].tolist(),
                "log_loss_train": float(log_loss_train),
                "log_loss_test": float(log_loss_test),
            }
        )

    return {
        "intercept": float(weights[0]),
        "weights": weights[1:].tolist(),
        "history": history,
    }


def compute_sklearn_logistic(X_train, y_train, X_test, y_test):
    """Compute logistic regression using scikit-learn."""
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]

    accuracy_train = np.mean(y_pred_train == y_train)
    accuracy_test = np.mean(y_pred_test == y_test)

    return {
        "weights": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "accuracy_train": float(accuracy_train),
        "accuracy_test": float(accuracy_test),
        "y_prob_test": y_prob_test.tolist(),
    }


def compute_roc_curve(y_true, y_prob):
    """Compute ROC curve points."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc)}


# ==================== FRAME GENERATION ====================


def sample_scatter(x, y, n_samples=None, seed=None):
    """Subsample scatter data for visualization."""
    if n_samples is None:
        n_samples = SCATTER_SAMPLE_SIZE
    if seed is None:
        seed = RANDOM_SEED
    if len(x) <= n_samples:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=n_samples, replace=False)
    idx.sort()
    return x[idx], y[idx]


def sample_scatter_2d(X, y, n_samples=None, seed=None):
    """Subsample 2D scatter data for visualization."""
    if n_samples is None:
        n_samples = SCATTER_SAMPLE_SIZE
    if seed is None:
        seed = RANDOM_SEED
    if len(y) <= n_samples:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n_samples, replace=False)
    idx.sort()
    return X[idx], y[idx]


def sample_points_sequence(n_total, n_frames, min_points=10):
    """Generate sequence of increasing number of points for animation.

    Starts from very few points to show instability, growing to full dataset.
    """
    return np.linspace(min_points, n_total, n_frames, dtype=int).tolist()


def _compute_scatter_indices(n_total, n_samples=None, seed=None):
    """Sample sorted indices for progressive scatter display."""
    if n_samples is None:
        n_samples = SCATTER_SAMPLE_SIZE
    if seed is None:
        seed = RANDOM_SEED
    if n_total <= n_samples:
        return np.arange(n_total)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_samples, replace=False)
    idx.sort()
    return idx


def build_linear_frames(data_linear):
    """Build animation frames for linear regression with growing data.

    GD usa warm start desde slope=0 para mostrar convergencia en tiempo real.
    Analytical y sklearn recalculan la solución instantánea en cada frame.
    """
    x_train = data_linear["x_train"]
    y_train = data_linear["y_train"]
    x_test = data_linear["x_test"]
    y_test = data_linear["y_test"]

    n_points_seq = sample_points_sequence(len(y_train), N_FRAMES)
    scatter_idx = _compute_scatter_indices(len(x_train))

    # GD warm start: arranca desde pendiente 0
    gd_slope = 0.0
    gd_intercept = 0.0
    lr = GD_LEARNING_RATE
    iters_per_frame = max(1, GD_MAX_ITERS // N_FRAMES)

    frames = []
    for i, n_points in enumerate(n_points_seq):
        x_sub = x_train[:n_points]
        y_sub = y_train[:n_points]

        # Analytical: solución instantánea
        ana = compute_analytical_linear(x_sub, y_sub, x_test, y_test)

        # GD: iteraciones con warm start sobre datos actuales
        m = len(y_sub)
        for _ in range(iters_per_frame):
            y_pred = gd_intercept + gd_slope * x_sub
            d_intercept = -2 / m * np.sum(y_sub - y_pred)
            d_slope = -2 / m * np.sum((y_sub - y_pred) * x_sub)
            gd_intercept -= lr * d_intercept
            gd_slope -= lr * d_slope

        # MSE del estado actual de GD sobre datos completos
        gd_pred_train = gd_intercept + gd_slope * x_train
        gd_pred_test = gd_intercept + gd_slope * x_test
        gd_mse_train = float(np.mean((y_train - gd_pred_train) ** 2))
        gd_mse_test = float(np.mean((y_test - gd_pred_test) ** 2))

        frames.append(
            {
                "frame": i,
                "n_points": int(n_points),
                "n_scatter": max(2, int(np.searchsorted(scatter_idx, n_points))),
                "analytical": ana,
                "gradient_descent": {
                    "slope": float(gd_slope),
                    "intercept": float(gd_intercept),
                    "mse_train": gd_mse_train,
                    "mse_test": gd_mse_test,
                },
            }
        )

    scatter_x = x_train[scatter_idx]
    scatter_y = y_train[scatter_idx]
    analytical_full = compute_analytical_linear(x_train, y_train, x_test, y_test)
    sklearn_full = compute_sklearn_linear(x_train, y_train, x_test, y_test)

    return {
        "frames": frames,
        "scatter": {"x": scatter_x.tolist(), "y": scatter_y.tolist()},
        "analytical": analytical_full,
        "sklearn": sklearn_full,
    }


def build_polynomial_frames(data_polynomial):
    """Build animation frames for polynomial regression with growing data.

    GD usa warm start desde coeficientes=0 para cada grado.
    Las curvas arrancan planas y convergen gradualmente.
    """
    x_train = data_polynomial["x_train"]
    y_train = data_polynomial["y_train"]
    x_test = data_polynomial["x_test"]
    y_test = data_polynomial["y_test"]

    n_points_seq = sample_points_sequence(len(y_train), N_FRAMES)
    scatter_idx = _compute_scatter_indices(len(x_train))

    # Learning rate menor para polynomial (Vandermonde amplifica magnitudes)
    poly_lr = 0.005
    poly_iters_per_frame = 20

    # Warm start: coeficientes en 0 para cada grado
    gd_coeffs = {deg: np.zeros(deg + 1) for deg in POLY_DEGREES}

    # sklearn de referencia sobre datos completos
    sklearn_by_degree = {}
    for degree in POLY_DEGREES:
        sklearn_by_degree[degree] = compute_sklearn_polynomial(x_train, y_train, x_test, y_test, degree)

    frames = []
    for frame_idx, n_points in enumerate(n_points_seq):
        x_sub = x_train[:n_points]
        y_sub = y_train[:n_points]
        m = len(y_sub)

        frame = {
            "frame": frame_idx,
            "n_points": int(n_points),
            "n_scatter": max(2, int(np.searchsorted(scatter_idx, n_points))),
        }

        for degree in POLY_DEGREES:
            coeffs = gd_coeffs[degree]
            X_sub = build_vandermonde(x_sub, degree)

            # GD warm start sobre datos actuales
            for _ in range(poly_iters_per_frame):
                y_pred = X_sub @ coeffs
                gradient = (2 / m) * (X_sub.T @ (y_pred - y_sub))
                # Clamp gradiente para evitar overflow en grados altos
                gradient = np.clip(gradient, -1e6, 1e6)
                coeffs = coeffs - poly_lr * gradient

            gd_coeffs[degree] = coeffs

            # MSE sobre datos completos
            X_train_full = build_vandermonde(x_train, degree)
            X_test_full = build_vandermonde(x_test, degree)
            mse_train = float(np.mean((y_train - X_train_full @ coeffs) ** 2))
            mse_test = float(np.mean((y_test - X_test_full @ coeffs) ** 2))

            frame[f"degree_{degree}"] = {
                "n_points": int(n_points),
                "coefficients": coeffs.tolist(),
                "mse_train": mse_train,
                "mse_test": mse_test,
            }

        frames.append(frame)

    scatter_x = x_train[scatter_idx]
    scatter_y = y_train[scatter_idx]

    return {
        "frames": frames,
        "scatter": {"x": scatter_x.tolist(), "y": scatter_y.tolist()},
        "sklearn": sklearn_by_degree,
    }


def _compute_log_loss(X, y, weights, intercept):
    """Compute log loss for given weights and data."""
    z = X @ weights + intercept
    p = sigmoid(z)
    return float(-np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15)))


def _compute_log_loss_aug(X_aug, y, weights):
    """Compute log loss para matriz aumentada (con columna de intercept)."""
    z = X_aug @ weights
    p = sigmoid(z)
    return float(-np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15)))


def build_logistic_frames(data_logistic):
    """Build animation frames for logistic regression with growing data.

    GD y Newton-Raphson usan warm start desde weights=0.
    sklearn recalcula la solución instantánea en cada frame.
    """
    X_train = data_logistic["x_train"]
    y_train = data_logistic["y_train"]
    X_test = data_logistic["x_test"]
    y_test = data_logistic["y_test"]

    n_points_seq = sample_points_sequence(len(y_train), N_FRAMES, min_points=30)
    scatter_idx = _compute_scatter_indices(len(y_train))

    # Warm start: todos los pesos en 0
    n_features = X_train.shape[1] + 1  # +1 por columna de intercept
    gd_weights = np.zeros(n_features)
    newton_weights = np.zeros(n_features)
    lr = LOGISTIC_LEARNING_RATE
    gd_iters_per_frame = max(1, LOGISTIC_MAX_ITERS // N_FRAMES)
    newton_iters_per_frame = max(1, NEWTON_MAX_ITERS // N_FRAMES)

    frames = []
    for _, n_points in enumerate(n_points_seq):
        X_sub = X_train[:n_points]
        y_sub = y_train[:n_points]

        # Saltar si no hay ambas clases
        if len(np.unique(y_sub)) < 2:
            continue

        X_sub_aug = np.column_stack([np.ones(len(X_sub)), X_sub])
        X_test_aug = np.column_stack([np.ones(len(X_test)), X_test])
        m = len(y_sub)

        # GD: iteraciones con warm start
        for _ in range(gd_iters_per_frame):
            z = X_sub_aug @ gd_weights
            p = sigmoid(z)
            grad = (1 / m) * (X_sub_aug.T @ (p - y_sub))
            gd_weights -= lr * grad

        # Newton-Raphson: iteraciones con warm start
        for _ in range(newton_iters_per_frame):
            z = X_sub_aug @ newton_weights
            p = sigmoid(z)
            grad = (1 / m) * (X_sub_aug.T @ (p - y_sub))
            D = np.diag(p * (1 - p))
            hessian = (1 / m) * (X_sub_aug.T @ D @ X_sub_aug)
            try:
                h_inv = np.linalg.inv(hessian + 1e-6 * np.eye(n_features))
                newton_weights -= h_inv @ grad
            except np.linalg.LinAlgError:
                newton_weights -= 0.1 * grad

        # Log losses del estado actual
        gd_ll_train = _compute_log_loss_aug(X_sub_aug, y_sub, gd_weights)
        gd_ll_test = _compute_log_loss_aug(X_test_aug, y_test, gd_weights)
        newton_ll_train = _compute_log_loss_aug(X_sub_aug, y_sub, newton_weights)
        newton_ll_test = _compute_log_loss_aug(X_test_aug, y_test, newton_weights)

        frames.append(
            {
                "frame": len(frames),
                "n_points": int(n_points),
                "n_scatter": max(2, int(np.searchsorted(scatter_idx, n_points))),
                "gradient_descent": {
                    "weights": gd_weights[1:].tolist(),
                    "intercept": float(gd_weights[0]),
                    "log_loss_train": gd_ll_train,
                    "log_loss_test": gd_ll_test,
                },
                "newton_raphson": {
                    "weights": newton_weights[1:].tolist(),
                    "intercept": float(newton_weights[0]),
                    "log_loss_train": newton_ll_train,
                    "log_loss_test": newton_ll_test,
                },
            }
        )

    # ROC y log-loss sobre sklearn con datos completos
    sklearn_full = compute_sklearn_logistic(X_train, y_train, X_test, y_test)
    roc_curve_result = compute_roc_curve(y_test, sklearn_full["y_prob_test"])
    sk_w = np.array(sklearn_full["weights"])
    sk_b = sklearn_full["intercept"]
    sklearn_full["log_loss_train"] = _compute_log_loss(X_train, y_train, sk_w, sk_b)
    sklearn_full["log_loss_test"] = _compute_log_loss(X_test, y_test, sk_w, sk_b)

    X_scatter = X_train[scatter_idx]
    y_scatter = y_train[scatter_idx]

    return {
        "frames": frames,
        "scatter": {
            "x_medinc": X_scatter[:, 0].tolist(),
            "x_houseage": X_scatter[:, 1].tolist(),
            "y": y_scatter.tolist(),
        },
        "roc_curve": roc_curve_result,
        "sklearn": sklearn_full,
    }


def transform(extract_result):
    """Transform extracted data into frames for visualization."""
    return {
        "linear": build_linear_frames(extract_result["linear"]),
        "polynomial": build_polynomial_frames(extract_result["polynomial"]),
        "logistic": build_logistic_frames(extract_result["logistic"]),
    }
