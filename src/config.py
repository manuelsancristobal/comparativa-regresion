"""Configuration for Comparativa Regresion project."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_PROCESSED = DATA_DIR / "processed"
VIZ_DIR = PROJECT_ROOT / "viz"
TESTS_DIR = PROJECT_ROOT / "tests"
JEKYLL_DIR = PROJECT_ROOT / "jekyll"

# Create directories if they don't exist
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
(VIZ_DIR / "assets" / "data").mkdir(parents=True, exist_ok=True)

# Jekyll repo configuration (opcional; solo necesario para deploy)
_jekyll_env = os.getenv("JEKYLL_REPO")
JEKYLL_REPO: Path | None = Path(_jekyll_env) if _jekyll_env else None
JEKYLL_BASE = (JEKYLL_REPO / "proyectos" / "comparativa-regresion") if JEKYLL_REPO else None
JEKYLL_DATA = (JEKYLL_BASE / "assets" / "data") if JEKYLL_BASE else None
JEKYLL_CSS = (JEKYLL_BASE / "assets" / "css") if JEKYLL_BASE else None
JEKYLL_JS = (JEKYLL_BASE / "assets" / "js") if JEKYLL_BASE else None
JEKYLL_PAGE = (JEKYLL_REPO / "_projects") if JEKYLL_REPO else None

# Colors by method
COLORS = {
    "analytical": "#1f77b4",  # blue
    "gradient_descent": "#d62728",  # red
    "sklearn": "#2ca02c",  # green
    "vandermonde": "#1f77b4",  # blue (degree 2)
    "polynomial_2": "#1f77b4",  # blue
    "polynomial_3": "#ff7f0e",  # orange
    "polynomial_4": "#d62728",  # red
    "newton_raphson": "#1f77b4",  # blue
    "logistic_gd": "#d62728",  # red
    "logistic_newton": "#1f77b4",  # blue
    "logistic_sklearn": "#2ca02c",  # green
    "cara": "#2ca02c",  # green (expensive)
    "barata": "#d62728",  # red (cheap)
}

# Regression parameters
GD_LEARNING_RATE = 0.01
GD_MAX_ITERS = 1000
LOGISTIC_LEARNING_RATE = 0.1
LOGISTIC_MAX_ITERS = 5000
NEWTON_MAX_ITERS = 20
POLY_DEGREES = [1, 2, 3, 4]
N_FRAMES = 100

# Data processing
TRAIN_TEST_SPLIT = 0.8
RANDOM_SEED = 42

# Sampling for visualization
SCATTER_SAMPLE_SIZE = 2000  # Stratified sample for D3.js scatter plots
