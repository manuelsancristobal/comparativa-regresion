"""Pytest configuration and fixtures."""

import pytest

from src.etl.extract import extract


@pytest.fixture
def extracted_data():
    """Load extracted data for tests."""
    return extract()


@pytest.fixture
def linear_data(extracted_data):
    """Get linear regression data."""
    return extracted_data["linear"]


@pytest.fixture
def polynomial_data(extracted_data):
    """Get polynomial regression data."""
    return extracted_data["polynomial"]


@pytest.fixture
def logistic_data(extracted_data):
    """Get logistic regression data."""
    return extracted_data["logistic"]
