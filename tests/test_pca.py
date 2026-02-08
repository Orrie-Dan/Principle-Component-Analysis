"""
Minimal tests for pca_africa package.
"""

import numpy as np
import pytest
from pca_africa import optimized_pca, explain_variance


def test_optimized_pca_returns_dict():
    """optimized_pca returns a dict with expected keys."""
    data = np.random.randn(20, 5)
    result = optimized_pca(data, threshold=0.95)
    assert isinstance(result, dict)
    assert 'reduced_data' in result
    assert 'eigenvalues' in result
    assert 'num_components' in result
    assert 'cumulative_variance' in result


def test_optimized_pca_reduced_shape():
    """Reduced data has correct shape (n_samples, num_components)."""
    data = np.random.randn(20, 5)
    result = optimized_pca(data, threshold=0.95)
    n_samples = data.shape[0]
    n_comp = result['num_components']
    assert result['reduced_data'].shape == (n_samples, n_comp)


def test_optimized_pca_num_components():
    """num_components is at least 1 and at most n_features."""
    data = np.random.randn(20, 5)
    result = optimized_pca(data, threshold=0.95)
    assert 1 <= result['num_components'] <= 5


def test_explain_variance_runs_without_error(capsys):
    """explain_variance(result) runs without raising."""
    data = np.random.randn(20, 5)
    result = optimized_pca(data, threshold=0.95)
    explain_variance(result)
    out, _ = capsys.readouterr()
    assert "PCA Variance Explained Summary" in out
    assert "Components selected" in out
