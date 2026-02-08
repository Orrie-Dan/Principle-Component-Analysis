"""
Core PCA implementation functions.
"""

import numpy as np
from typing import Dict, Tuple


def optimized_pca(data: np.ndarray, threshold: float = 0.95) -> Dict:
    """
    Perform optimized PCA on input data.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    threshold : float, optional
        Variance threshold for component selection (default: 0.95)

    Returns
    -------
    dict
        Dictionary containing:
        - 'reduced_data': Projected data
        - 'eigenvalues': Sorted eigenvalues
        - 'eigenvectors': Sorted eigenvectors
        - 'explained_variance_ratio': Variance explained per component
        - 'cumulative_variance': Cumulative variance explained
        - 'num_components': Number of components selected

    Examples
    --------
    >>> import numpy as np
    >>> from pca_africa import optimized_pca
    >>> data = np.random.randn(100, 10)
    >>> result = optimized_pca(data, threshold=0.95)
    >>> print(f"Components: {result['num_components']}")
    """
    # Standardize
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1
    standardized = (data - mean) / std

    # Covariance matrix
    cov_matrix = np.cov(standardized, rowvar=False)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort descending
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Calculate variance ratios
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Dynamic component selection
    num_components = np.argmax(cumulative_variance >= threshold) + 1

    # Project data
    reduced_data = np.dot(standardized, eigenvectors[:, :num_components])

    return {
        'reduced_data': reduced_data,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'num_components': num_components,
        'mean': mean,
        'std': std
    }


def naive_pca(data: np.ndarray, num_components: int) -> np.ndarray:
    """
    Naive PCA implementation (for comparison).

    Parameters
    ----------
    data : np.ndarray
        Input data matrix
    num_components : int
        Number of components to keep

    Returns
    -------
    np.ndarray
        Reduced data matrix
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1
    standardized = (data - mean) / std

    cov_matrix = np.cov(standardized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    full_projection = np.dot(standardized, eigenvectors)
    return full_projection[:, :num_components]


def explain_variance(result: Dict) -> None:
    """
    Print variance explanation summary.

    Parameters
    ----------
    result : dict
        Result dictionary from optimized_pca()
    """
    print("=" * 60)
    print("PCA Variance Explained Summary")
    print("=" * 60)
    print(f"Components selected: {result['num_components']}")
    print(f"Total variance explained: {result['cumulative_variance'][result['num_components']-1]*100:.2f}%")
    print("\nTop 5 Components:")
    for i in range(min(5, len(result['eigenvalues']))):
        print(f"  PC{i+1}: {result['explained_variance_ratio'][i]*100:.2f}%")
    print("=" * 60)
