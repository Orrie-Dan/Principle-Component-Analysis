"""
PCA Africa - A Python library for Principal Component Analysis
focused on African CO2 emissions data.
"""

__version__ = "0.1.0"
__author__ = "Nkusi Orrie Dan"

from .pca import optimized_pca, naive_pca, explain_variance
from .visualization import plot_scree, plot_before_after_pca, plot_performance
from .utils import load_data, clean_data, standardize_data

__all__ = [
    'optimized_pca',
    'naive_pca',
    'explain_variance',
    'plot_scree',
    'plot_before_after_pca',
    'plot_performance',
    'load_data',
    'clean_data',
    'standardize_data'
]
