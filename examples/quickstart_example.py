"""
Quick start example for pca-africa library.
"""

import numpy as np
from pca_africa import optimized_pca, plot_scree, explain_variance

# Generate sample data
np.random.seed(42)
data = np.random.randn(100, 10)

# Perform PCA
result = optimized_pca(data, threshold=0.95)

# Display results
explain_variance(result)

# Visualize
plot_scree(result)

print(f"\nOriginal shape: {data.shape}")
print(f"Reduced shape: {result['reduced_data'].shape}")
