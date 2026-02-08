# pca-africa

A Python library for Principal Component Analysis focused on African CO2 emissions data.

## Installation

### Option 1: Install from GitHub

```bash
pip install git+https://github.com/Orrie-Dan/Principle-Component-Analysis.git
```

### Option 2: Install from Source

```bash
git clone https://github.com/Orrie-Dan/Principle-Component-Analysis.git
cd Principle-Component-Analysis
pip install -e .
```

## Quick Start

```python
from pca_africa import optimized_pca, plot_scree, explain_variance
import numpy as np

# Load your data
data = np.random.randn(100, 16)  # Replace with your actual data

# Perform PCA
result = optimized_pca(data, threshold=0.95)

# View results
explain_variance(result)
plot_scree(result)
```

## Full Example

See [examples/quickstart_example.py](examples/quickstart_example.py) for a complete example.

## Development

```bash
# Clone the repo
git clone https://github.com/Orrie-Dan/Principle-Component-Analysis.git
cd Principle-Component-Analysis

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```
