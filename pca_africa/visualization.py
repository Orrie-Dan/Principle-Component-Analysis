"""
Visualization functions for PCA results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional


def plot_scree(result: Dict, figsize: tuple = (12, 5)) -> None:
    """
    Create a scree plot showing explained variance.

    Parameters
    ----------
    result : dict
        Result from optimized_pca()
    figsize : tuple, optional
        Figure size (default: (12, 5))
    """
    sns.set_theme(style="whitegrid")

    fig, ax1 = plt.subplots(figsize=figsize)

    n_components = len(result['eigenvalues'])
    components_range = range(1, n_components + 1)

    # Bar chart
    color_bar = '#2ecc71'
    ax1.bar(components_range, result['explained_variance_ratio'] * 100,
            color=color_bar, alpha=0.7, label='Individual Variance')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance (%)', fontsize=12, color=color_bar)
    ax1.tick_params(axis='y', labelcolor=color_bar)

    # Line chart
    ax2 = ax1.twinx()
    color_line = '#e74c3c'
    ax2.plot(components_range, result['cumulative_variance'] * 100,
             color=color_line, marker='o', linewidth=2, label='Cumulative Variance')
    ax2.axhline(y=95, color='blue', linestyle='--', linewidth=1.5, label='95% Threshold')
    ax2.axvline(x=result['num_components'], color='purple', linestyle=':',
                linewidth=2, label=f'Selected: PC{result["num_components"]}')
    ax2.set_ylabel('Cumulative Variance (%)', fontsize=12, color=color_line)
    ax2.tick_params(axis='y', labelcolor=color_line)
    ax2.set_ylim(0, 105)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

    plt.title('PCA Scree Plot — Explained Variance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_before_after_pca(original_data: np.ndarray,
                          reduced_data: np.ndarray,
                          labels: Optional[np.ndarray] = None,
                          feature_names: Optional[tuple] = None,
                          figsize: tuple = (16, 6)) -> None:
    """
    Plot original vs PCA-reduced data.

    Parameters
    ----------
    original_data : np.ndarray
        Standardized original data
    reduced_data : np.ndarray
        PCA-reduced data
    labels : np.ndarray, optional
        Labels for coloring points
    feature_names : tuple of (str, str), optional
        Names for the first two features (x_axis, y_axis) on the left plot
    figsize : tuple, optional
        Figure size (default: (16, 6))
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10.colors

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax1.scatter(original_data[mask, 0], original_data[mask, 1],
                       c=[colors[i % len(colors)]], label=label,
                       alpha=0.6, edgecolors='k', linewidths=0.5, s=40)
            ax2.scatter(reduced_data[mask, 0], reduced_data[mask, 1],
                       c=[colors[i % len(colors)]], label=label,
                       alpha=0.6, edgecolors='k', linewidths=0.5, s=40)
    else:
        ax1.scatter(original_data[:, 0], original_data[:, 1], alpha=0.6)
        ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.6)

    xlabel = feature_names[0] if feature_names else 'Feature 1'
    ylabel = feature_names[1] if feature_names else 'Feature 2'
    ax1.set_title('Original Data\n(First 2 Features)', fontsize=13, fontweight='bold')
    ax1.set_xlabel(xlabel, fontsize=11)
    ax1.set_ylabel(ylabel, fontsize=11)
    if labels is not None:
        ax1.legend(fontsize=8, loc='best')

    ax2.set_title('PCA Reduced Data\n(PC1 vs PC2)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Principal Component 1', fontsize=11)
    ax2.set_ylabel('Principal Component 2', fontsize=11)
    if labels is not None:
        ax2.legend(fontsize=8, loc='best')

    plt.suptitle('PCA Dimensionality Reduction', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_performance(naive_time: float, optimized_time: float,
                    result: Dict, figsize: tuple = (14, 5)) -> None:
    """
    Plot performance comparison.

    Parameters
    ----------
    naive_time : float
        Naive PCA execution time
    optimized_time : float
        Optimized PCA execution time
    result : dict
        PCA result dictionary
    figsize : tuple, optional
        Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Bar chart
    methods = ['Naive PCA', 'Optimized PCA']
    times = [naive_time * 1000, optimized_time * 1000]
    bar_colors = ['#e74c3c', '#2ecc71']

    bars = axes[0].bar(methods, times, color=bar_colors, width=0.4,
                       edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Execution Time (ms)', fontsize=12)
    axes[0].set_title('Performance Comparison', fontsize=13, fontweight='bold')

    for bar, t in zip(bars, times):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f'{t:.2f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Pie chart
    selected_var = result['cumulative_variance'][result['num_components'] - 1] * 100
    remaining_var = 100 - selected_var

    sizes = [selected_var, remaining_var]
    labels = [f'Selected {result["num_components"]} PCs\n({selected_var:.1f}%)',
              f'Remaining\n({remaining_var:.1f}%)']
    colors = ['#3498db', '#bdc3c7']
    explode = (0.05, 0)

    axes[1].pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='', startangle=90, shadow=True)
    axes[1].set_title('Variance Captured', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.show()
