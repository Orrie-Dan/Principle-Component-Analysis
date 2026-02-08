"""
Utility functions for data loading and preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load CO2 emissions dataset.

    Parameters
    ----------
    filepath : str
        Path to CSV file

    Returns
    -------
    pd.DataFrame
        Loaded dataframe
    """
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame, drop_threshold: int = 500) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Clean dataset by handling missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    drop_threshold : int, optional
        Minimum missing values to drop a column (default: 500)

    Returns
    -------
    tuple
        (cleaned numeric dataframe, labels dataframe or None)
    """
    # Save labels
    labels = df[['Country', 'Sub-Region']].copy() if 'Country' in df.columns else None

    # Select numeric columns
    df_numeric = df.select_dtypes(include=[np.number]).copy()

    # Drop columns with too many missing values
    missing_counts = df_numeric.isnull().sum()
    cols_to_drop = missing_counts[missing_counts >= drop_threshold].index
    df_numeric = df_numeric.drop(columns=cols_to_drop)

    # Fill remaining missing values
    df_numeric = df_numeric.fillna(df_numeric.mean())

    return df_numeric, labels


def standardize_data(data: np.ndarray) -> np.ndarray:
    """
    Standardize data to mean=0, std=1.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix

    Returns
    -------
    np.ndarray
        Standardized data
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1
    return (data - mean) / std
