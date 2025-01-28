"""
train_split.py

This module provides a utility function to split a DataFrame into training and testing datasets, 
suitable for machine learning models.

Functions:
    split_train_test(df: pd.DataFrame, input_col: str, target_col: str, test_size: float = 0.05, random_state: int = 21)
"""

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


def split_train_test(
    df: pd.DataFrame,
    input_col: str,
    target_col: str,
    test_size: float = 0.05,
    random_state: int = 21
):
    """
    Splits the input DataFrame into training and testing datasets for the specified input and target columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        input_col (str): Name of the column to be used as input features.
        target_col (str): Name of the column to be used as target labels.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.05.
        random_state (int): Random state for reproducibility. Default is 21.

    Returns:
        tuple: Four numpy arrays - (x_train, x_test, y_train, y_test).
    """
    if input_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Columns {input_col} and/or {target_col} not found in DataFrame.")

    # split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        df[input_col], df[target_col], test_size=test_size, random_state=random_state
    )

    # convert the split data to numpy arrays
    x_train = np.stack(x_train)
    x_test = np.stack(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test
