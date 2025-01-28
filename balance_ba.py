"""
balance_ba.py

This module provides a function to balance binding affinity data for a specific allele.
It divides binding affinity into specified bins, filters the data for a single allele, and
upsamples minority bins to match the size of the largest bin.

Functions:
    balance_binding_affinity(df: pd.DataFrame, allele: str) -> pd.DataFrame
"""

import pandas as pd
from sklearn.utils import resample

def balance_binding_affinity(df: pd.DataFrame, allele: str) -> pd.DataFrame:
    """
    Balance binding affinity data for a specified allele.

    Parameters:
        df (pd.DataFrame): A DataFrame with columns `allele` and `ba` (binding affinity, range 0-1).
        allele (str): The allele for which to balance the data.

    Returns:
        pd.DataFrame: A balanced DataFrame with upsampled binding affinity bins.
    """
    # ensure `ba` is numeric
    df['ba'] = pd.to_numeric(df['ba'], errors='coerce')
    
    # filter for the specified allele
    df_allele = df[df['allele'] == allele]

    if df_allele.empty:
        raise ValueError(f"No data found for allele {allele}. Please check your dataset and allele name.")

    # define bins and labels
    bins = [0, 0.2, 0.4, 0.45, 0.65, 1]
    labels = ['0-0.2', '0.2-0.4', '0.4-0.45', '0.45-0.65', '0.65-1']

    # bin the binding affinity data
    df_allele['ba_bin'] = pd.cut(df_allele['ba'], bins=bins, labels=labels, include_lowest=True)

    # get counts for each bin
    bin_counts = df_allele['ba_bin'].value_counts()
    print(f"Binding affinity bin counts for allele '{allele}':\n{bin_counts}")

    # determine the size to upsample to (maximum bin size)
    target_size = bin_counts.max()

    # upsample each bin
    balanced_bins = []
    for bin_label in labels:
        bin_data = df_allele[df_allele['ba_bin'] == bin_label]
        if not bin_data.empty:
            if bin_label == '0.4-0.45':  # keep this bin as is
                balanced_bins.append(bin_data)
            else:  # upsample other bins
                upsampled_bin = resample(bin_data, replace=True, n_samples=target_size, random_state=42)
                balanced_bins.append(upsampled_bin)

    # combine all bins and shuffle
    balanced_df = pd.concat(balanced_bins).sample(frac=1, random_state=42).reset_index(drop=True)

    # drop the `ba_bin` column used for balancing
    balanced_df.drop(columns=['ba_bin'], inplace=True)

    print(f"Balanced dataset shape for allele '{allele}': {balanced_df.shape}")
    return balanced_df
