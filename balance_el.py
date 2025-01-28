"""
balance_el.py

This module provides utilities for preprocessing peptide binding data, filtering alleles based on their 
row count, and balancing the binders and non-binders (through upsampling binders and downsampling non-binders).

Functions:
    preprocess_peptide_data(df: pd.DataFrame) -> pd.DataFrame
"""

import pandas as pd
from sklearn.utils import resample

def preprocess_peptide_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess peptide binding data by filtering alleles, balancing binders and non-binders across 
    the entire dataset, and merging everything into a single balanced dataset.

    This function filters alleles with more than 50 rows, separates binders and non-binders, 
    and then upsamples binders and downsamples non-binders to ensure a balanced dataset.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'peptide', 'binding', and 'allele' columns.

    Returns:
        pd.DataFrame: A processed and balanced DataFrame where binders and non-binders have equal
                      representation.
    """
    
    # Filter alleles with more than 50 rows
    allele_counts = df['allele'].value_counts()
    filtered_alleles = allele_counts[allele_counts > 50].index
    df = df[df['allele'].isin(filtered_alleles)]
    
    # Separate the data into binders (ba == 1) and non-binders (ba == 0)
    pos_el = df[df['ba'] == 1]  # Binders
    neg_el = df[df['ba'] == 0]  # Non-binders

    # Upsample binders (positive samples) to match the desired number of samples
    up_pos_el = resample(pos_el, replace=True, n_samples=1000000, random_state=42)

    # Downsample non-binders (negative samples) to match the desired number of samples
    # dw_neg_el = resample(neg_el, replace=False, n_samples=1000000, random_state=42)

    # Combine the upsampled binders and downsampled non-binders into one dataset
    balanced_el = pd.concat([up_pos_el, neg_el])
    
    # Shuffle the final dataset to ensure random order
    balanced_el = balanced_el.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_el
