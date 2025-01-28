"""
data_import.py

This module provides a utility function to filter a dataset based on the 'source' column.
Two datasets are available:
    1- "ba": binding affinity dataset
    2- "el": binary eluted ligand dataset

Functions:
    filter_dataset(file_path: str, source_type: str) -> pd.DataFrame
"""

import pandas as pd


def filter_dataset(pep_file_path: str, source_type: str, allele_file_path: str = None) -> pd.DataFrame:
    """
    Reads a dataset from a file and filters it based on the 'source' column.

    Args:
        pep_file_path (str): Path to the peptide dataset file.
        source_type (str): The filter criteria for the 'source' column. Can be 'ba' or 'el'.
        allele_file_path (str, optional): Path to the file with allele sequences (required for 'el').

    Returns:
        pd.DataFrame: A filtered DataFrame:
                      - For 'ba': Includes peptides where 'allele' starts with "HLA" and 'source' matches 'ba'.
                      - For 'el': Includes peptides where 'allele' starts with "HLA", 'source' matches 'el',
                                  and merges with allele sequences.

    Raises:
        ValueError: If the source_type is not 'ba' or 'el'.
        ValueError: If allele_file_path is not provided for 'el'.
    """
    if source_type not in ["ba", "el"]:
        raise ValueError("source_type must be 'ba' or 'el'")

    # read the peptide dataset
    dataset = pd.read_csv(pep_file_path, header=0, sep="\t")

    # filter the dataset to keep human alleles and respective source type
    filtered_dataset = dataset.loc[
        (dataset.allele.str.startswith("HLA")) &
        (dataset.source == source_type) &
        (pd.to_numeric(dataset.length, errors='coerce') < 13)
    ]

    # merge with allele sequences only for 'el'
    if source_type == "el":
        if not allele_file_path:
            raise ValueError("allele_file_path is required for 'el' source type.")
        
        # read allele sequences
        allele_seq = pd.read_csv(allele_file_path, sep=" ", header=None)
        allele_seq = allele_seq.rename(columns={0: "allele", 1: "seq"})

        # merge and drop rows with missing values where allele seq is not found
        filtered_dataset = pd.merge(
            left=filtered_dataset,
            right=allele_seq,
            left_on="allele",
            right_on="allele",
            how="left"
        ).dropna(how="any", axis=0)

        # rename columns for clarity
        filtered_dataset = filtered_dataset.rename(columns={"seq_x": "pep", "seq_y": "allele_seq"})

    return filtered_dataset