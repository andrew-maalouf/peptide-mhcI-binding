"""
blosum_func.py

This module provides utilities for encoding peptide and allele sequences 
using the BLOSUM50 substitution matrix.

Functions:
    fast_get_blosum_vector(aa)
    fast_peptide_to_blosum(seq, target_length=12)
    fast_allele_to_blosum(allele_seq)
    fast_combined_blosum_encoding(peptide_seq, allele_seq)
"""

import numpy as np
from Bio.Align import substitution_matrices

# initialize global variables
blosum50 = None
amino_acids = None
blosum_dict = None


def _initialize_blosum():
    """
    Initialize the BLOSUM50 matrix and precompute lookup dictionaries.
    
    This function is called automatically on module import.
    """
    global blosum50, amino_acids, blosum_dict
    blosum50 = substitution_matrices.load("BLOSUM50")
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    
    # Use np.int16 for more memory-efficient storage of matrix values
    blosum_dict = {aa: np.array([blosum50[aa][other_aa] for other_aa in amino_acids], dtype=np.int16) for aa in amino_acids}


# automatically initialize when the module is imported to make sure global variables are available for functions
_initialize_blosum()


def fast_get_blosum_vector(aa: str) -> np.ndarray:
    """
    Retrieve the BLOSUM50 encoding vector for a given amino acid.
    
    Args:
        aa (str): A single amino acid character.
        
    Returns:
        np.ndarray: The BLOSUM50 encoding vector of length 20 for the amino acid.
    """
    return blosum_dict.get(aa, np.zeros(len(amino_acids), dtype=np.int16))


def fast_peptide_to_blosum(seq: str, target_length: int = 12) -> np.ndarray:
    """
    Encode a peptide sequence using BLOSUM50 and pad/truncate to the target length.
    
    Args:
        seq (str): The peptide sequence to encode.
        target_length (int, optional): The fixed length to pad/truncate to. Defaults to 12.
        
    Returns:
        np.ndarray: A (target_length x 20) array representing the encoded peptide sequence.
    """
    blosum_array = np.array([fast_get_blosum_vector(aa) for aa in seq], dtype=np.int16)
    
    # Create a zero matrix of the target size, with dtype set to int16 to save memory
    padded_array = np.zeros((target_length, len(amino_acids)), dtype=np.int16)
    padded_array[:min(len(seq), target_length)] = blosum_array[:target_length]
    
    return padded_array


def fast_allele_to_blosum(allele_seq: str) -> np.ndarray:
    """
    Encode an allele sequence using BLOSUM50.
    
    Assumes the allele sequence has a fixed length of 34.
    
    Args:
        allele_seq (str): The allele sequence to encode.
        
    Returns:
        np.ndarray: A (34 x 20) array representing the encoded allele sequence.
    """
    if len(allele_seq) != 34:
        raise ValueError("Allele sequence must be represented with a length of 34 amino acids.")
    return np.array([fast_get_blosum_vector(aa) for aa in allele_seq], dtype=np.int16)


def fast_combined_blosum_encoding(peptide_seq: str, allele_seq: str) -> np.ndarray:
    """
    Combine encoded peptide and allele sequences into a single array.
    
    Args:
        peptide_seq (str): The peptide sequence to encode.
        allele_seq (str): The allele sequence to encode.
        
    Returns:
        np.ndarray: A ((target_length + 34) x 20) array representing the combined encoding.
    """
    peptide_encoding = fast_peptide_to_blosum(peptide_seq, target_length=12)
    allele_encoding = fast_allele_to_blosum(allele_seq)
    return np.vstack([peptide_encoding, allele_encoding])
