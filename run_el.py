import os
import numpy as np
from data_import import filter_dataset
from balance_el import preprocess_peptide_data
from blosum_func import fast_combined_blosum_encoding
from train_model_el import balanced_5_fold_cv_tf

# Define the file paths
pep_file_path = "./NetMHCpan-4.1_train_data/netcpan4.1_train.tsv" 
allele_file_path = "./NetMHCpan-4.1_train_data/training_allele_seq.pseudo"

# Specify the source type 
source_type = 'el'

# Filter dataset for the 'el' source type
filtered_data = filter_dataset(pep_file_path, source_type, allele_file_path)

# Perform Upsampling
filtered_data = preprocess_peptide_data(filtered_data)

# Apply encoding
filtered_data['input'] = [
    fast_combined_blosum_encoding(row['pep'], row['allele_seq']) 
    for _, row in filtered_data.iterrows()
]

# Define the input shape 
input_shape = (46, 20)  

# Directory to save the models
save_dir = "saved_models_el"

# Prepare input
X = filtered_data["input"]
X = np.stack(X)
y = filtered_data["ba"]
y = np.array(y)

# Start cross-validation
fold_scores, best_model_path = balanced_5_fold_cv_tf(
    X, y, save_dir=save_dir, k=5, input_shape=input_shape, batch_size=32, epochs=200
)

# Print the results
print(f"\nMean Accuracy from 5-Fold Cross Validation: {np.mean(fold_scores):.4f}")
print(f"Best Model saved at: {best_model_path}")