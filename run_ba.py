import os
from train_model_ba import do_it_all_ba

# Define the file paths and output directory
pep_file_path = "./NetMHCpan-4.1_train_data/netcpan4.1_train.tsv" 
allele_file_path = "./NetMHCpan-4.1_train_data/training_allele_seq.pseudo"
save_dir = "saved_models_ba"

do_it_all_ba(
    pep_file_path = pep_file_path, 
    allele_file_path = allele_file_path,
    allele = "HLA-A26:01", 
    save_dir = save_dir, 
    source_type="ba", 
    batch_size=128, 
    epochs=200, 
    model_name= "HLA-A26:01"
)

do_it_all_ba(
    pep_file_path = pep_file_path, 
    allele_file_path = allele_file_path,
    allele = "HLA-A32:01", 
    save_dir = save_dir, 
    source_type="ba", 
    batch_size=128, 
    epochs=200, 
    model_name= "HLA-A32:01"
)

do_it_all_ba(
    pep_file_path = pep_file_path, 
    allele_file_path = allele_file_path,
    allele = "HLA-A74:01", 
    save_dir = save_dir, 
    source_type="ba", 
    batch_size=128, 
    epochs=200, 
    model_name= "HLA-A74:01"
)

do_it_all_ba(
    pep_file_path = pep_file_path, 
    allele_file_path = allele_file_path,
    allele = "HLA-B48:01", 
    save_dir = save_dir, 
    source_type="ba", 
    batch_size=128, 
    epochs=200, 
    model_name= "HLA-B48:01"
)
