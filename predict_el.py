import os
import numpy as np
import pandas as pd
import tensorflow as tf
from blosum_func import fast_combined_blosum_encoding
from get_model_el import get_model 


allele_file_path="./NetMHCpan-4.1_train_data/training_allele_seq.pseudo"
model_path="./saved_models_el/model_fold_4.h5"

def predict_el(df, model_path=model_path, peptide_col="peptide", allele_col="HLA", pred_col="el_pred", allele_file_path=allele_file_path):
    """
    Loads a trained model, sanitizes allele column, merges with allele sequences, encodes peptides and alleles using 
    combined BLOSUM encoding, and predicts eluted ligand values.

    Args:
        model_path (str): Path to the trained Keras model file.
        df (pd.DataFrame): DataFrame containing peptides and allele sequences to predict.
        peptide_col (str): Column name in the DataFrame containing peptide sequences. Default is "peptide".
        allele_col (str): Column name in the DataFrame containing allele sequences. Default is "HLA".
        pred_col (str): Column name where predictions will be stored. Default is "el_pred".
        allele_file_path (str): Path to the allele file containing allele names and sequences. Default is None.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column containing predictions.
    """
    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # loading the model fails
    # model will be rebuilt again and only the weights will be loaded
    model = get_model()
    model.load_weights(model_path)
    print(f"Model loaded from {model_path}")

    # check if the required columns exist in the DataFrame
    if peptide_col not in df.columns:
        raise ValueError(f"Column '{peptide_col}' not found in the DataFrame")
    if allele_col not in df.columns:
        raise ValueError(f"Column '{allele_col}' not found in the DataFrame")

    # Sanitize the HLA column to remove "*" characters
    df[allele_col] = df[allele_col].str.replace("*", "", regex=False)

    if allele_file_path:
        # Load the allele sequences from the provided file
        allele_seq = pd.read_csv(allele_file_path, sep=" ", header=None)
        allele_seq = allele_seq.rename(columns={0: "allele", 1: "seq"})

        # Merge the allele sequences with the dataset
        df = pd.merge(
            left=df,
            right=allele_seq,
            left_on=allele_col,
            right_on="allele",
            how="left"
        ).dropna(how="any", axis=0)
        print(f"Allele sequences merged successfully.")

    # encode peptides and alleles using combined BLOSUM encoding
    df["encoded_input"] = df.apply(
        lambda row: fast_combined_blosum_encoding(row[peptide_col], row["seq"]), axis=1
    )

    # convert encoded inputs to a numpy array for prediction
    X = np.stack(df["encoded_input"].values)

    # make predictions
    predictions = model.predict(X, verbose=1)

    # store predictions in the DataFrame
    df[pred_col] = predictions.squeeze()

    # drop the intermediate encoded_input column
    df.drop(columns=["encoded_input"], inplace=True)

    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict eluted ligand values using a trained model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file containing peptide and allele sequences.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output CSV file with predictions.")
    parser.add_argument("--peptide_col", type=str, default="peptide", help="Name of the column containing peptide sequences. Default is 'pep'.")
    parser.add_argument("--allele_col", type=str, default="HLA", help="Name of the column containing allele sequences. Default is 'HLA'.")
    parser.add_argument("--pred_col", type=str, default="el_pred", help="Name of the column to store predictions. Default is 'el_pred'.")
    parser.add_argument("--allele_file_path", type=str, default=allele_file_path, required=False, help="Path to the allele sequence file. Default is None.")

    args = parser.parse_args()

    # load input data
    input_df = pd.read_csv(args.input_file)

    # run predictions
    output_df = predict_el(
        model_path=args.model_path,
        df=input_df,
        peptide_col=args.peptide_col,
        allele_col=args.allele_col,
        pred_col=args.pred_col,
        allele_file_path=args.allele_file_path
    )

    # save output data
    output_df.to_csv(args.output_file, index=False)
    print(f"Predictions saved to {args.output_file}")

    # commands that were run
    # python predict_el.py --model_path .\saved_models_el\model_fold_4.h5 --input_file .\test_data_EL\A2601.csv --output_file .\pred_EL\A2601_pred.csv
    # python predict_el.py --model_path .\saved_models_el\model_fold_4.h5 --input_file .\test_data_EL\A3201.csv --output_file .\pred_EL\A3201_pred.csv
    # python predict_el.py --model_path .\saved_models_el\model_fold_4.h5 --input_file .\test_data_EL\B4801.csv --output_file .\pred_EL\B4801_pred.csv
