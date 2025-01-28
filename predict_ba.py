import os
import numpy as np
import pandas as pd
import tensorflow as tf
from blosum_func import fast_peptide_to_blosum
from tensorflow.keras import layers, backend as K


# first define custom loss functio to be able to load the model
def custom_logarithmic_loss(y_true, y_pred):
    """
    Custom logarithmic loss function for binding affinity prediction.

    Args:
        y_true (tf.Tensor): True labels (ground truth).
        y_pred (tf.Tensor): Predicted labels from the model.

    Returns:
        tf.Tensor: Computed loss value.
    """
    # clip predictions to prevent log(0) issues
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = -K.log(1 - K.square(y_pred - y_true))
    return K.mean(loss)


def predict_binding_affinity(model_path, df, peptide_col="peptide", pred_col="my_pred"):
    """
    Loads a trained model, encodes peptides using BLOSUM, and predicts binding affinities.

    Args:
        model_path (str): Path to the trained Keras model file.
        df (pd.DataFrame): DataFrame containing the peptides to predict.
        peptide_col (str): Column name in the DataFrame containing peptide sequences. Default is "peptide".
        pred_col (str): Column name where predictions will be stored. Default is "my_pred".

    Returns:
        pd.DataFrame: The input DataFrame with an additional column containing predictions.
    """
    # check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # load the model
    model = tf.keras.models.load_model(model_path, custom_objects={'custom_logarithmic_loss':custom_logarithmic_loss})
    print(f"Model loaded from {model_path}")

    # check if the peptide column exists in the DataFrame
    if peptide_col not in df.columns:
        raise ValueError(f"Column '{peptide_col}' not found in the DataFrame")

    # encode peptides using BLOSUM
    df["encoded_peptide"] = df[peptide_col].apply(fast_peptide_to_blosum)

    # Convert encoded peptides to a numpy array for prediction
    X = np.stack(df["encoded_peptide"].values)

    # make predictions
    predictions = model.predict(X, verbose=1)

    # store predictions in the DataFrame
    df[pred_col] = predictions.squeeze()

    # drop the intermediate encoded_peptide column
    df.drop(columns=["encoded_peptide"], inplace=True)

    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict binding affinity using a trained model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file containing peptide sequences.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output CSV file with predictions.")
    parser.add_argument("--peptide_col", type=str, default="peptide", help="Name of the column containing peptide sequences. Default is 'peptide'.")
    parser.add_argument("--pred_col", type=str, default="my_pred", help="Name of the column to store predictions. Default is 'my_pred'.")

    args = parser.parse_args()

    # load input data
    input_df = pd.read_csv(args.input_file)

    # run predictions
    output_df = predict_binding_affinity(
        model_path=args.model_path,
        df=input_df,
        peptide_col=args.peptide_col,
        pred_col=args.pred_col
    )

    # save output data
    output_df.to_csv(args.output_file, index=False)
    print(f"Predictions saved to {args.output_file}")


    # commands that were run in terminal:
    #python predict_ba.py --model_path .\saved_models_ba\HLA-B48_01.keras --input_file .\test_data\B4801.csv --output_file .\pred\B4801_pred.csv
    #python predict_ba.py --model_path .\saved_models_ba\HLA-A74_01.keras --input_file .\test_data\A7401.csv --output_file .\pred\A7401_pred.csv
    #python predict_ba.py --model_path .\saved_models_ba\HLA-A26_01.keras --input_file .\test_data\A2601.csv --output_file .\pred\A2601_pred.csv
    #python predict_ba.py --model_path .\saved_models_ba\HLA-A32_01.keras --input_file .\test_data\A3201.csv --output_file .\pred\A3201_pred.csv