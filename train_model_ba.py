"""
train_model_ba.py

This module provides functionality to train a Keras model for binding affinity prediction
and save the trained model to a specified directory.

Functions:
    train_model(X, y, save_dir, model, batch_size, epochs, name):
        Trains the given model on the provided data and saves the trained model.
"""

import os
import tensorflow as tf
import numpy as np
from data_import import filter_dataset
from balance_ba import balance_binding_affinity
from blosum_func import fast_peptide_to_blosum
from get_model_ba import get_ba_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping



def train_model(X, y, save_dir, model, batch_size=128, epochs=100, name="model"):
    """
    Trains the provided Keras model on the given dataset and saves it to the specified directory.

    Args:
        X (numpy.ndarray or tf.Tensor): Input data for training.
        y (numpy.ndarray or tf.Tensor): Target labels for training.
        save_dir (str): Directory where the trained model will be saved.
        model (tf.keras.Model): Compiled Keras model to train.
        batch_size (int): Batch size for training. Default is 128.
        epochs (int): Number of training epochs. Default is 100.
        name (str): Name of the saved model file (without extension).

    Returns:
        tf.keras.callbacks.History: Training history object.
    """
    import re

    # sanitize the model name
    name = re.sub(r'[^\w\-]', '_', name)

    # ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # define the path to save the model
    model_path = os.path.join(save_dir, f"{name}.keras")

    # define callbacks for training
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    # train the model
    history = model.fit(
        X,
        y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.05,
        verbose=1,
        callbacks=[reduce_lr, early_stopping]
    )

    # save the trained model
    model.save(model_path)
    print(f"Model saved to {model_path}")

    return history




def do_it_all_ba(
    pep_file_path, 
    allele_file_path,
    allele, 
    save_dir, 
    source_type="ba", 
    batch_size=128, 
    epochs=200, 
    model_name="binding_affinity_model"
):
    """
    Performs all steps from dataset preprocessing to training a model for binding affinity prediction.
    
    Args:
        pep_file_path (str): Path to the peptide dataset file.
        allele_file_path (str): Path to the allele sequence file.
        save_dir (str): Directory where the trained model will be saved.
        source_type (str): Source type to filter ('ba' by default).
        batch_size (int): Batch size for training. Default is 32.
        epochs (int): Number of epochs for training. Default is 100.
        model_name (str): Name of the saved model file (without extension).

    Returns:
        tf.keras.callbacks.History: Training history object.
    """
    # filter dataset
    filtered_data = filter_dataset(pep_file_path, source_type, allele_file_path)

    # filter specific allele and perform upsampling
    filtered_data = balance_binding_affinity(filtered_data, allele)

    # apply BLOSUM encoding
    filtered_data['input'] = [
        fast_peptide_to_blosum(row['seq']) 
        for _, row in filtered_data.iterrows()
    ]

    # prepare input and labels
    X = filtered_data["input"]
    X = np.stack(X)
    y = filtered_data["ba"]
    y = np.array(y)

    # define the model
    input_shape = (12,20)
    model = get_ba_model(input_shape=input_shape)

    # train the model
    history = train_model(X, y, save_dir, model, batch_size=batch_size, epochs=epochs, name=model_name)

    print("Training complete!")
    return history