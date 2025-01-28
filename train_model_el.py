"""
balanced_5_fold_cv_tf.py

This module provides functionality for performing k-fold stratified cross-validation 
on a given dataset using TensorFlow. It trains a model on each fold, evaluates its 
performance, and saves the model with the highest accuracy. The module includes 
callbacks for learning rate reduction and early stopping to improve the training process.

Key Function:
    - balanced_5_fold_cv_tf(X, y, save_dir="models", k=5, input_shape=None, batch_size=32, epochs=10):
      This function performs k-fold stratified cross-validation on the dataset, 
      splits the data into training and test sets for each fold, builds and trains 
      a model, evaluates the model's performance, and saves the best model.

Dependencies:
    - numpy
    - pandas
    - sklearn
    - tensorflow

The function assumes that a model definition function (`get_model`) is available 
for constructing the model, and it uses the TensorFlow Keras API for building, training, 
and evaluating the neural network.

The models are saved in the specified directory, and the best model is selected 
based on the highest fold accuracy.

Usage Example:
    X, y = load_your_data()  # X: feature matrix, y: labels
    fold_scores, best_model_path = balanced_5_fold_cv_tf(X, y, save_dir="models", k=5, input_shape=(49, 20))
    print("Best model saved at:", best_model_path)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
from get_model_el import get_model 
from tensorflow.keras import backend as K

def balanced_5_fold_cv_tf(X, y, save_dir="models_el", k=5, input_shape=None, batch_size=64, epochs=200):
    """
    Perform k-fold stratified cross-validation on the given dataset using TensorFlow.
    The function trains a model on each fold, evaluates its performance, and saves the model 
    with the highest accuracy.

    Args:
    - X (np.ndarray or pd.DataFrame): Feature matrix for training (shape: [n_samples, n_features]).
    - y (np.ndarray or pd.Series): Labels corresponding to the feature matrix (shape: [n_samples]).
    - save_dir (str): Directory path where the models will be saved. Default is 'models_el'.
    - k (int): Number of folds in the cross-validation (default is 5).
    - input_shape (tuple): Shape of input features (e.g., (49, 20) for 49 time steps, 20 features).
    - batch_size (int): The batch size for training the model. Default is 32.
    - epochs (int): The number of epochs for training. Default is 200.

    Returns:
    - fold_scores (list): List of accuracy scores for each fold.
    - best_model_path (str): Path to the model with the highest accuracy.
    
    Raises:
    - ValueError: If `X` and `y` do not have compatible shapes.
    """
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # initialize Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_scores = []  # list to store accuracy scores for each fold
    best_score = -np.inf  # initialize to negative infinity to find the best score
    best_model_path = None  # will store the path of the best model

    # define callbacks for training
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    # loop through each fold
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        """
        For each fold:
        - Split the data into training and testing sets.
        - Build and train the model.
        - Evaluate and save the model with the best performance.
        """
        print(f"Starting Fold {fold + 1}/{k}...")
        
        # Split data into training and testing sets
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Build and compile the model
        model = get_model(input_shape=input_shape)

        # Train the model with callbacks for learning rate reduction and early stopping
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=[reduce_lr, early_stopping]
        )
        
        # Evaluate the model on the test set
        y_pred = (model.predict(X_test) > 0.5).astype("int32")  # Convert predictions to binary values (0 or 1)
        score = accuracy_score(y_test, y_pred)  # Calculate accuracy for the fold
        fold_scores.append(score)
        print(f"Fold {fold + 1} Accuracy: {score:.4f}")
        
        # Save the model for the current fold
        model_path = os.path.join(save_dir, f"model_fold_{fold + 1}.h5")
        model.save(model_path)

        # Update the best model if this fold's model is better
        if score > best_score:
            best_score = score
            best_model_path = model_path

        K.clear_session()  # Clear session to free memory

    # Output the final results
    print(f"\nMean Accuracy: {np.mean(fold_scores):.4f}")
    print(f"Best Model Saved at: {best_model_path}")
    return fold_scores, best_model_path
