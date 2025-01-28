"""
get_model_ba.py

This module provides a function to create and compile a Keras model for predicting binding affinity
for one specific allele. The model's input shape is (12, 20), and it uses a custom logarithmic loss 
function as its loss function.

Functions:
    get_ba_model(input_shape: tuple) -> tf.keras.Model:
        Creates and compiles the Keras model for binding affinity prediction.
"""

import tensorflow as tf
from tensorflow.keras import layers, backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Masking, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam


# custom loss function
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


def get_ba_model(input_shape: tuple = (12, 20)) -> tf.keras.Model:
    """
    Creates and compiles a Keras model for predicting binding affinity for one specific allele.

    The model consists of:
        - Bidirectional LSTM layers for sequence processing.
        - Dense layer for feature extraction and prediction.
        - A final Dense layer with a linear activation function for regression output.

    Args:
        input_shape (tuple): The shape of the input tensor. Default is (12, 20).

    Returns:
        tf.keras.Model: A compiled Keras model.
    """
    # input layer (expecting shape (batch_size, 12, 20))
    input_layer = Input(shape=input_shape)

    # add Masking layer
    masked_input = Masking(mask_value=0.0)(input_layer)

    # first Bidirectional LSTM layer
    x = Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(masked_input)

    # second Bidirectional LSTM layer
    x = Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=regularizers.l2(0.01)))(x)

    x = Dropout(0.5)(x)

    # dense layer
    x = Dense(16, activation='relu')(x)

    # output layer with linear activation
    outputs = layers.Dense(1, activation='linear')(x)

    # define the model
    model = tf.keras.Model(inputs=input_layer, outputs=outputs)

    # compile the model
    model.compile(
        optimizer=Adam(),
        loss=custom_logarithmic_loss,
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )

    return model
