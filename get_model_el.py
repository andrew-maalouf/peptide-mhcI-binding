"""
get_model_el.py

This script defines a deep learning model for binary classification (eluted ligand prediction).
The model consists of multiple LSTM layers, with masking, dropout, and regularization applied.

Functions:
    get_model() -> tensorflow.keras.Model: Returns the compiled model.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Masking, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

def get_model(input_shape=(46, 20)):
    """
    Defines and compiles a Bidirectional LSTM model for eluted ligand prediction.
    
    Args:
        input_shape (tuple): The shape of the input data, default is (49, 20).
        
    Returns:
        model (tensorflow.keras.Model): The compiled model ready for training.
    """
    
    # input layer (expecting shape (batch_size, 49, 20))
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

    # output layer for eluted ligand prediction (sigmoid output)
    el_output = Dense(1, activation='sigmoid', name='el_output')(x)

    # define the model
    model = Model(inputs=input_layer, outputs=[el_output])

    # compile the model
    model.compile(optimizer=Adam(),
                  loss={'el_output': 'binary_crossentropy'}, 
                  metrics={'el_output': ['accuracy']})

    return model

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=15, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
