# src/model_builder.py

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop

def build_model(input_shape, output_shape, hidden_layers=2, units=128,
                activation='relu', dropout_rate=0.1,
                optimizer='adam', learning_rate=1e-3):
    
    inputs = Input(shape=(input_shape,))
    x = inputs

    for _ in range(hidden_layers):
        x = Dense(units, activation=activation)(x)
        x = Dropout(dropout_rate)(x)

    outputs = Dense(output_shape)(x)

    model = Model(inputs=inputs, outputs=outputs)

    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model
