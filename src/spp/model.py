"""LSTM model (Keras 3) and training.

``build_lstm(report=True)`` is the network from the original project:
LSTM(50, return_sequences) -> LSTM(50) -> Dense(1), Adam, mean squared error.
The default adds dropout against overfitting. The original trained for one
epoch with batch size 1; here it trains for several epochs with early
stopping on a validation split.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np

from .features import Prepared

# Hide TensorFlow's start-up notices (oneDNN, CPU instructions, no GPU on Windows).
# Keras is imported inside the functions below, so this runs before TensorFlow loads.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def build_lstm(window: int, units: int = 50, dropout: float = 0.2, report: bool = False):
    import keras
    from keras import layers

    blocks = [keras.Input((window, 1)), layers.LSTM(units, return_sequences=True)]
    if not report and dropout:
        blocks.append(layers.Dropout(dropout))
    blocks.append(layers.LSTM(units))
    if not report and dropout:
        blocks.append(layers.Dropout(dropout))
    blocks.append(layers.Dense(1))
    model = keras.Sequential(blocks, name="report_lstm" if report else "lstm")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mean_squared_error")
    return model


def train(prep: Prepared, epochs: int = 20, batch_size: int = 32, report: bool = False,
          seed: int = 42, verbose: int = 0, on_epoch: Callable[[int, dict], None] | None = None):  # fmt: skip
    import keras

    keras.utils.set_random_seed(seed)
    model = build_lstm(prep.window, report=report)
    callbacks = [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    if on_epoch is not None:
        callbacks.append(keras.callbacks.LambdaCallback(on_epoch_end=lambda e, logs: on_epoch(e, logs or {})))
    history = model.fit(
        prep.x_train, prep.y_train,
        validation_split=0.1, shuffle=False,  # keep time order in the validation split
        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose,
    )  # fmt: skip
    return model, history.history


def predictor(model) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a Keras model as a plain function: windows (n, w, 1) -> scaled predictions (n,)."""
    return lambda x: np.asarray(model.predict(x, verbose=0)).ravel()
