from typing import List

import keras


def get_toxy_callbacks() -> List[keras.callbacks.Callback]:
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,  # Number of epochs to wait before stopping
        restore_best_weights=True,
        verbose=1,
    )

    best_model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/best_model.keras",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,  # Saves entire model, not just weights
        verbose=1,
    )

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir="logs",
        histogram_freq=1,
        write_graph=True,
    )

    return [early_stopping, best_model_checkpoint, tensorboard_callback]
