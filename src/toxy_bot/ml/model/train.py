from typing import Optional, Tuple

import keras
import tensorflow as tf
from official.nlp import optimization

from toxy_bot.ml.model.callbacks import get_toxy_callbacks
from toxy_bot.ml.model.metrics import get_toxy_metrics


def train_model(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: Optional[tf.data.Dataset],
    epochs: int,
) -> Tuple:  # TODO: Fix typehint
    loss = keras.losses.BinaryCrossentropy()
    metrics = get_toxy_metrics()
    callbacks = get_toxy_callbacks()

    steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(
        init_lr=init_lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type="adamw",
    )

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print("Training model...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )

    return model, history
