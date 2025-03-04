from typing import Dict

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # noqa: F401

from toxy_bot.ml.model.preprocess import make_bert_preprocess_model
from toxy_bot.ml.model.utils import get_encoder_url
from toxy_bot.utils import config


def make_bert_classifier_model(bert_model: str) -> tf.keras.Model:  # type: ignore
    num_labels = len(config.DATASET_LABELS)
    return Classifier(bert_model, num_labels)


class Classifier(tf.keras.Model):  # type: ignore
    """BERT-based classifier for multi-label text classification."""

    def __init__(self, bert_model: str, num_labels: int) -> None:
        """Initializes the classifier with a BERT encoder and classification head.

        Args:
            bert_model (str): Name of the BERT model.
            num_labels (int): Number of output labels.
        """
        super().__init__(name="prediction")
        self.tfhub_handle_encoder = get_encoder_url(bert_model)
        self.encoder = hub.KerasLayer(
            self.tfhub_handle_encoder, trainable=True, name="BERT_encoder"
        )
        self.dropout = tf.keras.layers.Dropout(0.1)  # type: ignore
        self.classifier = tf.keras.layers.Dense(num_labels, activation="sigmoid")  # type: ignore

    def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Forward pass of the model.

        Args:
            inputs (Dict[str, tf.Tensor]): Preprocessed input tensors.

        Returns:
            tf.Tensor: Model predictions.
        """
        encoder_outputs = self.encoder(inputs)
        pooled_output = encoder_outputs["pooled_output"]
        x = self.dropout(pooled_output)
        return self.classifier(x)


if __name__ == "__main__":
    BERT_MODEL_NAME = "small_bert/bert_en_uncased_L-4_H-512_A-8"
    preprocessor = make_bert_preprocess_model(BERT_MODEL_NAME)
    classifier = make_bert_classifier_model(BERT_MODEL_NAME)

    test_text = [tf.constant(["some random test sentence", "some different text"])]
    text_preprocessed = preprocessor(test_text)
    test_result = classifier(text_preprocessed)
    print(test_result)

    print(classifier.summary())
