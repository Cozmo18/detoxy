from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # noqa: F401

from toxy_bot.ml.model.utils import get_preprocess_url

tf.get_logger().setLevel("ERROR")


def make_bert_preprocess_model(
    bert_model: str,
    sentence_features: List[str] = ["comment_text"],
    seq_length: int = 128,
) -> tf.keras.Model:  # type: ignore
    """Creates a BERT preprocessing model for tokenizing and encoding input text.

    Args:
        bert_model (str): The name of the BERT model to use for preprocessing.
        sentence_features (List[str]): List of string feature names.
        seq_length (int): Maximum sequence length for BERT inputs.

    Returns:
        tf.keras.Model: A TensorFlow model for preprocessing text input.
    """
    if not isinstance(sentence_features, list) or not all(
        isinstance(ft, str) for ft in sentence_features
    ):
        raise ValueError("sentence_features must be a list of strings.")

    # Define input layers for sentence features
    input_segments = [
        tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)  # type: ignore
        for ft in sentence_features
    ]

    # Load preprocessing model
    preprocess_url = get_preprocess_url(bert_model)
    bert_preprocess = hub.load(preprocess_url)
    tokenizer = hub.KerasLayer(
        bert_preprocess.tokenize,  # type: ignore
        name="tokenizer",  # type: ignore
    )  # Tokenization layer

    # Tokenize input sentences
    segments = [tokenizer(s) for s in input_segments]

    # Pack inputs for BERT model
    packer = hub.KerasLayer(
        bert_preprocess.bert_pack_inputs,  # type: ignore
        arguments=dict(seq_length=seq_length),
        name="packer",
    )
    model_inputs = packer(segments)

    return tf.keras.Model(input_segments, model_inputs)  # type: ignore


# Example usage
if __name__ == "__main__":
    BERT_MODEL_NAME = "small_bert/bert_en_uncased_L-4_H-512_A-8"
    preprocess_model = make_bert_preprocess_model(BERT_MODEL_NAME)

    test_preprocess_model = make_bert_preprocess_model(
        BERT_MODEL_NAME, ["my_input1", "my_input2"]
    )

    test_text = [
        np.array(["some random test sentence"]),
        np.array(["another sentence"]),
    ]
    text_preprocessed = test_preprocess_model(test_text)

    print("Keys           : ", list(text_preprocessed.keys()))
    print("Shape Word Ids : ", text_preprocessed["input_word_ids"].shape)
    print("Shape Mask     : ", text_preprocessed["input_mask"].shape)
    print("Shape Type Ids : ", text_preprocessed["input_type_ids"].shape)

    preprocess_model.summary()
