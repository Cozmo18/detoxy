import keras


def load_trained_model(model_path: str) -> keras.Model:
    """Loads a trained model from the given path.

    Args:
        model_path (str): Path to the saved Keras model.

    Returns:
        tf.keras.Model: Loaded model.
    """
    return keras.models.load_model(model_path)  # type: ignore


def print_bert_results(test_text, result):
    pass


# def predict_text(model: tf.keras.Model, text: list[str]) -> dict:
#     """Predicts toxicity labels for a given text input.

#     Args:
#         model (tf.keras.Model): Trained model.
#         text (list[str]): List of input texts.

#     Returns:
#         dict: Dictionary mapping input texts to their predicted toxicity labels.
#     """
#     bert_model_name = CONFIG["model"]["bert_checkpoint"]
#     preprocessor = make_bert_preprocess_model(bert_model_name)

#     inputs = preprocessor(tf.constant(text))
#     predictions = model.predict(inputs)
#     label_names = CONFIG["dataset"]["labels"]

#     results = {t: dict(zip(label_names, preds.tolist())) for t, preds in zip(text, predictions)}
#     return results
