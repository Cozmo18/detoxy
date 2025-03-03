import keras


def build_export_model(
    bert_preprocessor: keras.Model,
    classifier: keras.Model,
) -> keras.Model:
    if not bert_preprocessor.built:
        raise ValueError("Cannot build export model: preprocessor is not built yet.")
    if not classifier.built:
        raise ValueError("Cannot build export model: classifier is not built yet.")

    preprocess_inputs = bert_preprocessor.inputs
    bert_encoder_inputs = bert_preprocessor(preprocess_inputs)
    outputs = classifier(bert_encoder_inputs)
    export_model = keras.Model(preprocess_inputs, outputs)

    return export_model
