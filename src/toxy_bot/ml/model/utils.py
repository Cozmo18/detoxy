# def connect_to_tpu_worker():
#     if os.environ["COLAB_TPU_ADDR"]:
#         cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="")
#         tf.config.experimental_connect_to_cluster(cluster_resolver)
#         tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
#         strategy = tf.distribute.TPUStrategy(cluster_resolver)
#         print("Using TPU")
#     elif tf.config.list_physical_devices("GPU"):
#         strategy = tf.distribute.MirroredStrategy()
#         print("Using GPU")
#     else:
#         raise ValueError("Running on CPU is not recommended.")


def get_encoder_url(bert_model: str) -> str:
    """Retrieves the encoder URL for the given BERT model.

    Args:
        bert_model (str): The name of the BERT model.

    Returns:
        str: TensorFlow Hub URL for encoding.

    Raises:
        ValueError: If the provided model name is invalid.
    """
    map_name_to_handle = {
        "bert_en_uncased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3",
        "bert_en_uncased_L-24_H-1024_A-16": "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/3",
        "bert_en_wwm_uncased_L-24_H-1024_A-16": "https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/3",
        "bert_en_cased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3",
        "bert_en_cased_L-24_H-1024_A-16": "https://tfhub.dev/tensorflow/bert_en_cased_L-24_H-1024_A-16/3",
        "bert_en_wwm_cased_L-24_H-1024_A-16": "https://tfhub.dev/tensorflow/bert_en_wwm_cased_L-24_H-1024_A-16/3",
        "bert_multi_cased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3",
        "small_bert/bert_en_uncased_L-2_H-128_A-2": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1",
        "small_bert/bert_en_uncased_L-2_H-256_A-4": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1",
        "small_bert/bert_en_uncased_L-2_H-512_A-8": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1",
        "small_bert/bert_en_uncased_L-2_H-768_A-12": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1",
        "small_bert/bert_en_uncased_L-4_H-128_A-2": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1",
        "small_bert/bert_en_uncased_L-4_H-256_A-4": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1",
        "small_bert/bert_en_uncased_L-4_H-512_A-8": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1",
        "small_bert/bert_en_uncased_L-4_H-768_A-12": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1",
        "small_bert/bert_en_uncased_L-6_H-128_A-2": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1",
        "small_bert/bert_en_uncased_L-6_H-256_A-4": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1",
        "small_bert/bert_en_uncased_L-6_H-512_A-8": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1",
        "small_bert/bert_en_uncased_L-6_H-768_A-12": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1",
        "small_bert/bert_en_uncased_L-8_H-128_A-2": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1",
        "small_bert/bert_en_uncased_L-8_H-256_A-4": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1",
        "small_bert/bert_en_uncased_L-8_H-512_A-8": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1",
        "small_bert/bert_en_uncased_L-8_H-768_A-12": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1",
        "small_bert/bert_en_uncased_L-10_H-128_A-2": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1",
        "small_bert/bert_en_uncased_L-10_H-256_A-4": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1",
        "small_bert/bert_en_uncased_L-10_H-512_A-8": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1",
        "small_bert/bert_en_uncased_L-10_H-768_A-12": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1",
        "small_bert/bert_en_uncased_L-12_H-128_A-2": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
        "small_bert/bert_en_uncased_L-12_H-256_A-4": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1",
        "small_bert/bert_en_uncased_L-12_H-512_A-8": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1",
        "small_bert/bert_en_uncased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1",
        "albert_en_base": "https://tfhub.dev/tensorflow/albert_en_base/2",
        "albert_en_large": "https://tfhub.dev/tensorflow/albert_en_large/2",
        "albert_en_xlarge": "https://tfhub.dev/tensorflow/albert_en_xlarge/2",
        "albert_en_xxlarge": "https://tfhub.dev/tensorflow/albert_en_xxlarge/2",
        "electra_small": "https://tfhub.dev/google/electra_small/2",
        "electra_base": "https://tfhub.dev/google/electra_base/2",
        "experts_pubmed": "https://tfhub.dev/google/experts/bert/pubmed/2",
        "experts_wiki_books": "https://tfhub.dev/google/experts/bert/wiki_books/2",
        "talking-heads_base": "https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1",
        "talking-heads_large": "https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_large/1",
    }
    if bert_model not in map_name_to_handle:
        raise ValueError(f"Invalid BERT model: {bert_model}")
    return map_name_to_handle[bert_model]


def get_preprocess_url(bert_model: str) -> str:
    """Retrieves the preprocessing URL for the given BERT model.

    Args:
        bert_model (str): The name of the BERT model.

    Returns:
        str: TensorFlow Hub URL for preprocessing.

    Raises:
        ValueError: If the provided model name is invalid.
    """

    map_model_to_preprocess = {
        "bert_en_uncased_L-24_H-1024_A-16": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "bert_en_uncased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "bert_en_wwm_cased_L-24_H-1024_A-16": "https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3",
        "bert_en_cased_L-24_H-1024_A-16": "https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3",
        "bert_en_cased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3",
        "bert_en_wwm_uncased_L-24_H-1024_A-16": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-2_H-128_A-2": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-2_H-256_A-4": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-2_H-512_A-8": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-2_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-4_H-128_A-2": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-4_H-256_A-4": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-4_H-512_A-8": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-4_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-6_H-128_A-2": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-6_H-256_A-4": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-6_H-512_A-8": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-6_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-8_H-128_A-2": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-8_H-256_A-4": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-8_H-512_A-8": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-8_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-10_H-128_A-2": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-10_H-256_A-4": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-10_H-512_A-8": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-10_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-12_H-128_A-2": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-12_H-256_A-4": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-12_H-512_A-8": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "bert_multi_cased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3",
        "albert_en_base": "https://tfhub.dev/tensorflow/albert_en_preprocess/3",
        "albert_en_large": "https://tfhub.dev/tensorflow/albert_en_preprocess/3",
        "albert_en_xlarge": "https://tfhub.dev/tensorflow/albert_en_preprocess/3",
        "albert_en_xxlarge": "https://tfhub.dev/tensorflow/albert_en_preprocess/3",
        "electra_small": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "electra_base": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "experts_pubmed": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "experts_wiki_books": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "talking-heads_base": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "talking-heads_large": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    }

    if bert_model not in map_model_to_preprocess:
        raise ValueError(f"Invalid BERT model: {bert_model}")
    return map_model_to_preprocess[bert_model]
