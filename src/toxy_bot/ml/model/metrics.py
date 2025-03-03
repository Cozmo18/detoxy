from typing import List

import keras


def get_toxy_metrics() -> List[keras.metrics.Metric]:
    """Returns a list of evaluation metrics for multi-label classification."""
    return [
        keras.metrics.AUC(name="auc"),
        keras.metrics.AUC(name="prc", curve="PR"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ]
