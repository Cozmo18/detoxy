from toxy_bot.ml.data.convert import convert_to_tf_datasets
from toxy_bot.ml.data.download import download_dataset
from toxy_bot.ml.data.preprocess import process_data


def run_pipeline(val_size: float = 0.2, batch_size: int = 128) -> None:
    print("Running data pipeline...")
    download_dataset()
    process_data(val_size=val_size)
    convert_to_tf_datasets(batch_size=batch_size)
    print("Data pipeline completed.")


if __name__ == "__main__":
    run_pipeline()
