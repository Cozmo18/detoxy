from toxy_bot.ml.data.convert import convert_to_tf_datasets
from toxy_bot.ml.data.download import download_dataset
from toxy_bot.ml.data.preprocess import process_data


def run_pipeline() -> None:
    print("Running data pipeline...")
    download_dataset()
    process_data()
    convert_to_tf_datasets()
    print("Data pipeline completed.")


if __name__ == "__main__":
    run_pipeline()
