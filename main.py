import logging
from pathlib import Path

from src.binary_classifier import run_binary_classification
from src.data_loader import load_data
from src.feature_semantics import run_feature_semantics
from src.multiclass_classifier import run_multiclass_classification


def setup_logging():
    log_dir = Path("./results")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "script.log"),
            logging.StreamHandler(),
        ],
    )


def main():
    setup_logging()
    train, test = load_data()

    run_binary_classification()
    run_multiclass_classification()
    run_feature_semantics()


if __name__ == "__main__":
    main()
