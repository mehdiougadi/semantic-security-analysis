import logging
from pathlib import Path

from src.data_loader import load_data


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


if __name__ == "__main__":
    main()
