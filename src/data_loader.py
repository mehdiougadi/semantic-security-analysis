import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TRAIN_PATH = "data/UNSW_NB15_training-set.csv"
TEST_PATH = "data/UNSW_NB15_testing-set.csv"

CATEGORICAL_FEATURES = ["proto", "service", "state"]
TARGET_BINARY = "label"
TARGET_MULTICLASS = "attack_cat"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading training and testing datasets from disk...")
    try:
        train = pd.read_csv(TRAIN_PATH)
        test = pd.read_csv(TEST_PATH)
        logger.info(f"Train shape: {train.shape} | Test shape: {test.shape}")
        return train, test
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(
        "Cleaning dataset: normalizing column names and handling missing values..."
    )
    try:
        df.columns = df.columns.str.strip().str.lower()
        missing_before = df.isnull().sum().sum()
        df = df.dropna()
        missing_after = df.isnull().sum().sum()
        logger.info(
            f"Dropped {missing_before - missing_after} missing values. "
            f"Remaining nulls: {missing_after}"
        )
        return df
    except Exception as e:
        logger.error(f"Failed to clean data: {e}")
        raise


def encode_categoricals(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(
        f"Encoding categorical features using LabelEncoder: {CATEGORICAL_FEATURES}"
    )
    try:
        encoders = {}
        for col in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            combined = pd.concat([train[col], test[col]], axis=0).astype(str)
            le.fit(combined)
            train[col] = le.transform(train[col].astype(str))
            test[col] = le.transform(test[col].astype(str))
            encoders[col] = le
        logger.info("Categorical encoding completed successfully.")
        return train, test
    except Exception as e:
        logger.error(f"Failed to encode categorical features: {e}")
        raise


def encode_attack_cat(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:
    logger.info(
        "Encoding target column 'attack_cat' for multi-class classification..."
    )
    try:
        le = LabelEncoder()
        combined = pd.concat(
            [train[TARGET_MULTICLASS], test[TARGET_MULTICLASS]], axis=0
        ).astype(str)
        le.fit(combined)
        train[TARGET_MULTICLASS] = le.transform(
            train[TARGET_MULTICLASS].astype(str)
        )
        test[TARGET_MULTICLASS] = le.transform(
            test[TARGET_MULTICLASS].astype(str)
        )
        logger.info(f"Attack categories encoded: {list(le.classes_)}")
        return train, test, le
    except Exception as e:
        logger.error(f"Failed to encode attack_cat: {e}")
        raise


def get_features_and_targets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    logger.info(f"Splitting features and target column: '{target}'...")
    try:
        drop_cols = [TARGET_BINARY, TARGET_MULTICLASS, "id"]
        drop_cols = [c for c in drop_cols if c in train.columns]

        x_train = train.drop(columns=drop_cols)
        y_train = train[target]
        x_test = test.drop(columns=drop_cols)
        y_test = test[target]

        logger.info(f"x_train: {x_train.shape} | x_test: {x_test.shape}")
        return x_train, y_train, x_test, y_test
    except Exception as e:
        logger.error(f"Failed to split features and targets: {e}")
        raise


def prepare_binary(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    logger.info("Preparing data pipeline for binary classification...")
    try:
        train = clean_data(train)
        test = clean_data(test)
        train, test = encode_categoricals(train, test)
        x_train, y_train, x_test, y_test = get_features_and_targets(
            train, test, TARGET_BINARY
        )
        logger.info("Binary classification data pipeline complete.")
        return x_train, y_train, x_test, y_test
    except Exception as e:
        logger.error(f"Binary preparation pipeline failed: {e}")
        raise


def prepare_multiclass(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, LabelEncoder]:
    logger.info("Preparing data pipeline for multi-class classification...")
    try:
        train = clean_data(train)
        test = clean_data(test)
        train, test = encode_categoricals(train, test)
        train, test, le = encode_attack_cat(train, test)
        x_train, y_train, x_test, y_test = get_features_and_targets(
            train, test, TARGET_MULTICLASS
        )
        logger.info("Multi-class classification data pipeline complete.")
        return x_train, y_train, x_test, y_test, le
    except Exception as e:
        logger.error(f"Multi-class preparation pipeline failed: {e}")
        raise
