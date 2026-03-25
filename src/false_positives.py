import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.binary_classifier import train_binary_model
from src.data_loader import load_data, prepare_binary

logger = logging.getLogger(__name__)

RESULTS_PATH = Path("results/false_positives")


def extract_false_positives(
    model, x_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    logger.info(
        "Extracting false positive samples (normal traffic classified as attack)..."
    )
    try:
        y_pred = model.predict(x_test)
        fp_mask = (y_test == 0) & (y_pred == 1)
        fp_samples = x_test[fp_mask].copy()
        fp_samples["true_label"] = y_test[fp_mask].values
        fp_samples["predicted_label"] = y_pred[fp_mask]
        logger.info(
            f"Found {len(fp_samples)} false positives out of "
            f"{(y_test == 0).sum()} total normal samples."
        )
        return fp_samples
    except Exception as e:
        logger.error(f"Failed to extract false positives: {e}")
        raise


def compare_fp_vs_attacks(
    fp_samples: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    logger.info(
        "Comparing false positive feature profiles vs true attack traffic..."
    )
    try:
        numeric_features = ["dur", "spkts", "dpkts", "sload", "dload", "sbytes"]
        available = [f for f in numeric_features if f in x_test.columns]

        attack_samples = x_test[y_test == 1]
        normal_samples = x_test[y_test == 0]

        comparison = pd.DataFrame(
            {
                "False Positives (mean)": fp_samples[available].mean(),
                "True Attacks (mean)": attack_samples[available].mean(),
                "True Normal (mean)": normal_samples[available].mean(),
            }
        ).round(4)

        RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        path = RESULTS_PATH / "fp_vs_attacks_comparison.csv"
        comparison.to_csv(path)
        logger.info(f"FP vs attacks comparison saved to {path}")

        return comparison
    except Exception as e:
        logger.error(f"Failed to compare FP vs attacks: {e}")
        raise


def save_fp_examples(fp_samples: pd.DataFrame, n: int = 10) -> None:
    logger.info(f"Saving top {n} false positive examples to CSV...")
    try:
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        path = RESULTS_PATH / "fp_examples.csv"
        fp_samples.head(n).to_csv(path, index=False)
        logger.info(f"False positive examples saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save FP examples: {e}")
        raise


def plot_fp_feature_distributions(
    fp_samples: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    save: bool = True,
) -> None:
    logger.info(
        "Plotting feature distributions: false positives vs true attack traffic..."
    )
    try:
        numeric_features = ["dur", "spkts", "sload", "sbytes"]
        available = [f for f in numeric_features if f in x_test.columns]
        attack_samples = x_test[y_test == 1]

        fig, axes = plt.subplots(1, len(available), figsize=(18, 4))

        for ax, feat in zip(axes, available, strict=False):
            fp_vals = fp_samples[feat].dropna()
            attack_vals = attack_samples[feat].dropna()
            cap = x_test[feat].quantile(0.95)

            ax.hist(
                attack_vals.clip(upper=cap),
                bins=40,
                alpha=0.6,
                color="tomato",
                label="True Attacks",
            )
            ax.hist(
                fp_vals.clip(upper=cap),
                bins=40,
                alpha=0.6,
                color="orange",
                label="False Positives",
            )
            ax.set_title(feat)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.legend()

        plt.suptitle(
            "False Positives vs True Attacks - Feature Distributions", y=1.02
        )
        plt.tight_layout()

        if save:
            RESULTS_PATH.mkdir(parents=True, exist_ok=True)
            path = RESULTS_PATH / "fp_feature_distributions.png"
            plt.savefig(path, bbox_inches="tight")
            logger.info(f"FP feature distribution plot saved to {path}")

        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot FP feature distributions: {e}")
        raise


def plot_fp_load_vs_normal(
    fp_samples: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    save: bool = True,
) -> None:
    logger.info("Plotting load comparison: false positives vs normal traffic...")
    try:
        normal_samples = x_test[y_test == 0]
        load_features = ["sload", "dload"]
        available = [f for f in load_features if f in x_test.columns]

        fig, axes = plt.subplots(1, len(available), figsize=(12, 5))

        for ax, feat in zip(axes, available, strict=False):
            cap = x_test[feat].quantile(0.95)
            data = [
                normal_samples[feat].dropna().clip(upper=cap),
                fp_samples[feat].dropna().clip(upper=cap),
            ]
            ax.boxplot(data, labels=["Normal", "False Positives"], patch_artist=True)
            ax.set_title(f"{feat.upper()} - Normal vs False Positives")
            ax.set_xlabel("Traffic Type")
            ax.set_ylabel(feat)

        plt.suptitle("Load Features: Do High-Load Benign Flows Resemble Attacks?")
        plt.tight_layout()

        if save:
            RESULTS_PATH.mkdir(parents=True, exist_ok=True)
            path = RESULTS_PATH / "fp_load_vs_normal.png"
            plt.savefig(path)
            logger.info(f"FP load vs normal plot saved to {path}")

        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot FP load vs normal: {e}")
        raise


def plot_fp_heatmap(fp_samples: pd.DataFrame, save: bool = True) -> None:
    logger.info("Plotting heatmap of false positive feature means...")
    try:
        numeric_features = ["dur", "spkts", "dpkts", "sload", "dload", "sbytes"]
        available = [f for f in numeric_features if f in fp_samples.columns]

        fp_mean = fp_samples[available].mean().to_frame(name="FP Mean").T
        fp_normalized = (fp_mean - fp_mean.min()) / (fp_mean.max() - fp_mean.min())

        plt.figure(figsize=(10, 2))
        sns.heatmap(
            fp_normalized,
            annot=True,
            fmt=".2f",
            cmap="Oranges",
            linewidths=0.5,
        )
        plt.title("Normalized Mean Feature Values of False Positives")
        plt.tight_layout()

        if save:
            RESULTS_PATH.mkdir(parents=True, exist_ok=True)
            path = RESULTS_PATH / "fp_heatmap.png"
            plt.savefig(path)
            logger.info(f"FP heatmap saved to {path}")

        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot FP heatmap: {e}")
        raise


def run_false_positive_analysis() -> pd.DataFrame:
    logger.info("Starting false positive analysis pipeline...")
    try:
        train, test = load_data()
        x_train, y_train, x_test, y_test = prepare_binary(train, test)
        model = train_binary_model(x_train, y_train)
        fp_samples = extract_false_positives(model, x_test, y_test)
        comparison = compare_fp_vs_attacks(fp_samples, x_test, y_test)
        save_fp_examples(fp_samples)
        plot_fp_feature_distributions(fp_samples, x_test, y_test)
        plot_fp_load_vs_normal(fp_samples, x_test, y_test)
        plot_fp_heatmap(fp_samples)
        logger.info("False positive analysis pipeline complete.")
        return comparison
    except Exception as e:
        logger.error(f"False positive analysis pipeline failed: {e}")
        raise
