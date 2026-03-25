import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.binary_classifier import train_binary_model
from src.data_loader import load_data, prepare_binary

logger = logging.getLogger(__name__)

RESULTS_PATH = Path("results/false_negatives")


def extract_false_negatives(
    model, x_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    logger.info("Extracting false negative samples (attacks classified as normal)...")
    try:
        y_pred = model.predict(x_test)
        fn_mask = (y_test == 1) & (y_pred == 0)
        fn_samples = x_test[fn_mask].copy()
        fn_samples["true_label"] = y_test[fn_mask].values
        fn_samples["predicted_label"] = y_pred[fn_mask]
        logger.info(
            f"Found {len(fn_samples)} false negatives out of "
            f"{(y_test == 1).sum()} total attack samples."
        )
        return fn_samples
    except Exception as e:
        logger.error(f"Failed to extract false negatives: {e}")
        raise


def compare_fn_vs_normal(
    fn_samples: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    logger.info(
        "Comparing false negative feature profiles vs true normal traffic..."
    )
    try:
        numeric_features = ["dur", "spkts", "dpkts", "sload", "dload", "sbytes"]
        available = [f for f in numeric_features if f in x_test.columns]

        normal_samples = x_test[y_test == 0]

        comparison = pd.DataFrame(
            {
                "False Negatives (mean)": fn_samples[available].mean(),
                "Normal Traffic (mean)": normal_samples[available].mean(),
            }
        ).round(4)

        RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        path = RESULTS_PATH / "fn_vs_normal_comparison.csv"
        comparison.to_csv(path)
        logger.info(f"FN vs normal comparison saved to {path}")

        return comparison
    except Exception as e:
        logger.error(f"Failed to compare FN vs normal: {e}")
        raise


def save_fn_examples(fn_samples: pd.DataFrame, n: int = 10) -> None:
    logger.info(f"Saving top {n} false negative examples to CSV...")
    try:
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        path = RESULTS_PATH / "fn_examples.csv"
        fn_samples.head(n).to_csv(path, index=False)
        logger.info(f"False negative examples saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save FN examples: {e}")
        raise


def plot_fn_feature_distributions(
    fn_samples: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    save: bool = True,
) -> None:
    logger.info("Plotting feature distributions: false negatives vs normal traffic...")
    try:
        numeric_features = ["dur", "spkts", "sload", "sbytes"]
        available = [f for f in numeric_features if f in x_test.columns]
        normal_samples = x_test[y_test == 0]

        fig, axes = plt.subplots(1, len(available), figsize=(18, 4))

        for ax, feat in zip(axes, available, strict=False):
            fn_vals = fn_samples[feat].dropna()
            normal_vals = normal_samples[feat].dropna()
            cap = x_test[feat].quantile(0.95)

            ax.hist(
                normal_vals.clip(upper=cap),
                bins=40,
                alpha=0.6,
                color="steelblue",
                label="Normal",
            )
            ax.hist(
                fn_vals.clip(upper=cap),
                bins=40,
                alpha=0.6,
                color="tomato",
                label="False Negatives",
            )
            ax.set_title(feat)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.legend()

        plt.suptitle(
            "False Negatives vs Normal Traffic - Feature Distributions", y=1.02
        )
        plt.tight_layout()

        if save:
            RESULTS_PATH.mkdir(parents=True, exist_ok=True)
            path = RESULTS_PATH / "fn_feature_distributions.png"
            plt.savefig(path, bbox_inches="tight")
            logger.info(f"FN feature distribution plot saved to {path}")

        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot FN feature distributions: {e}")
        raise


def plot_fn_heatmap(fn_samples: pd.DataFrame, save: bool = True) -> None:
    logger.info("Plotting heatmap of false negative feature means...")
    try:
        numeric_features = ["dur", "spkts", "dpkts", "sload", "dload", "sbytes"]
        available = [f for f in numeric_features if f in fn_samples.columns]

        fn_mean = fn_samples[available].mean().to_frame(name="FN Mean").T
        fn_normalized = (fn_mean - fn_mean.min()) / (fn_mean.max() - fn_mean.min())

        plt.figure(figsize=(10, 2))
        sns.heatmap(
            fn_normalized,
            annot=True,
            fmt=".2f",
            cmap="Reds",
            linewidths=0.5,
        )
        plt.title("Normalized Mean Feature Values of False Negatives")
        plt.tight_layout()

        if save:
            RESULTS_PATH.mkdir(parents=True, exist_ok=True)
            path = RESULTS_PATH / "fn_heatmap.png"
            plt.savefig(path)
            logger.info(f"FN heatmap saved to {path}")

        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot FN heatmap: {e}")
        raise


def run_false_negative_analysis() -> pd.DataFrame:
    logger.info("Starting false negative analysis pipeline...")
    try:
        train, test = load_data()
        x_train, y_train, x_test, y_test = prepare_binary(train, test)
        model = train_binary_model(x_train, y_train)
        fn_samples = extract_false_negatives(model, x_test, y_test)
        comparison = compare_fn_vs_normal(fn_samples, x_test, y_test)
        save_fn_examples(fn_samples)
        plot_fn_feature_distributions(fn_samples, x_test, y_test)
        plot_fn_heatmap(fn_samples)
        logger.info("False negative analysis pipeline complete.")
        return comparison
    except Exception as e:
        logger.error(f"False negative analysis pipeline failed: {e}")
        raise
