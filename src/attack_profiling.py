import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_loader import load_data

logger = logging.getLogger(__name__)

RESULTS_PATH = Path("results/attack_profiling")

TARGET_CATEGORIES = ["DoS", "Exploits", "Reconnaissance"]

PROFILE_FEATURES = {
    "Load Features": ["sload", "dload"],
    "Packet Features": ["spkts", "dpkts"],
    "Duration": ["dur"],
    "Service Usage": ["service"],
}


def load_attack_data() -> pd.DataFrame:
    logger.info("Loading and preparing data for attack behavior profiling...")
    try:
        train, test = load_data()
        df = pd.concat([train, test], axis=0)
        df.columns = df.columns.str.strip().str.lower()
        df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
        logger.info(
            f"Combined dataset shape: {df.shape} | "
            f"Attack categories: {df['attack_cat'].unique().tolist()}"
        )
        return df
    except Exception as e:
        logger.error(f"Failed to load attack data: {e}")
        raise


def compute_attack_statistics(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(
        f"Computing attack behavior statistics for: {TARGET_CATEGORIES}..."
    )
    try:
        numeric_features = ["sload", "dload", "spkts", "dpkts", "dur"]
        available = [f for f in numeric_features if f in df.columns]

        filtered = df[df["attack_cat"].isin(TARGET_CATEGORIES)]
        stats = (
            filtered.groupby("attack_cat")[available]
            .agg(["mean", "median", "std"])
            .round(4)
        )

        RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        path = RESULTS_PATH / "attack_statistics.csv"
        stats.to_csv(path)
        logger.info(f"Attack statistics saved to {path}")

        return stats
    except Exception as e:
        logger.error(f"Failed to compute attack statistics: {e}")
        raise


def plot_load_comparison(df: pd.DataFrame, save: bool = True) -> None:
    logger.info("Plotting load feature comparison across attack categories...")
    try:
        filtered = df[df["attack_cat"].isin(TARGET_CATEGORIES)]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, feat in zip(axes, ["sload", "dload"], strict=False):
            data = [
                filtered[filtered["attack_cat"] == cat][feat]
                .dropna()
                .clip(upper=filtered[feat].quantile(0.95))
                for cat in TARGET_CATEGORIES
            ]
            ax.boxplot(data, labels=TARGET_CATEGORIES, patch_artist=True)
            ax.set_title(f"{feat.upper()} by Attack Category")
            ax.set_xlabel("Attack Category")
            ax.set_ylabel(feat)

        plt.suptitle("Load Features: DoS vs Exploits vs Reconnaissance")
        plt.tight_layout()

        if save:
            RESULTS_PATH.mkdir(parents=True, exist_ok=True)
            path = RESULTS_PATH / "load_comparison.png"
            plt.savefig(path)
            logger.info(f"Load comparison plot saved to {path}")

        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot load comparison: {e}")
        raise


def plot_packet_comparison(df: pd.DataFrame, save: bool = True) -> None:
    logger.info("Plotting packet feature comparison across attack categories...")
    try:
        filtered = df[df["attack_cat"].isin(TARGET_CATEGORIES)]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, feat in zip(axes, ["spkts", "dpkts"], strict=False):
            data = [
                filtered[filtered["attack_cat"] == cat][feat]
                .dropna()
                .clip(upper=filtered[feat].quantile(0.95))
                for cat in TARGET_CATEGORIES
            ]
            ax.boxplot(data, labels=TARGET_CATEGORIES, patch_artist=True)
            ax.set_title(f"{feat.upper()} by Attack Category")
            ax.set_xlabel("Attack Category")
            ax.set_ylabel(feat)

        plt.suptitle("Packet Features: DoS vs Exploits vs Reconnaissance")
        plt.tight_layout()

        if save:
            RESULTS_PATH.mkdir(parents=True, exist_ok=True)
            path = RESULTS_PATH / "packet_comparison.png"
            plt.savefig(path)
            logger.info(f"Packet comparison plot saved to {path}")

        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot packet comparison: {e}")
        raise


def plot_duration_comparison(df: pd.DataFrame, save: bool = True) -> None:
    logger.info("Plotting duration comparison across attack categories...")
    try:
        filtered = df[df["attack_cat"].isin(TARGET_CATEGORIES)]

        plt.figure(figsize=(8, 5))
        data = [
            filtered[filtered["attack_cat"] == cat]["dur"]
            .dropna()
            .clip(upper=filtered["dur"].quantile(0.95))
            for cat in TARGET_CATEGORIES
        ]
        plt.boxplot(data, labels=TARGET_CATEGORIES, patch_artist=True)
        plt.title("Connection Duration: DoS vs Exploits vs Reconnaissance")
        plt.xlabel("Attack Category")
        plt.ylabel("Duration (seconds)")
        plt.tight_layout()

        if save:
            RESULTS_PATH.mkdir(parents=True, exist_ok=True)
            path = RESULTS_PATH / "duration_comparison.png"
            plt.savefig(path)
            logger.info(f"Duration comparison plot saved to {path}")

        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot duration comparison: {e}")
        raise


def plot_service_usage(df: pd.DataFrame, save: bool = True) -> None:
    logger.info("Plotting top service usage per attack category...")
    try:
        filtered = df[df["attack_cat"].isin(TARGET_CATEGORIES)]
        fig, axes = plt.subplots(1, len(TARGET_CATEGORIES), figsize=(18, 5))

        for ax, cat in zip(axes, TARGET_CATEGORIES, strict=False):
            top_services = (
                filtered[filtered["attack_cat"] == cat]["service"]
                .value_counts()
                .head(6)
            )
            ax.bar(top_services.index, top_services.values, color="steelblue")
            ax.set_title(f"Top Services - {cat}")
            ax.set_xlabel("Service")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=30)

        plt.suptitle("Service Usage by Attack Category")
        plt.tight_layout()

        if save:
            RESULTS_PATH.mkdir(parents=True, exist_ok=True)
            path = RESULTS_PATH / "service_usage.png"
            plt.savefig(path)
            logger.info(f"Service usage plot saved to {path}")

        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot service usage: {e}")
        raise


def plot_heatmap_summary(df: pd.DataFrame, save: bool = True) -> None:
    logger.info("Plotting heatmap summary of mean feature values per attack...")
    try:
        numeric_features = ["sload", "dload", "spkts", "dpkts", "dur"]
        available = [f for f in numeric_features if f in df.columns]

        filtered = df[df["attack_cat"].isin(TARGET_CATEGORIES)]
        summary = filtered.groupby("attack_cat")[available].mean()
        summary_normalized = (summary - summary.min()) / (
            summary.max() - summary.min()
        )

        plt.figure(figsize=(10, 4))
        sns.heatmap(
            summary_normalized,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            linewidths=0.5,
        )
        plt.title("Normalized Mean Feature Values per Attack Category")
        plt.ylabel("Attack Category")
        plt.tight_layout()

        if save:
            RESULTS_PATH.mkdir(parents=True, exist_ok=True)
            path = RESULTS_PATH / "heatmap_summary.png"
            plt.savefig(path)
            logger.info(f"Heatmap summary saved to {path}")

        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot heatmap summary: {e}")
        raise


def run_attack_profiling() -> pd.DataFrame:
    logger.info("Starting attack behavior profiling pipeline...")
    try:
        df = load_attack_data()
        stats = compute_attack_statistics(df)
        plot_load_comparison(df)
        plot_packet_comparison(df)
        plot_duration_comparison(df)
        plot_service_usage(df)
        plot_heatmap_summary(df)
        logger.info("Attack profiling pipeline complete.")
        return stats
    except Exception as e:
        logger.error(f"Attack profiling pipeline failed: {e}")
        raise
