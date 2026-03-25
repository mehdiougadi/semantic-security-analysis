import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import load_data

logger = logging.getLogger(__name__)

RESULTS_PATH = Path("results/feature_semantics")

FEATURE_GROUPS = {
    "Network Context": {
        "features": ["proto", "service", "state"],
        "security_meaning": (
            "Identifies the communication protocol, application service, "
            "and connection state. Helps distinguish legitimate traffic from "
            "suspicious protocol abuse."
        ),
        "role_in_detection": (
            "Unusual protocol/service combinations (e.g. TCP on non-standard ports) "
            "are strong indicators of scanning or exploitation attempts."
        ),
    },
    "Traffic Behavior": {
        "features": ["Sload", "Dload", "Spkts", "Dpkts", "Sbytes", "Dbytes"],
        "security_meaning": (
            "Measures the volume and rate of traffic in both directions. "
            "Captures how much data is being transferred and at what speed."
        ),
        "role_in_detection": (
            "Abnormally high source load or packet count with low destination "
            "response is a classic DoS or flood attack signature."
        ),
    },
    "Temporal Context": {
        "features": ["dur", "Stime", "Ltime"],
        "security_meaning": (
            "Captures the duration and timestamps of connections. "
            "Reveals how long a session lasts and when it occurred."
        ),
        "role_in_detection": (
            "Very short durations with high packet rates suggest DoS. "
            "Very long durations may indicate data exfiltration "
            "or persistent backdoors."
        ),
    },
    "Connection Patterns": {
        "features": [
            "ct_srv_src",
            "ct_dst_ltm",
            "ct_src_dport_ltm",
            "ct_dst_sport_ltm",
            "ct_dst_src_ltm",
            "ct_ftp_cmd",
            "ct_flw_http_mthd",
            "ct_src_ltm",
            "ct_srv_dst",
        ],
        "security_meaning": (
            "Counts of recent connections grouped by source, destination, "
            "service, and port combinations within a time window."
        ),
        "role_in_detection": (
            "High counts of connections from one source to many destinations "
            "or ports reveal reconnaissance and scanning behavior."
        ),
    },
}


def build_feature_table() -> pd.DataFrame:
    logger.info("Building structured feature semantics table...")
    try:
        rows = []
        for group, info in FEATURE_GROUPS.items():
            rows.append(
                {
                    "Feature Group": group,
                    "Example Features": ", ".join(info["features"]),
                    "Security Meaning": info["security_meaning"],
                    "Role in Attack Detection": info["role_in_detection"],
                }
            )
        df = pd.DataFrame(rows)
        logger.info(f"Feature table built with {len(df)} groups.")
        return df
    except Exception as e:
        logger.error(f"Failed to build feature table: {e}")
        raise


def save_feature_table(df: pd.DataFrame, save: bool = True) -> None:
    logger.info("Saving feature semantics table as CSV...")
    try:
        if save:
            RESULTS_PATH.mkdir(parents=True, exist_ok=True)
            path = RESULTS_PATH / "feature_groups.csv"
            df.to_csv(path, index=False)
            logger.info(f"Feature table saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save feature table: {e}")
        raise


def plot_feature_group_counts(save: bool = True) -> None:
    logger.info("Plotting feature count per group...")
    try:
        group_names = list(FEATURE_GROUPS.keys())
        feature_counts = [len(v["features"]) for v in FEATURE_GROUPS.values()]

        plt.figure(figsize=(8, 5))
        plt.bar(group_names, feature_counts, color="steelblue")
        plt.title("Number of Features per Semantic Group")
        plt.xlabel("Feature Group")
        plt.ylabel("Feature Count")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()

        if save:
            RESULTS_PATH.mkdir(parents=True, exist_ok=True)
            path = RESULTS_PATH / "feature_group_counts.png"
            plt.savefig(path)
            logger.info(f"Feature group count plot saved to {path}")

        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot feature group counts: {e}")
        raise


def analyze_feature_distributions(save: bool = True) -> None:
    logger.info(
        "Analyzing feature distributions across normal and attack traffic..."
    )
    try:
        train, _ = load_data()
        train.columns = train.columns.str.strip().str.lower()

        sample_features = ["dur", "spkts", "sload", "dload", "sbytes"]
        available = [f for f in sample_features if f in train.columns]

        fig, axes = plt.subplots(1, len(available), figsize=(18, 4))

        for ax, feat in zip(axes, available):
            for label, color, name in zip(
                [0, 1], ["steelblue", "tomato"], ["Normal", "Attack"]
            ):
                subset = train[train["label"] == label][feat].dropna()
                subset = subset[subset < subset.quantile(0.99)]
                ax.hist(subset, bins=40, alpha=0.6, color=color, label=name)
            ax.set_title(feat)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.legend()

        plt.suptitle("Feature Distributions: Normal vs Attack Traffic", y=1.02)
        plt.tight_layout()

        if save:
            RESULTS_PATH.mkdir(parents=True, exist_ok=True)
            path = RESULTS_PATH / "feature_distributions.png"
            plt.savefig(path, bbox_inches="tight")
            logger.info(f"Feature distribution plot saved to {path}")

        plt.close()
    except Exception as e:
        logger.error(f"Failed to analyze feature distributions: {e}")
        raise


def run_feature_semantics() -> pd.DataFrame:
    logger.info("Starting feature semantics analysis pipeline...")
    try:
        df = build_feature_table()
        save_feature_table(df)
        plot_feature_group_counts()
        analyze_feature_distributions()
        logger.info("Feature semantics pipeline complete.")
        return df
    except Exception as e:
        logger.error(f"Feature semantics pipeline failed: {e}")
        raise
