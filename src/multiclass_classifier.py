import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from xgboost import XGBClassifier

from src.data_loader import load_data, prepare_multiclass

logger = logging.getLogger(__name__)

RESULTS_PATH = Path("results/multiclass")


def train_multiclass_model(x_train, y_train) -> XGBClassifier:
    logger.info("Training XGBoost multi-class classifier (attack_cat)...")
    try:
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softmax",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        logger.info("Multi-class model training complete.")
        return model
    except Exception as e:
        logger.error(f"Failed to train multi-class model: {e}")
        raise


def evaluate_multiclass_model(model: XGBClassifier, x_test, y_test, le) -> dict:
    logger.info("Evaluating multi-class model and computing per-class metrics...")
    try:
        y_pred = model.predict(x_test)
        class_names = list(le.classes_)

        cm = confusion_matrix(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average=None)
        f1 = f1_score(y_test, y_pred, average=None)

        metrics = {
            "confusion_matrix": cm,
            "per_class_recall": dict(zip(class_names, recall, strict=False)),
            "per_class_f1": dict(zip(class_names, f1, strict=False)),
            "class_names": class_names,
        }

        logger.info(
            f"\nClassification Report:\n"
            f"{classification_report(y_test, y_pred, target_names=class_names)}"
        )

        for cls in class_names:
            logger.info(
                f"[{cls}] Recall: {metrics['per_class_recall'][cls]:.4f} "
                f"| F1: {metrics['per_class_f1'][cls]:.4f}"
            )

        return metrics
    except Exception as e:
        logger.error(f"Failed to evaluate multi-class model: {e}")
        raise


def plot_confusion_matrix(cm, class_names: list, save: bool = True) -> None:
    logger.info("Plotting multi-class confusion matrix...")
    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Multi-Class Classification - Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save:
            RESULTS_PATH.mkdir(parents=True, exist_ok=True)
            path = RESULTS_PATH / "confusion_matrix.png"
            plt.savefig(path)
            logger.info(f"Multi-class confusion matrix saved to {path}")

        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot multi-class confusion matrix: {e}")
        raise


def plot_per_class_metrics(metrics: dict, save: bool = True) -> None:
    logger.info("Plotting per-class Recall and F1-score bar charts...")
    try:
        class_names = metrics["class_names"]
        recall_vals = list(metrics["per_class_recall"].values())
        f1_vals = list(metrics["per_class_f1"].values())

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].bar(class_names, recall_vals, color="steelblue")
        axes[0].set_title("Per-Class Recall")
        axes[0].set_xlabel("Attack Category")
        axes[0].set_ylabel("Recall")
        axes[0].set_xticklabels(class_names, rotation=45, ha="right")
        axes[0].set_ylim(0, 1.1)

        axes[1].bar(class_names, f1_vals, color="darkorange")
        axes[1].set_title("Per-Class F1-Score")
        axes[1].set_xlabel("Attack Category")
        axes[1].set_ylabel("F1-Score")
        axes[1].set_xticklabels(class_names, rotation=45, ha="right")
        axes[1].set_ylim(0, 1.1)

        plt.tight_layout()

        if save:
            RESULTS_PATH.mkdir(parents=True, exist_ok=True)
            path = RESULTS_PATH / "per_class_metrics.png"
            plt.savefig(path)
            logger.info(f"Per-class metrics chart saved to {path}")

        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot per-class metrics: {e}")
        raise


def run_multiclass_classification() -> tuple[XGBClassifier, dict]:
    logger.info("Starting multi-class classification pipeline...")
    try:
        train, test = load_data()
        x_train, y_train, x_test, y_test, le = prepare_multiclass(train, test)
        model = train_multiclass_model(x_train, y_train)
        metrics = evaluate_multiclass_model(model, x_test, y_test, le)
        plot_confusion_matrix(metrics["confusion_matrix"], metrics["class_names"])
        plot_per_class_metrics(metrics)
        logger.info("Multi-class classification pipeline complete.")
        return model, metrics
    except Exception as e:
        logger.error(f"Multi-class classification pipeline failed: {e}")
        raise
