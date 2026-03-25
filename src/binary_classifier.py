import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from src.data_loader import load_data, prepare_binary

logger = logging.getLogger(__name__)

RESULTS_PATH = Path("results/binary")


def train_binary_model(x_train, y_train) -> XGBClassifier:
    logger.info("Training XGBoost binary classifier (Label: 0=Normal, 1=Attack)...")
    try:
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        logger.info("Binary model training complete.")
        return model
    except Exception as e:
        logger.error(f"Failed to train binary model: {e}")
        raise


def evaluate_binary_model(model: XGBClassifier, x_test, y_test) -> dict:
    logger.info("Evaluating binary model and computing metrics...")
    try:
        y_pred = model.predict(x_test)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "fpr": fpr,
            "fnr": fnr,
            "confusion_matrix": cm,
        }

        logger.info(f"Precision:  {precision:.4f}")
        logger.info(f"Recall:     {recall:.4f}")
        logger.info(f"F1-Score:   {f1:.4f}")
        logger.info(f"FPR:        {fpr:.4f}")
        logger.info(f"FNR:        {fnr:.4f}")
        logger.info(
            f"\nClassification Report:\n{classification_report(y_test, y_pred)}"
        )

        return metrics
    except Exception as e:
        logger.error(f"Failed to evaluate binary model: {e}")
        raise


def plot_confusion_matrix(cm, save: bool = True) -> None:
    logger.info("Plotting binary confusion matrix...")
    try:
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Attack"],
            yticklabels=["Normal", "Attack"],
        )
        plt.title("Binary Classification - Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()

        if save:
            RESULTS_PATH.mkdir(parents=True, exist_ok=True)
            path = RESULTS_PATH / "confusion_matrix.png"
            plt.savefig(path)
            logger.info(f"Confusion matrix saved to {path}")

        plt.show()
    except Exception as e:
        logger.error(f"Failed to plot confusion matrix: {e}")
        raise


def run_binary_classification() -> tuple[XGBClassifier, dict]:
    logger.info("Starting binary classification pipeline...")
    try:
        train, test = load_data()
        x_train, y_train, x_test, y_test = prepare_binary(train, test)
        model = train_binary_model(x_train, y_train)
        metrics = evaluate_binary_model(model, x_test, y_test)
        plot_confusion_matrix(metrics["confusion_matrix"])
        logger.info("Binary classification pipeline complete.")
        return model, metrics
    except Exception as e:
        logger.error(f"Binary classification pipeline failed: {e}")
        raise
