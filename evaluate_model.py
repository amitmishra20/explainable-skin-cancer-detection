"""
Comprehensive Model Evaluation Script for Explainable Skin Cancer Detection.
Computes test accuracy, per-class precision/recall/F1, confusion matrix,
and sample predictions on the held-out test set.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import CANONICAL_CLASSES, CLASS_METADATA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvaluateModel")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Skin Lesion Classifier on Test Set")
    parser.add_argument("--model_path", type=str, default="models/skin_lesion_model.keras")
    parser.add_argument("--classes_path", type=str, default="models/classes.json")
    parser.add_argument("--test_csv", type=str, default="results/test_split.csv")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--models_dir", type=str, default="models")
    return parser.parse_args()


def plot_and_save_confusion_matrix(cm: np.ndarray, classes: List[str], save_path: str):
    """Generates and saves a publication-quality confusion matrix heatmap."""
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        square=True
    )
    plt.title("Held-Out Test Set Confusion Matrix (HAM10000 7-Class)", fontsize=13, pad=12)
    plt.xlabel("Predicted Label", fontsize=11, labelpad=8)
    plt.ylabel("Ground Truth Label", fontsize=11, labelpad=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Confusion matrix saved to: {save_path}")


def evaluate(args):
    import tensorflow as tf

    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found at: {args.model_path}. Train the model first.")
        return False

    if not os.path.exists(args.classes_path):
        logger.error(f"Classes file not found at: {args.classes_path}.")
        return False

    with open(args.classes_path, "r") as f:
        classes = json.load(f)

    if not os.path.exists(args.test_csv):
        logger.error(f"Test split CSV not found at: {args.test_csv}. Run training or data preparation first.")
        return False

    test_df = pd.read_csv(args.test_csv)
    if len(test_df) == 0:
        logger.error("Test dataset dataframe is empty.")
        return False

    logger.info(f"Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)

    logger.info(f"Evaluating on {len(test_df)} held-out test images...")
    img_size = (224, 224)

    y_true = []
    y_pred = []
    y_probs = []
    sample_records = []

    for idx, row in test_df.iterrows():
        img_path = row["path"]
        true_label = int(row["label"])

        try:
            raw = tf.io.read_file(img_path)
            img = tf.image.decode_image(raw, channels=3, expand_animations=False)
            img = tf.image.resize(img, img_size)
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
            tensor = tf.expand_dims(img, axis=0)

            probs = model.predict(tensor, verbose=0)[0]
            pred_label = int(np.argmax(probs))

            y_true.append(true_label)
            y_pred.append(pred_label)
            y_probs.append(probs.tolist())

            if len(sample_records) < 15:
                sample_records.append({
                    "image_id": str(row.get("image_id", idx)),
                    "path": img_path,
                    "true_class": classes[true_label],
                    "predicted_class": classes[pred_label],
                    "is_correct": bool(true_label == pred_label),
                    "confidence": float(probs[pred_label])
                })
        except Exception as e:
            logger.warning(f"Skipping corrupt or unreadable image {img_path}: {e}")

    if len(y_true) == 0:
        logger.error("No test images could be evaluated.")
        return False

    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    eval_data = {
        "accuracy": acc,
        "sample_count": len(y_true),
        "classes": classes,
        "classification_report": report,
        "sample_predictions": sample_records
    }

    # Save to both results/ and models/ for app compatibility
    for dest_dir in [args.results_dir, args.models_dir]:
        rep_path = os.path.join(dest_dir, "evaluation_report.json")
        with open(rep_path, "w") as f:
            json.dump(eval_data, f, indent=2)
        plot_and_save_confusion_matrix(cm, classes, os.path.join(dest_dir, "confusion_matrix.png"))

    logger.info("=== EVALUATION COMPLETE ===")
    logger.info(f"Test Accuracy: {acc * 100:.2f}%")
    logger.info(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
    logger.info(f"Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")
    return True


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
