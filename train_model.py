"""
End-to-End Training Pipeline for HAM10000 Skin Lesion Classification.
Transfer Learning with MobileNetV2 + 7-Class Head + Group Leakage Prevention.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure local imports work regardless of execution directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import (
    CANONICAL_CLASSES,
    CLASS_TO_IDX,
    CLASS_METADATA,
    discover_dataset,
    create_demo_samples
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainModel")

DEFAULT_SEED = 42
DEFAULT_IMG_SIZE = (224, 224)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on HAM10000 dataset")
    parser.add_argument("--data_dir", type=str, default="data/HAM10000", help="Path to HAM10000 data directory")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save trained model artifacts")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save curves and metrics")
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "5")), help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--smoke_test", action="store_true", help="Run a rapid smoke test with minimal steps")
    return parser.parse_args()


def build_mobilenetv2_model(num_classes: int = 7, img_size: Tuple[int, int] = (224, 224)) -> "tf.keras.Model":
    """
    Builds MobileNetV2 transfer learning architecture with a custom classification head.
    """
    import tensorflow as tf
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze pretrained backbone for initial transfer learning

    inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3), name="input_image")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="batch_norm")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_1")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ham10000_mobilenetv2")
    return model


def prepare_datasets(df: pd.DataFrame, batch_size: int, img_size: Tuple[int, int], seed: int = DEFAULT_SEED):
    """
    Splits dataframe using GroupShuffleSplit on lesion_id (leakage prevention)
    and constructs efficient tf.data pipelines with on-the-fly augmentation.
    """
    import tensorflow as tf
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.utils.class_weight import compute_class_weight

    groups = df["lesion_id"].astype(str) if "lesion_id" in df.columns else df["image_id"].astype(str)

    # 1. Split Train+Val vs Test (15% test)
    split1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trainval_idx, test_idx = next(split1.split(df, df["label"], groups))

    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # 2. Split Train vs Val (approx 15% of total, so ~17.65% of trainval)
    groups_trainval = (trainval_df["lesion_id"].astype(str) if "lesion_id" in trainval_df.columns
                       else trainval_df["image_id"].astype(str))
    split2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=seed)
    train_idx, val_idx = next(split2.split(trainval_df, trainval_df["label"], groups_trainval))

    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    logger.info(f"Split completed without lesion leakage: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Data augmentation for training
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
        tf.keras.layers.RandomRotation(0.15, seed=seed),
        tf.keras.layers.RandomZoom(0.1, seed=seed),
    ], name="training_augmentation")

    def load_and_preprocess(path, label, is_training=False):
        img_raw = tf.io.read_file(path)
        img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, img_size)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        if is_training:
            img = data_augmentation(tf.expand_dims(img, 0))[0]
        one_hot = tf.one_hot(label, depth=len(CANONICAL_CLASSES))
        return img, one_hot

    def build_tf_ds(frame: pd.DataFrame, is_training: bool):
        ds = tf.data.Dataset.from_tensor_slices((frame["path"].values, frame["label"].values))
        if is_training:
            ds = ds.shuffle(buffer_size=min(len(frame), 2000), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(lambda p, l: load_and_preprocess(p, l, is_training), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = build_tf_ds(train_df, True)
    val_ds = build_tf_ds(val_df, False)
    test_ds = build_tf_ds(test_df, False)

    # Compute balanced class weights
    unique_labels = np.arange(len(CANONICAL_CLASSES))
    train_labels = train_df["label"].values
    present_labels = np.unique(train_labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=present_labels,
        y=train_labels
    )
    class_weights = {int(label): float(w) for label, w in zip(present_labels, weights)}
    # Fill any missing class with neutral weight 1.0
    for c in unique_labels:
        if c not in class_weights:
            class_weights[c] = 1.0

    return (train_ds, val_ds, test_ds), (train_df, val_df, test_df), class_weights


def plot_training_history(history: Dict, save_path: str):
    """Saves training and validation accuracy/loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["loss"]) + 1)

    # Loss plot
    ax1.plot(epochs, history["loss"], "o-", label="Training Loss", color="#e74c3c")
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], "s--", label="Validation Loss", color="#3498db")
    ax1.set_title("Categorical Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, history["accuracy"], "o-", label="Training Accuracy", color="#2ecc71")
    if "val_accuracy" in history:
        ax2.plot(epochs, history["val_accuracy"], "s--", label="Validation Accuracy", color="#f39c12")
    ax2.set_title("Classification Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    logger.info(f"Training curves saved to: {save_path}")


def train(args):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(DEFAULT_SEED)

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    logger.info("Checking dataset availability...")
    dataset_summary = discover_dataset(args.data_dir)

    if not dataset_summary["is_available"]:
        logger.warning("Full HAM10000 dataset is not detected.")
        for issue in dataset_summary["issues"]:
            logger.warning(f"  - {issue}")
        
        if not args.smoke_test:
            logger.error(
                "Training aborted because dataset is missing or incomplete.\n"
                f"Please place HAM10000_metadata.csv and image folders under: {args.data_dir}\n"
                "To run a pipeline verification smoke test with sample images, use:\n"
                "  python train_model.py --smoke_test"
            )
            return False

        # Create demo samples for smoke test
        logger.info("Running smoke test mode with synthetic dermoscopic samples...")
        sample_paths = create_demo_samples("data/sample_images")
        sample_records = []
        for i, path in enumerate(sample_paths):
            cls = CANONICAL_CLASSES[i % len(CANONICAL_CLASSES)]
            sample_records.append({
                "image_id": os.path.splitext(os.path.basename(path))[0],
                "lesion_id": f"LES_{i}",
                "dx": cls,
                "path": path,
                "label": CLASS_TO_IDX[cls]
            })
        # Repeat to simulate multiple batches
        df = pd.DataFrame(sample_records * 8)
    else:
        df = dataset_summary["df"]

    (train_ds, val_ds, test_ds), (train_df, val_df, test_df), class_weights = prepare_datasets(
        df, batch_size=args.batch_size, img_size=DEFAULT_IMG_SIZE, seed=DEFAULT_SEED
    )

    # Save split records for evaluation scripts
    train_df.to_csv(os.path.join(args.results_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(args.results_dir, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(args.results_dir, "test_split.csv"), index=False)

    logger.info("Instantiating MobileNetV2 architecture...")
    model = build_mobilenetv2_model(num_classes=len(CANONICAL_CLASSES), img_size=DEFAULT_IMG_SIZE)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model_save_path = os.path.join(args.models_dir, "skin_lesion_model.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    epochs = 1 if args.smoke_test else args.epochs
    logger.info(f"Commencing model training for {epochs} epochs...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )
    # Ensure best model is saved to disk
    model.save(model_save_path)

    # Save canonical classes mapping
    classes_path = os.path.join(args.models_dir, "classes.json")
    with open(classes_path, "w") as f:
        json.dump(CANONICAL_CLASSES, f, indent=2)

    # Save training configuration
    config_path = os.path.join(args.models_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "architecture": "MobileNetV2",
            "img_size": list(DEFAULT_IMG_SIZE),
            "canonical_classes": CANONICAL_CLASSES,
            "class_metadata": CLASS_METADATA,
            "trained_epochs": len(history.history["loss"]),
            "is_smoke_test": args.smoke_test
        }, f, indent=2)

    # Save training curves and history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_path = os.path.join(args.results_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)

    plot_training_history(history_dict, os.path.join(args.results_dir, "training_curves.png"))

    logger.info("Training complete.")
    logger.info(f"Model saved to: {model_save_path}")
    logger.info(f"Classes saved to: {classes_path}")
    return True


if __name__ == "__main__":
    args = parse_args()
    train(args)
