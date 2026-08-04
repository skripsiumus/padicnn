"""
Training model CNN untuk klasifikasi penyakit daun padi.

Perintah paling mudah:
python train_model.py

Perintah opsional:
python train_model.py --epochs 30 --batch-size 32 --model simple_cnn
python train_model.py --epochs 15 --model mobilenet

Catatan:
- simple_cnn tidak perlu internet.
- mobilenet biasanya lebih bagus, tetapi pertama kali perlu internet untuk download bobot ImageNet.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models

from config import (
    BATCH_SIZE,
    CLASSIFICATION_REPORT_PATH,
    CLASS_NAMES_PATH,
    CONFUSION_MATRIX_PATH,
    DATA_DIR,
    DATASET_SUMMARY_PATH,
    EPOCHS,
    HISTORY_PATH,
    IMAGE_SIZE,
    MODEL_INFO_PATH,
    MODEL_PATH,
    SEED,
)
from utils import find_dataset_root, make_dataset_summary, save_class_names, save_json


def build_simple_cnn(image_size: int, num_classes: int) -> tf.keras.Model:
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.10),
        ],
        name="augmentasi_data",
    )

    model = models.Sequential(
        [
            layers.Input(shape=(image_size, image_size, 3)),
            augmentation,
            layers.Rescaling(1.0 / 255),

            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(256, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.40),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="Simple_CNN_Rice_Disease",
    )

    return model


def build_mobilenet(image_size: int, num_classes: int) -> tf.keras.Model:
    """Transfer learning berbasis MobileNetV2, masih termasuk keluarga CNN."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.10)(x)
    x = layers.RandomZoom(0.10)(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="MobileNetV2_Rice_Disease")


def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history_df: pd.DataFrame, output_dir: Path) -> None:
    if {"accuracy", "val_accuracy"}.issubset(history_df.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history_df["accuracy"], marker="o", label="Training Accuracy")
        ax.plot(history_df["val_accuracy"], marker="o", label="Validation Accuracy")
        ax.set_title("Grafik Akurasi Model")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "accuracy_plot.png", dpi=160)
        plt.close(fig)

    if {"loss", "val_loss"}.issubset(history_df.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history_df["loss"], marker="o", label="Training Loss")
        ax.plot(history_df["val_loss"], marker="o", label="Validation Loss")
        ax.set_title("Grafik Loss Model")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "loss_plot.png", dpi=160)
        plt.close(fig)


def evaluate_model(model: tf.keras.Model, val_ds, class_names: list[str], output_dir: Path) -> dict:
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(predictions, axis=1).tolist())

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(CLASSIFICATION_REPORT_PATH)

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(CONFUSION_MATRIX_PATH)

    accuracy = float(report.get("accuracy", 0.0))
    macro_f1 = float(report.get("macro avg", {}).get("f1-score", 0.0))

    print("\nClassification Report")
    print(report_df)
    print("\nConfusion Matrix")
    print(cm_df)

    return {"accuracy": accuracy, "macro_f1": macro_f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Training CNN penyakit daun padi")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Folder dataset")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Jumlah epoch")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--img-size", type=int, default=IMAGE_SIZE, help="Ukuran gambar")
    parser.add_argument("--model", choices=["simple_cnn", "mobilenet"], default="simple_cnn", help="Jenis model")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset_root = find_dataset_root(Path(args.data_dir))
    summary_df = make_dataset_summary(dataset_root)
    summary_df.to_csv(DATASET_SUMMARY_PATH, index=False)

    print(f"Dataset digunakan: {dataset_root}")
    print("\nRingkasan dataset:")
    print(summary_df)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_root,
        validation_split=0.2,
        subset="training",
        seed=args.seed,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        label_mode="int",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_root,
        validation_split=0.2,
        subset="validation",
        seed=args.seed,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        label_mode="int",
    )

    class_names = train_ds.class_names
    save_class_names(class_names, CLASS_NAMES_PATH)

    print("\nKelas terdeteksi:")
    for number, name in enumerate(class_names, start=1):
        print(f"{number}. {name}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    if args.model == "mobilenet":
        model = build_mobilenet(args.img_size, len(class_names))
    else:
        model = build_simple_cnn(args.img_size, len(class_names))

    model = compile_model(model)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(HISTORY_PATH, index=False)
    plot_history(history_df, MODEL_PATH.parent)

    model.save(MODEL_PATH)
    metrics = evaluate_model(model, val_ds, class_names, MODEL_PATH.parent)

    info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "model_type": args.model,
        "image_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "class_names": class_names,
        "num_classes": len(class_names),
        "num_images": int(summary_df["jumlah_gambar"].sum()) if not summary_df.empty else 0,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }
    save_json(info, MODEL_INFO_PATH)

    print("\nTraining selesai.")
    print(f"Model tersimpan di: {MODEL_PATH}")
    print(f"Akurasi validasi akhir: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
