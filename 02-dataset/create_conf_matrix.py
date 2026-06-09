#!/usr/bin/env python3

import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import tensorflow as tf


LABELS = ["like", "dislike", "stop", "rock", "peace"]
IMG_SIZE = 96
SEED = 42
EPOCHS = 5
BATCH_SIZE = 32

HAGRID_DIR = ROOT / "misc/gesture_dataset_sample"
HAGRID_ANNOT_DIR = HAGRID_DIR / "_annotations"
SUBMISSION_IMAGE_DIR = ROOT / "02-dataset/images"
SUBMISSION_ANNOTATIONS = ROOT / "02-dataset/annot-lennart-bart.json"
OUTPUT_PATH = ROOT / "02-dataset/conf-matrix.png"


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def crop_bbox(image: np.ndarray, bbox: list[float]) -> np.ndarray:
    height, width = image.shape[:2]
    x, y, w, h = bbox
    left = max(0, int(round(x * width)))
    top = max(0, int(round(y * height)))
    right = min(width, int(round((x + w) * width)))
    bottom = min(height, int(round((y + h) * height)))
    return image[top:bottom, left:right]


def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype("float32") / 255.0


def load_hagrid_training_data() -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []

    for label_index, label in enumerate(LABELS):
        annotations = load_json(HAGRID_ANNOT_DIR / f"{label}.json")
        image_paths = sorted((HAGRID_DIR / label).glob("*.jpg"))
        for image_path in image_paths:
            entry = annotations.get(image_path.stem)
            if not entry or not entry.get("bboxes"):
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            crop = crop_bbox(image, entry["bboxes"][0])
            if crop.size == 0:
                continue
            images.append(preprocess_crop(crop))
            labels.append(label_index)

    return np.array(images), np.array(labels)


def load_submission_data() -> tuple[np.ndarray, np.ndarray]:
    annotations = load_json(SUBMISSION_ANNOTATIONS)
    images: list[np.ndarray] = []
    labels: list[int] = []

    for label_index, label in enumerate(LABELS):
        for image_path in sorted((SUBMISSION_IMAGE_DIR / label).glob("*.jpg")):
            entry = annotations[image_path.stem]
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read {image_path}")
            crop = crop_bbox(image, entry["bboxes"][0])
            images.append(preprocess_crop(crop))
            labels.append(label_index)

    return np.array(images), np.array(labels)


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(LABELS), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))

    fig = plt.figure(figsize=(10, 10))
    ConfusionMatrixDisplay(conf_matrix, display_labels=LABELS).plot(ax=plt.gca())
    plt.xticks(rotation=90, ha="center")
    plt.title("Task 2 confusion matrix")
    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH)
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}", flush=True)


def main() -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    x, y = load_hagrid_training_data()
    print(f"Loaded {len(x)} HaGRID training crops.", flush=True)
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )

    model = build_model()
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    x_submission, y_submission = load_submission_data()
    predictions = model.predict(x_submission, verbose=0)
    y_pred = predictions.argmax(axis=1)
    print(f"Predicted {len(x_submission)} submitted images.", flush=True)
    save_confusion_matrix(y_submission, y_pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
