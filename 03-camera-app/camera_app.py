#!/usr/bin/env python3

import argparse
import json
import os
import random
import time
from collections import deque

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib-cache"))

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


LABELS = ["like", "dislike", "stop", "peace"]
FILTER_NAMES = ["normal", "sepia", "sketch"]
FILTER_STRENGTHS = [0.0, 0.25, 0.50, 0.75, 1.0]
IMG_SIZE = 96
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Gesture-controlled selfie camera.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--dataset", default="misc/gesture_dataset_sample")
    train.add_argument("--model", default="03-camera-app/gesture_camera.keras")
    train.add_argument("--epochs", type=int, default=15)
    train.add_argument("--batch-size", type=int, default=32)

    run = subparsers.add_parser("run")
    run.add_argument("--model", default="03-camera-app/gesture_camera.keras")
    run.add_argument("--timer", type=float, default=3.0)
    run.add_argument("--output", default="03-camera-app/selfie.jpg")
    run.add_argument("--camera", type=int, default=0)
    run.add_argument("--confidence", type=float, default=0.85)
    run.add_argument("--margin", type=float, default=0.20)
    run.add_argument("--repeat-delay", type=float, default=3.0)
    run.add_argument("--roi-scale", type=float, default=0.45)

    return parser.parse_args()


def bbox_to_pixels(bbox, image):
    height, width = image.shape[:2]
    x, y, w, h = bbox
    return int(x * width), int(y * height), int((x + w) * width), int((y + h) * height)


def preprocess_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype("float32") / 255.0


def load_training_data(dataset_path):
    images = []
    labels = []

    for label_index, label in enumerate(LABELS):
        annotation_path = os.path.join(dataset_path, "_annotations", f"{label}.json")
        with open(annotation_path) as f:
            annotations = json.load(f)
        image_path = os.path.join(dataset_path, label)

        for filename in sorted(os.listdir(image_path)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            uid = filename.split(".")[0]
            if uid not in annotations:
                continue

            img = cv2.imread(os.path.join(image_path, filename))
            if img is None:
                continue

            annotation = annotations[uid]
            for i, bbox in enumerate(annotation["bboxes"]):
                if annotation["labels"][i] != label:
                    continue

                x1, y1, x2, y2 = bbox_to_pixels(bbox, img)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                images.append(preprocess_image(crop))
                labels.append(label_index)

    return np.array(images), np.array(labels)


def build_model():
    from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
    from keras.models import Sequential

    model = Sequential(
        [
            Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(len(LABELS), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(args):
    from keras.callbacks import EarlyStopping
    from keras.utils import set_random_seed

    random.seed(SEED)
    np.random.seed(SEED)
    set_random_seed(SEED)

    x, y = load_training_data(args.dataset)
    print("loaded images:", len(x))

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    model = build_model()
    stop_early = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[stop_early],
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    model_dir = os.path.dirname(args.model)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    model.save(args.model)
    print("test accuracy:", round(float(accuracy), 4))
    print("test loss:", round(float(loss), 4))
    print("saved model:", args.model)


def center_crop(frame, scale):
    height, width = frame.shape[:2]
    size = int(min(width, height) * scale)
    x1 = (width - size) // 2
    y1 = (height - size) // 2
    x2 = x1 + size
    y2 = y1 + size
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def predict_gesture(model, crop):
    sample = preprocess_image(crop).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    prediction = model.predict(sample, verbose=0)[0]
    top_indices = np.argsort(prediction)[-2:][::-1]
    confidence = float(prediction[top_indices[0]])
    margin = float(prediction[top_indices[0]] - prediction[top_indices[1]])
    return LABELS[top_indices[0]], confidence, margin


def apply_sepia(frame):
    kernel = np.array(
        [
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189],
        ]
    )
    filtered = cv2.transform(frame, kernel)
    return np.clip(filtered, 0, 255).astype("uint8")


def apply_sketch(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    sketch = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        2,
    )
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def apply_filter(frame, filter_index, strength_index):
    strength = FILTER_STRENGTHS[strength_index]
    if filter_index == 0 or strength == 0.0:
        return frame

    if filter_index == 1:
        filtered = apply_sepia(frame)
    else:
        filtered = apply_sketch(frame)
    return cv2.addWeighted(filtered, strength, frame, 1.0 - strength, 0)


def stable_prediction(history):
    if len(history) == history.maxlen and history[0] is not None and len(set(history)) == 1:
        return history[0]
    return None


def draw_overlay(
    frame,
    roi_box,
    label,
    confidence,
    margin,
    accepted_label,
    filter_index,
    strength_index,
    repeat_delay,
    capture_time,
):
    x1, y1, x2, y2 = roi_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 2)

    strength = int(FILTER_STRENGTHS[strength_index] * 100)
    lines = [
        f"prediction: {label} {confidence:.2f} margin {margin:.2f}",
        f"accepted: {accepted_label or '-'}",
        f"filter: {FILTER_NAMES[filter_index]}",
        f"strength: {strength}%",
        f"repeat delay: {repeat_delay:.1f}s",
        "stop=capture  peace=filter  like=strength+  dislike=strength-  q=quit",
    ]

    if capture_time is not None:
        remaining = max(0, capture_time - time.time())
        lines.insert(0, f"capturing in {remaining:.1f}s")

    y = 28
    for line in lines:
        cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        y += 28


def run_camera(args):
    from keras.models import load_model

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"model not found: {args.model}")

    model = load_model(args.model)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("could not open camera")

    history = deque(maxlen=3)
    last_action = 0
    filter_index = 0
    strength_index = 0
    capture_time = None

    print("controls: stop=capture, peace=filter, like=strength+, dislike=strength-, q=quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        roi, roi_box = center_crop(frame, args.roi_scale)
        label, confidence, margin = predict_gesture(model, roi)
        accepted_label = label if confidence >= args.confidence and margin >= args.margin else None
        history.append(accepted_label)
        gesture = stable_prediction(history)

        now = time.time()
        if gesture is not None and now - last_action > args.repeat_delay:
            if gesture == "stop" and capture_time is None:
                capture_time = now + args.timer
            elif gesture == "peace":
                filter_index = (filter_index + 1) % len(FILTER_NAMES)
            elif gesture == "like":
                strength_index = min(len(FILTER_STRENGTHS) - 1, strength_index + 1)
            elif gesture == "dislike":
                strength_index = max(strength_index - 1, 0)
            last_action = now
            history.clear()

        display = apply_filter(frame.copy(), filter_index, strength_index)

        if capture_time is not None and now >= capture_time:
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(args.output, display)
            print("saved:", args.output)
            capture_time = None

        draw_overlay(
            display,
            roi_box,
            label,
            confidence,
            margin,
            accepted_label,
            filter_index,
            strength_index,
            args.repeat_delay,
            capture_time,
        )
        cv2.imshow("Gesture Camera", display)

        if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    if args.command == "train":
        train_model(args)
    elif args.command == "run":
        run_camera(args)


if __name__ == "__main__":
    main()
