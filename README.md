[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/cMaQVOgt)

# Assignment 05

## Contents

- `01-hyperparameters/hyperparameters.ipynb`: Task 1 activation-function experiment.
- `02-dataset/`: Task 2 annotations, submitted images, confusion-matrix script, and output.
- `03-camera-app/camera_app.py`: Task 3 gesture-controlled camera app.

Install the Python dependencies into a local Python 3.11 virtual environment:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Task 3: Gesture-controlled camera app

The app uses a small CNN trained on four hand-pose classes: `like`, `dislike`, `stop`, and `peace`. Run it with a trained model:

```bash
.venv/bin/python 03-camera-app/camera_app.py run \
  --model 03-camera-app/gesture_camera.keras \
  --timer 3 \
  --output 03-camera-app/selfie.jpg
```

Main gestures:

- `stop`: start the selfie countdown and save the filtered frame.
- `peace`: cycle through `normal`, `sepia`, and `sketch` filters.
- `like`: increase filter strength.
- `dislike`: decrease filter strength.

The live view shows the prediction region, confidence, margin, active filter, strength, and repeat delay. Actions require the same accepted prediction for three frames; use `q` or `Esc` to quit. Run `--help` on the `run` command for camera, confidence, margin, repeat-delay, and ROI options.

If `03-camera-app/gesture_camera.keras` is missing, retrain it from the local HaGRID subset:

```bash
.venv/bin/python 03-camera-app/camera_app.py train --dataset misc/gesture_dataset_sample --model 03-camera-app/gesture_camera.keras
```

The HaGRID subset in `misc/` is intentionally ignored because the assignment says not to commit the dataset.
