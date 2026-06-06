# Requires mataas.h5 and mataas.json in the same folder.
# Model input: 224x224 RGB hand-crop, MobileNetV2 normalisation (x/127.5 - 1). Output: 24 classes (A-Y, no J/Z).

import json
import time
import threading
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
import tensorflow as tf

MODEL_PATH = "mataas.h5"
LABELS_PATH = "mataas.json"
HAND_MODEL_PATH = "gesture_recognizer.task"
IMG_SIZE = 224
OFFSET = 50
MIN_CONF = 0.6

_HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS


class _DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


def load_resources():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"DepthwiseConv2D": _DepthwiseConv2D},
        compile=False,
    )
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    return model, labels


def get_hand_crop(frame, landmarks):
    """Square crop centred on the hand bounding box — location-invariant."""
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + OFFSET
    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(w, int(cx + half))
    y2 = min(h, int(cy + half))
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for conn in _HAND_CONNECTIONS:
        cv2.line(frame, pts[conn.start], pts[conn.end], (0, 200, 0), 1)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)


def preprocess(crop):
    img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 127.5 - 1.0
    return np.expand_dims(img, axis=0)


class InferenceWorker:
    """Runs model.predict in a background thread; never blocks the camera loop."""

    def __init__(self, model, labels):
        self._model = model
        self._labels = labels
        self._lock = threading.Lock()
        self._pending_crop = None
        self._result = ("", 0.0)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, crop):
        with self._lock:
            self._pending_crop = crop

    def latest(self):
        with self._lock:
            return self._result

    def _run(self):
        while True:
            crop = None
            with self._lock:
                if self._pending_crop is not None:
                    crop = self._pending_crop
                    self._pending_crop = None
            if crop is not None:
                inp = preprocess(crop)
                probs = self._model({'input_2': inp}, training=False).numpy()[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
                with self._lock:
                    self._result = (self._labels[idx].upper(), conf)
            else:
                time.sleep(0.005)


def main():
    print("Loading model...")
    model, labels = load_resources()
    print("Model loaded.")

    worker = InferenceWorker(model, labels)

    opts = mp_vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.7,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    recognizer = mp_vision.GestureRecognizer.create_from_options(opts)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("CAMERA ERROR")
        return

    prev_time = time.time()
    use_full_frame = False
    print("Press 'f' to toggle full-frame / hand-crop mode. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                crop, (x1, y1, x2, y2) = get_hand_crop(frame, landmarks)

                input_img = frame if use_full_frame else crop
                if input_img.size > 0:
                    worker.submit(input_img)

                letter, conf = worker.latest()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if letter:
                    color = (0, 255, 0) if conf >= MIN_CONF else (0, 140, 255)
                    cv2.putText(
                        frame, f"{letter}  {conf:.2f}",
                        (x1, max(y1 - 15, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3,
                    )

                draw_landmarks(frame, landmarks)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            mode_label = "FULL" if use_full_frame else "CROP"
            cv2.putText(frame, f"FPS: {fps:.1f}  [{mode_label}]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("ASL Alphabet MVP", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                use_full_frame = not use_full_frame
                print(f"Mode: {'FULL FRAME' if use_full_frame else 'HAND CROP'}")

    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.close()


if __name__ == "__main__":
    main()
