import cv2
import pickle
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
from landmark_utils import extract_features
from collections import deque, Counter

SMOOTH_WINDOW = 10
recent_preds = deque(maxlen=SMOOTH_WINDOW)

HAND_MODEL_PATH = 'gesture_recognizer.task'
MIN_CONF = 0.35

try:
    with open('landmark_model.pickle', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print('NO MODEL - run train_model.py first')
    raise SystemExit

_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS


class _HandWrapper:
    def __init__(self, landmark_list):
        self.landmark = landmark_list


def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for conn in _CONNECTIONS:
        cv2.line(frame, pts[conn.start], pts[conn.end], (0, 200, 0), 1)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)


def get_bbox(frame, landmarks):
    h, w = frame.shape[:2]
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    pad = 20
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w, max(xs) + pad)
    y2 = min(h, max(ys) + pad)
    return x1, y1, x2, y2


opts = mp_vision.GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.7,
    running_mode=mp_vision.RunningMode.VIDEO,
)
recognizer = mp_vision.GestureRecognizer.create_from_options(opts)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('CAMERA ERROR')
    raise SystemExit

prev_time = time.time()

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
            wrapped = _HandWrapper(landmarks)

            draw_landmarks(frame, landmarks)

            features = extract_features(wrapped)
            probs = model.predict_proba([features])[0]
            raw_letter = model.classes_[int(np.argmax(probs))]

            recent_preds.append(raw_letter)

            non_none = [p for p in recent_preds if p is not None]
            if non_none:
                smoothed_letter, smoothed_count = Counter(non_none).most_common(1)[0]
                stability = smoothed_count / SMOOTH_WINDOW

                if stability >= 0.6:
                    color = (0, 255, 0)
                else:
                    color = (0, 140, 255)

                x1, y1, x2, y2 = get_bbox(frame, landmarks)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{smoothed_letter}  ({smoothed_count}/{SMOOTH_WINDOW})'
                cv2.putText(frame, label,
                            (x1, max(y1 - 15, 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        else:
            recent_preds.append(None)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('ASL Landmarks', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f'ERROR: {e}')
finally:
    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()
