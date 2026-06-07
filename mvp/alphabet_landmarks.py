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

# --- motion (J/Z) constants ---
INDEX_TIP         = 8
PINKY_TIP         = 20
MIN_MOTION        = 80
J_VERTICAL_MARGIN = 30
J_HOOK_WINDOW     = 8
Z_MIN_REVERSALS   = 2

# --- motion state ---
mode             = 'STATIC'   # 'STATIC' | 'MOTION'
path_index       = deque(maxlen=30)
path_pinky       = deque(maxlen=30)
recording        = False
last_motion_result = ''

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


def _path_distance(pts):
    total = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        total += (dx*dx + dy*dy) ** 0.5
    return total


def _draw_path(frame, pts, color):
    pt_list = list(pts)
    for i in range(1, len(pt_list)):
        cv2.line(frame, pt_list[i-1], pt_list[i], color, 2)
    if pt_list:
        cv2.circle(frame, pt_list[-1], 5, color, -1)


def _count_x_reversals(pts):
    reversals, last_sign = 0, 0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        if dx == 0:
            continue
        sign = 1 if dx > 0 else -1
        if last_sign != 0 and sign != last_sign:
            reversals += 1
        last_sign = sign
    return reversals


def classify_motion(pi, pp):
    """Returns ('J'|'Z'|'UNCLEAR', diag dict). Mirrors motion_jz.py logic."""
    idx_dist   = _path_distance(pi)
    pinky_dist = _path_distance(pp)
    diag = {'index_dist': round(idx_dist, 1), 'pinky_dist': round(pinky_dist, 1),
            'net_vertical': None, 'x_reversals': None}

    if pinky_dist > idx_dist and pinky_dist > MIN_MOTION:
        if len(pp) < 4:
            return 'UNCLEAR', diag
        net_v = pp[-1][1] - pp[0][1]
        diag['net_vertical'] = round(net_v, 1)
        tail = pp[-J_HOOK_WINDOW:] if len(pp) >= J_HOOK_WINDOW else pp
        if net_v > J_VERTICAL_MARGIN and _count_x_reversals(tail) >= 1:
            return 'J', diag
        return 'UNCLEAR', diag
    elif idx_dist >= pinky_dist and idx_dist > MIN_MOTION:
        if len(pi) < 4:
            return 'UNCLEAR', diag
        rev = _count_x_reversals(pi)
        diag['x_reversals'] = rev
        return ('Z' if rev >= Z_MIN_REVERSALS else 'UNCLEAR'), diag
    return 'UNCLEAR', diag


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

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = recognizer.recognize_for_video(mp_image, timestamp_ms)

        if mode == 'STATIC':
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

                    color = (0, 255, 0) if stability >= 0.6 else (0, 140, 255)

                    x1, y1, x2, y2 = get_bbox(frame, landmarks)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{smoothed_letter}  ({smoothed_count}/{SMOOTH_WINDOW})'
                    cv2.putText(frame, label,
                                (x1, max(y1 - 15, 30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
            else:
                recent_preds.append(None)

        else:  # MOTION mode
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                idx_pt   = (int(landmarks[INDEX_TIP].x * w),
                            int(landmarks[INDEX_TIP].y * h))
                pinky_pt = (int(landmarks[PINKY_TIP].x * w),
                            int(landmarks[PINKY_TIP].y * h))
                if recording:
                    path_index.append(idx_pt)
                    path_pinky.append(pinky_pt)
                cv2.circle(frame, idx_pt,   6, (255, 100,  0), -1)
                cv2.circle(frame, pinky_pt, 6, (0,   200,  0), -1)

            _draw_path(frame, path_index, (255, 100,  0))
            _draw_path(frame, path_pinky, (0,   200,  0))

            rec_label = 'RECORDING' if recording else 'idle  (SPACE to record)'
            rec_color = (0, 0, 220) if recording else (180, 180, 180)
            cv2.putText(frame, rec_label, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, rec_color, 2)

            if last_motion_result:
                res_color = (0, 220, 0) if last_motion_result in ('J', 'Z') else (0, 140, 255)
                cv2.putText(frame, last_motion_result, (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.0, res_color, 6)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        mode_color = (200, 200, 0) if mode == 'MOTION' else (200, 200, 200)
        cv2.putText(frame, f'MODE: {mode}  (M to toggle)',
                    (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 1)

        cv2.imshow('ASL Landmarks', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('m'):
            mode = 'MOTION' if mode == 'STATIC' else 'STATIC'
            path_index.clear()
            path_pinky.clear()
            recording = False
            last_motion_result = ''
            recent_preds.clear()
        if key == ord(' ') and mode == 'MOTION':
            if not recording:
                path_index.clear()
                path_pinky.clear()
                last_motion_result = ''
                recording = True
            else:
                recording = False
                pi = list(path_index)
                pp = list(path_pinky)
                label, diag = classify_motion(pi, pp)
                last_motion_result = label
                print(f'J/Z => {label} | {diag}')

except Exception as e:
    print(f'ERROR: {e}')
finally:
    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()
