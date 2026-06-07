import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions
from collections import deque

HAND_MODEL_PATH = 'gesture_recognizer.task'

INDEX_TIP  = 8
PINKY_TIP  = 20

# --- exact copy of alphabet_landmarks.py detection setup ---
class _HandWrapper:
    def __init__(self, landmark_list):
        self.landmark = landmark_list

opts = mp_vision.GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.7,
    running_mode=mp_vision.RunningMode.VIDEO,
)
recognizer = mp_vision.GestureRecognizer.create_from_options(opts)
# --- end copy ---

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('CAMERA ERROR')
    raise SystemExit

MIN_MOTION          = 80    # px — below this = not enough movement
J_VERTICAL_MARGIN   = 30    # px — net downward travel required for J
J_HOOK_WINDOW       = 8     # last N points checked for the hook reversal
Z_MIN_REVERSALS     = 2     # horizontal direction changes required for Z

path_index  = deque(maxlen=30)
path_pinky  = deque(maxlen=30)
recording   = False
last_result = ''   # displayed until next recording


def path_distance(pts):
    total = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        total += (dx*dx + dy*dy) ** 0.5
    return total


def draw_path(frame, pts, color):
    pt_list = list(pts)
    for i in range(1, len(pt_list)):
        cv2.line(frame, pt_list[i-1], pt_list[i], color, 2)
    if pt_list:
        cv2.circle(frame, pt_list[-1], 5, color, -1)


def count_x_reversals(pts):
    """Count sign changes in consecutive x-steps (left/right direction flips)."""
    reversals = 0
    last_sign = 0
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
    """
    pi, pp — lists of (x,y) pixel points for index tip and pinky tip.
    Returns ('J'|'Z'|'UNCLEAR', dict of diagnostic numbers).
    """
    idx_dist   = path_distance(pi)
    pinky_dist = path_distance(pp)

    diag = {
        'index_dist':   round(idx_dist,   1),
        'pinky_dist':   round(pinky_dist, 1),
        'net_vertical': None,
        'x_reversals':  None,
    }

    # --- decide active finger ---
    if pinky_dist > idx_dist and pinky_dist > MIN_MOTION:
        # candidate J — check pinky path shape
        if len(pp) < 4:
            return 'UNCLEAR', diag
        net_vertical = pp[-1][1] - pp[0][1]   # positive = moved down (y grows down)
        diag['net_vertical'] = round(net_vertical, 1)

        # hook: look for a horizontal direction reversal in the last J_HOOK_WINDOW pts
        tail = pp[-J_HOOK_WINDOW:] if len(pp) >= J_HOOK_WINDOW else pp
        hook = count_x_reversals(tail) >= 1

        if net_vertical > J_VERTICAL_MARGIN and hook:
            return 'J', diag
        return 'UNCLEAR', diag

    elif idx_dist >= pinky_dist and idx_dist > MIN_MOTION:
        # candidate Z — check index path for zigzag
        if len(pi) < 4:
            return 'UNCLEAR', diag
        reversals = count_x_reversals(pi)
        diag['x_reversals'] = reversals

        if reversals >= Z_MIN_REVERSALS:
            return 'Z', diag
        return 'UNCLEAR', diag

    else:
        return 'UNCLEAR', diag


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # --- exact copy of alphabet_landmarks.py live path ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = recognizer.recognize_for_video(mp_image, timestamp_ms)
        # --- end copy ---

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            idx_pt   = (int(landmarks[INDEX_TIP].x * w),
                        int(landmarks[INDEX_TIP].y * h))
            pinky_pt = (int(landmarks[PINKY_TIP].x * w),
                        int(landmarks[PINKY_TIP].y * h))

            if recording:
                path_index.append(idx_pt)
                path_pinky.append(pinky_pt)

            # draw live fingertip dots even when idle
            cv2.circle(frame, idx_pt,   6, (255, 100,   0), -1)
            cv2.circle(frame, pinky_pt, 6, (0,   200,   0), -1)

        # draw recorded paths
        draw_path(frame, path_index, (255, 100,   0))  # blue  = index
        draw_path(frame, path_pinky, (0,   200,   0))  # green = pinky

        # status overlay
        if recording:
            cv2.putText(frame, 'RECORDING', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 220), 3)
        else:
            cv2.putText(frame, 'idle', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (180, 180, 180), 2)

        # classification result
        if last_result:
            color = (0, 220, 0) if last_result in ('J', 'Z') else (0, 140, 255)
            cv2.putText(frame, last_result, (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 4.0, color, 6)

        cv2.putText(frame,
                    'SPACE = start/stop recording   Q = quit',
                    (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, 'blue=index  green=pinky',
                    (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('motion_jz', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            if not recording:
                path_index.clear()
                path_pinky.clear()
                last_result = ''
                recording = True
                print('--- Recording started ---')
            else:
                recording = False
                pi = list(path_index)
                pp = list(path_pinky)
                label, diag = classify_motion(pi, pp)
                last_result = label
                print(f'--- Recording stopped ---')
                print(f'  Index tip : {len(pi)} points, distance = {diag["index_dist"]} px')
                print(f'  Pinky tip : {len(pp)} points, distance = {diag["pinky_dist"]} px')
                print(f'  net_vertical = {diag["net_vertical"]}  '
                      f'x_reversals = {diag["x_reversals"]}')
                print(f'  => CLASSIFIED AS: {label}')

except Exception as e:
    print(f'ERROR: {e}')
finally:
    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()
