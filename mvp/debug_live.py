"""
Temporary debug script — mirrors alphabet_landmarks.py's feature pipeline exactly.
Show the letter 'L' to the camera when prompted.
"""
import cv2
import pickle
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions
from landmark_utils import extract_features

TARGET_LETTER = 'L'
HAND_MODEL_PATH = 'gesture_recognizer.task'

# --- exact copy of alphabet_landmarks.py setup ---
with open('landmark_model.pickle', 'rb') as f:
    model = pickle.load(f)

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

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('CAMERA ERROR')
    raise SystemExit

# Load a stored training vector for TARGET_LETTER
with open('landmark_data.pickle', 'rb') as f:
    data = pickle.load(f)
stored_vecs = [f for f, l in zip(data['features'], data['labels']) if l == TARGET_LETTER]
stored_vec = np.array(stored_vecs[-1])  # last stored sample

print(f"\n=== STORED training vector for '{TARGET_LETTER}' ===")
print(f"  values : {list(np.round(stored_vec, 4))}")
print(f"  min={stored_vec.min():.6f}  max={stored_vec.max():.6f}  mean={stored_vec.mean():.6f}")

print(f"\n>>> Show '{TARGET_LETTER}' to the camera. A window will open — put your hand in frame.\n")

try:
    for frame_idx in range(300):
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, f"Show '{TARGET_LETTER}' — waiting for hand ({frame_idx}/300)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('debug_live', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # --- exact copy of alphabet_landmarks.py live path ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = recognizer.recognize_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            wrapped = _HandWrapper(landmarks)
            features = extract_features(wrapped)
            # --- end of copied path ---

            live_vec = np.array(features)

            print(f"=== LIVE vector (frame {frame_idx}) ===")
            print(f"  values : {list(np.round(live_vec, 4))}")
            print(f"  min={live_vec.min():.6f}  max={live_vec.max():.6f}  mean={live_vec.mean():.6f}")

            pred = model.predict([features])[0]
            probs = model.predict_proba([features])[0]
            top3_idx = np.argsort(probs)[::-1][:3]

            print(f"\n  model.predict        = '{pred}'")
            print(f"  top-3 probabilities:")
            for i in top3_idx:
                print(f"    {model.classes_[i]} : {probs[i]:.4f}")

            print(f"\n=== SIDE-BY-SIDE (first 9 values) ===")
            print(f"  stored '{TARGET_LETTER}': {list(np.round(stored_vec[:9], 4))}")
            print(f"  live          : {list(np.round(live_vec[:9], 4))}")

            diff = np.abs(live_vec - stored_vec)
            print(f"\n  |live - stored|  mean={diff.mean():.4f}  max={diff.max():.4f}")
            break
    else:
        print("No hand detected in 60 frames.")
finally:
    cap.release()
    recognizer.close()
    cv2.destroyAllWindows()
