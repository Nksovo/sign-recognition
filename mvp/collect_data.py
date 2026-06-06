import cv2
import numpy as np
import pickle
import os
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions
from landmark_utils import extract_features

LETTERS = ['M', 'N']
SAMPLES_PER_LETTER = 100
DATA_FILE = 'landmark_data.pickle'
HAND_MODEL_PATH = 'gesture_recognizer.task'


class _HandWrapper:
    """Makes Tasks API landmark list look like mp.solutions hand_landmarks."""
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
    exit()

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    print(f'Loaded existing data: {len(data["labels"])} samples')
else:
    data = {'features': [], 'labels': []}

try:
    import time
    for letter in LETTERS:
        # Remove any existing samples for this letter before recollecting
        keep = [(f, l) for f, l in zip(data['features'], data['labels']) if l != letter]
        removed = len(data['labels']) - len(keep)
        if removed:
            print(f'Removed {removed} old samples for {letter}')
        data['features'] = [f for f, l in keep]
        data['labels'] = [l for f, l in keep]

        # Wait for user to press S or Q
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, letter, (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 0), 12)
            cv2.putText(frame, f'Show sign for {letter}. Press S to start, Q to quit',
                        (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Collect Data', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                break
            if key == ord('q'):
                raise KeyboardInterrupt

        # Capture SAMPLES_PER_LETTER samples
        count = 0
        while count < SAMPLES_PER_LETTER:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks and len(result.hand_landmarks) == 1:
                wrapped = _HandWrapper(result.hand_landmarks[0])
                features = extract_features(wrapped)
                data['features'].append(features)
                data['labels'].append(letter)
                count += 1

            cv2.putText(frame, f'{letter}: {count}/{SAMPLES_PER_LETTER}',
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
            cv2.imshow('Collect Data', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

except KeyboardInterrupt:
    print('Collection interrupted.')
finally:
    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()

with open(DATA_FILE, 'wb') as f:
    pickle.dump(data, f)

from collections import Counter
counts = Counter(data['labels'])
for ltr in sorted(counts.keys()):
    print(f'  {ltr}: {counts[ltr]} samples')
print(f'Total: {len(data["labels"])} samples saved to {DATA_FILE}')
