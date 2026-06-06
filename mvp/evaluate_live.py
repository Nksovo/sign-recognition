import cv2
import pickle
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
from sklearn.metrics import confusion_matrix, accuracy_score
from landmark_utils import extract_features

HAND_MODEL_PATH = 'gesture_recognizer.task'
FRAMES_PER_LETTER = 30

try:
    with open('landmark_model.pickle', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print('NO MODEL')
    raise SystemExit

TEST_LETTERS = sorted(list(model.classes_))

_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS


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

results_pairs = []   # list of (true_label, pred_label)
tested_letters = []
quit_early = False

try:
    for letter in TEST_LETTERS:
        if quit_early:
            break

        # READY screen
        action = None
        while action is None:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            # dark overlay for readability
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
            # big letter centred
            txt = f'Show: {letter}'
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 4, 8)
            cv2.putText(frame, txt, ((w - tw) // 2, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8)
            # progress indicator
            idx = TEST_LETTERS.index(letter) + 1
            cv2.putText(frame, f'Letter {idx}/{len(TEST_LETTERS)}',
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            # instructions at bottom
            cv2.putText(frame,
                        'S = start  |  K = skip  |  Q = quit',
                        (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Live Evaluation', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                action = 'start'
            elif key == ord('k'):
                action = 'skip'
            elif key == ord('q'):
                action = 'quit'
                quit_early = True

        if action in ('skip', 'quit'):
            continue

        # Collect FRAMES_PER_LETTER predictions (throttled to 1 per 0.5s)
        tested_letters.append(letter)
        count = 0
        last_capture = 0.0
        while count < FRAMES_PER_LETTER:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            current_pred = ''
            now = time.time()
            if (result.hand_landmarks and len(result.hand_landmarks) == 1
                    and now - last_capture >= 0.5):
                wrapped = _HandWrapper(result.hand_landmarks[0])
                features = extract_features(wrapped)
                current_pred = model.predict([features])[0]
                results_pairs.append((letter, current_pred))
                count += 1
                last_capture = now

            cv2.putText(frame, f'{letter}: {count}/{FRAMES_PER_LETTER}',
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            if current_pred:
                color = (0, 255, 0) if current_pred == letter else (0, 0, 255)
                cv2.putText(frame, f'Pred: {current_pred}',
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.imshow('Live Evaluation', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                quit_early = True
                break

except Exception as e:
    print(f'ERROR: {e}')
finally:
    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()

# Report
if not results_pairs:
    print('No data collected.')
    raise SystemExit

y_true = [p[0] for p in results_pairs]
y_pred = [p[1] for p in results_pairs]

overall = accuracy_score(y_true, y_pred)
print(f'\nOverall accuracy: {overall * 100:.1f}%\n')

# Per-letter accuracy sorted worst to best
per_letter = {}
for letter in tested_letters:
    pairs = [(t, p) for t, p in results_pairs if t == letter]
    correct = sum(1 for t, p in pairs if t == p)
    total = len(pairs)
    per_letter[letter] = (correct, total)

print('Per-letter accuracy (worst to best):')
sorted_letters = sorted(per_letter, key=lambda l: per_letter[l][0] / per_letter[l][1])
for l in sorted_letters:
    correct, total = per_letter[l]
    pct = correct / total * 100
    print(f'  {l}: {pct:.0f}%  ({correct}/{total})')

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=tested_letters)
fig, ax = plt.subplots(figsize=(max(8, len(tested_letters)), max(6, len(tested_letters) - 2)))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=tested_letters,
            yticklabels=tested_letters, ax=ax, cmap='Blues')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('ASL Live Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
print('\nSaved confusion_matrix.png')

# Plain-language summary
weak = [(l, per_letter[l][0] / per_letter[l][1]) for l in tested_letters
        if per_letter[l][0] / per_letter[l][1] < 0.80]
if not weak:
    print('\nAll tested letters scored >= 80%. No weak letters.')
else:
    print('\nLetters under 80% accuracy:')
    for l, acc in sorted(weak, key=lambda x: x[1]):
        row_idx = tested_letters.index(l)
        row = cm[row_idx].copy()
        row[row_idx] = 0  # zero out correct diagonal
        confused_with_idx = int(np.argmax(row))
        confused_with = tested_letters[confused_with_idx]
        confused_count = row[confused_with_idx]
        print(f'  {l} ({acc*100:.0f}%) — most confused with {confused_with} ({confused_count}x)')
