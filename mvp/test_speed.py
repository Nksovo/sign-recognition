import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "gesture_recognizer.task"
RUNS = 100

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
)
recognizer = vision.GestureRecognizer.create_from_options(options)

blank = np.zeros((480, 640, 3), dtype=np.uint8)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=blank)

times = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    recognizer.recognize(mp_image)
    times.append((time.perf_counter() - t0) * 1000)

recognizer.close()

avg_ms = sum(times) / len(times)
cps = 1000.0 / avg_ms
print(f"Average: {avg_ms:.2f} ms/call")
print(f"Throughput: {cps:.1f} calls/sec")
print("FAST" if avg_ms < 100 else "SLOW")
