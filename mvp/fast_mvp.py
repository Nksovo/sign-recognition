# gesture_recognizer.task must be in the same folder as this script

import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "gesture_recognizer.task"
MIN_SCORE = 0.6

def main():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("CAMERA ERROR")
        return

    timestamp_ms = 0
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)
            timestamp_ms += 33

            if result.gestures:
                gesture = result.gestures[0][0]
                if gesture.score >= MIN_SCORE:
                    label = f"{gesture.category_name} ({gesture.score:.2f})"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                    x = (frame.shape[1] - text_size[0]) // 2
                    cv2.putText(frame, label, (x, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Fast ASL MVP", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.close()

if __name__ == "__main__":
    main()
