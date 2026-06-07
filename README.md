# ASL Static Alphabet Recognizer

## What it does

A real-time American Sign Language (ASL) alphabet recognizer. It uses a webcam to read
hand signs and predict letters live, with on-screen feedback. It recognizes the 24
static letters of the ASL alphabet at about 89% live accuracy, with temporal smoothing
for stable predictions.

## How it works

- MediaPipe detects 21 hand landmarks from the webcam.
- A shared feature extractor (`landmark_utils.py`) converts the landmarks into 42
  scale-normalized x,y features (translation- and scale-invariant; z/depth was tested
  and removed because MediaPipe's estimated depth was too noisy and hurt accuracy).
- A scikit-learn Random Forest classifier predicts the letter.
- The live app shows a majority-vote (temporal smoothing) over the last ~10 frames so
  predictions stay stable instead of flickering.

## Letters supported

- **Supported:** the 24 static letters — A B C D E F G H I K L M N O P Q R S T U V W X Y
- **Not supported:** J and Z. These are motion-based letters (drawn in the air), and a
  single-frame static classifier cannot represent movement. They are planned as a
  separate future feature using sequence/motion recognition.

## Known limitations

- Trained on one person's hand in limited sessions, so it works best for that signer and
  may be less accurate for other people or very different lighting/angles.
- A few visually similar letters are weaker because x,y landmarks cannot see finger
  depth/crossing well: notably D (confused with L), I (with Y), B (with F), R (with U),
  and M (with N). Temporal smoothing reduces but does not eliminate this.
- J and Z are not recognized (see above).

## Project files (in `mvp/`)

| File | Purpose |
|---|---|
| `landmark_utils.py` | Shared feature extractor — 42 scale-normalized x,y features |
| `collect_data.py` | Records hand-landmark samples per letter from the webcam |
| `train_model.py` | Trains the Random Forest on collected data |
| `alphabet_landmarks.py` | Real-time recognition app with temporal smoothing |
| `evaluate_live.py` | Prompted live evaluation; saves a confusion matrix image |
| `fast_mvp.py` | Earlier MediaPipe gesture demo (kept for reference) |

## How to run

1. Activate the virtual environment (Windows): from the `mvp/` folder, run
   `..\venv\Scripts\activate`
2. **Recognition:** `python alphabet_landmarks.py` (press `q` to quit)
3. **Rebuild the model:** `python collect_data.py` then `python train_model.py`
4. **Measure accuracy:** `python evaluate_live.py` (produces `confusion_matrix.png`)

## Roadmap

- **Planned:** motion recognition for J and Z using a frame-sequence model (e.g. LSTM)
  on a separate branch.
- **Possible:** angle-based features to better separate D/I/B/R/M; assembling recognized
  letters into words; integration into a video-call plugin.
