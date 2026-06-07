# ASL Alphabet Recognizer

## What it does

A real-time American Sign Language (ASL) alphabet recognizer. It uses a webcam to read
hand signs and predict all 26 letters live, with on-screen feedback. The 24 static
letters run at about 89% live accuracy with temporal smoothing. J and Z are recognized
via trajectory-based motion recognition using a mode toggle.

## How it works

- MediaPipe detects 21 hand landmarks from the webcam.
- A shared feature extractor (`landmark_utils.py`) converts the landmarks into 42
  scale-normalized x,y features (translation- and scale-invariant; z/depth was tested
  and removed because MediaPipe's estimated depth was too noisy and hurt accuracy).
- A scikit-learn Random Forest classifier predicts static letters (A–Y excluding J/Z).
- The live app shows a majority-vote (temporal smoothing) over the last ~10 frames so
  predictions stay stable instead of flickering.
- J and Z use a separate motion path: the app tracks the fingertip trajectory over a
  short recording window and classifies by shape — a downward pinky hook for J, an
  index-finger zigzag for Z.

## Letters supported

All 26 ASL alphabet letters are recognized:

- **A–Y (excluding J and Z) — 24 static letters** recognized by a single-frame Random
  Forest classifier. Press no special key; predictions appear continuously.
- **J and Z — motion letters** recognized by fingertip trajectory. Press `m` to switch
  to MOTION mode, then press `SPACE` to start a short recording, sign J or Z, and press
  `SPACE` again to see the result. Press `m` again to return to static mode.
  - J: pinky finger traces a downward hook.
  - Z: index finger traces a left-right-left zigzag.

## Known limitations

- Trained on one person's hand in limited sessions, so it works best for that signer and
  may be less accurate for other people or very different lighting/angles.
- A few visually similar static letters are weaker because x,y landmarks cannot see
  finger depth/crossing well: notably D (confused with L), I (with Y), B (with F),
  R (with U), and M (with N). Temporal smoothing reduces but does not eliminate this.
- J/Z motion classification uses geometric heuristics (hook + vertical travel for J;
  direction-reversal count for Z) and may need threshold tuning per signer.

## Project files (in `mvp/`)

| File | Purpose |
|---|---|
| `landmark_utils.py` | Shared feature extractor — 42 scale-normalized x,y features |
| `collect_data.py` | Records hand-landmark samples per letter from the webcam |
| `train_model.py` | Trains the Random Forest on collected data |
| `alphabet_landmarks.py` | Real-time recognition app — static mode + J/Z motion mode |
| `motion_jz.py` | Standalone J/Z trajectory visualizer and classifier (dev tool) |
| `evaluate_live.py` | Prompted live evaluation; saves a confusion matrix image |
| `fast_mvp.py` | Earlier MediaPipe gesture demo (kept for reference) |

## How to run

1. Activate the virtual environment (Windows): from the `mvp/` folder, run
   `..\venv\Scripts\activate`
2. **Recognition:** `python alphabet_landmarks.py` (press `q` to quit)
   - Default: STATIC mode for A–Y.
   - Press `m` to toggle MOTION mode for J and Z; SPACE starts/stops a recording.
3. **Rebuild the model:** `python collect_data.py` then `python train_model.py`
4. **Measure accuracy:** `python evaluate_live.py` (produces `confusion_matrix.png`)

## Roadmap

- **Possible:** angle-based features to better separate D/I/B/R/M; assembling recognized
  letters into words; integration into a video-call plugin.
