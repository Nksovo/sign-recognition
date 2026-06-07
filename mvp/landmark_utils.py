import numpy as np


def normalize_xy(xs, ys):
    """xs, ys are parallel lists of length 21 (MediaPipe image-space coords).
    Returns 42 scale-normalized floats, x and y only (z dropped).
    1. Shift so the hand's min corner is at origin.
    2. Scale = max over all shifted x and y values (guard zero with 1e-6).
    3. Interleave: sx_i/scale, sy_i/scale for each of 21 landmarks.
    """
    min_x = min(xs)
    min_y = min(ys)
    sx = [x - min_x for x in xs]
    sy = [y - min_y for y in ys]
    scale = max(sx + sy)
    if scale == 0:
        scale = 1e-6
    out = []
    for i in range(21):
        out.append(sx[i] / scale)
        out.append(sy[i] / scale)
    return out


def extract_features(hand_landmarks):
    """hand_landmarks.landmark is 21 points with .x, .y (z ignored).
    Returns 42 scale-normalized floats.
    """
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    return normalize_xy(xs, ys)


if __name__ == '__main__':
    class _FakeLM:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.z = 999.0  # should be ignored

    class _FakeHand:
        def __init__(self):
            self.landmark = [_FakeLM(i * 0.03, i * 0.05) for i in range(21)]

    result = extract_features(_FakeHand())
    assert len(result) == 42, f"Expected 42, got {len(result)}"
    assert result[0] == 0.0 and result[1] == 0.0, "First point should be at origin after shift"
    print('SELF-TEST OK: 42 features (scale-normalized x,y, no z)')
