import numpy as np


def extract_features(hand_landmarks):
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    min_x = min(xs)
    min_y = min(ys)
    out = []
    for x, y in zip(xs, ys):
        out.append(x - min_x)
        out.append(y - min_y)
    return out
