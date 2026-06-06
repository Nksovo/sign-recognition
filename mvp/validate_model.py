import pickle
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

try:
    with open('landmark_data.pickle', 'rb') as f:
        data = pickle.load(f)
except FileNotFoundError:
    print('NO DATA - run collect_data.py first')
    raise SystemExit

X = np.array(data['features'])
y = data['labels']

print('Per-class counts:')
for label, count in sorted(Counter(y).items()):
    print(f'  {label}: {count}')
print(f'Total: {len(y)} samples, {len(set(y))} classes\n')

clf = RandomForestClassifier(n_estimators=100, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

print('5-fold cross-validation results:')
for i, s in enumerate(scores, 1):
    print(f'  Fold {i}: {s * 100:.1f}%')
print(f'Mean:  {scores.mean() * 100:.1f}%')
print(f'Std:   {scores.std() * 100:.1f}%')
print()
print('If folds are all high AND similar, it generalizes. If train accuracy was ~100% but folds are much lower, it memorized.')
