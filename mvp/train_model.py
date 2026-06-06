import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

try:
    with open('landmark_data.pickle', 'rb') as f:
        data = pickle.load(f)
except FileNotFoundError:
    print('NO DATA - run collect_data.py first')
    raise SystemExit

X = np.array(data['features'])
y = data['labels']

print(f'X.shape: {X.shape}')
counts = Counter(y)
for label, count in sorted(counts.items()):
    print(f'  {label}: {count} samples')

if len(X) == 0:
    print('Empty dataset, stopping.')
    raise SystemExit

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f'Test accuracy: {acc * 100:.1f}%')

print('NOTE: high accuracy on 3 distinct letters is expected and does NOT prove it generalizes — real test is the live camera.')

with open('landmark_model.pickle', 'wb') as f:
    pickle.dump(clf, f)

print('MODEL SAVED')
