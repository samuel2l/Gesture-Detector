import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

with open('/Users/samuel/gesture detector/data.pickle', 'rb') as f:
    dataset = pickle.load(f)

data = dataset['data']
labels = dataset['labels']
max_length = max(len(item) for item in data)
data_padded = [np.pad(item, (0, max_length - len(item)), 'constant') for item in data]
data_padded = np.array(data_padded)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

x_train, x_test, y_train, y_test = train_test_split(data_padded, labels_encoded, test_size=0.2, stratify=labels_encoded)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)


score = accuracy_score(y_test, y_predict)
print(f'{score * 100}% of samples were classified correctly!')

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("First 10 predictions:", y_predict[:10])
print("First 10 true labels:", y_test[:10])

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
