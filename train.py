import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the dataset from the pickle file (a dictionary containing 'data' and 'labels')
with open('/Users/samuel/gesture detector/data.pickle', 'rb') as f:
    dataset = pickle.load(f)

# Extract data and labels from the dictionary
data = dataset['data']  # Assume this is a list of lists, but needs validation
labels = dataset['labels']  # Assume this is a list of strings or labels

# Ensure all data samples have the same length by padding or truncating
# Find the length of the largest sample (maximum number of landmarks)
max_length = max(len(item) for item in data)

# Pad each data point with zeros to ensure all samples have the same length
# This uses NumPy's pad function to ensure uniform length across samples
data_padded = [np.pad(item, (0, max_length - len(item)), 'constant') for item in data]

# Convert data to a NumPy array for easier processing
data_padded = np.array(data_padded)

# Encode labels into numerical format (if they are strings)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the data into training and test sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(data_padded, labels_encoded, test_size=0.2, stratify=labels_encoded)

# Initialize and train the RandomForest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Calculate accuracy of the predictions
score = accuracy_score(y_test, y_predict)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model to a pickle file for future use
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

# Optional: Print the first few predictions and their corresponding true labels
print("First 10 predictions:", y_predict[:10])
print("First 10 true labels:", y_test[:10])

# # Save the model to a pickle file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
