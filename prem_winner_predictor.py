# Project: Premier League Winner Predictor Model (AI Powered with TensorFlow)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the uploaded dataset
data = pd.read_csv('Data.csv')

# Preprocess categorical features
data['Avg Rating'] = data['Avg Rating'].map({'Very High': 4, 'High': 3, 'Good': 2, 'Moderate': 1, 'Low': 0})
data['Injuries'] = data['Injuries'].map({'Low': 0, 'Moderate': 1, 'High': 2, 'Very High': 3})
data['2023–24 Pos.'] = data['2023–24 Pos.'].str.replace('st|nd|rd|th', '', regex=True).astype(int)

# Features and target
X = data[['Rank', 'Avg Goals/Game', 'Market Value (€M)', 'Avg Rating', 'Injuries', '2023–24 Pos.']]
y = data['Winner']

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Build TensorFlow model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2)

# Predict using the current season's data
X_scaled_full = scaler.transform(X)
predictions = model.predict(X_scaled_full)

# Get predicted classes
predicted_classes = np.argmax(predictions, axis=1)

# Attach predictions to the original data
data['Predicted_Winner'] = label_encoder.inverse_transform(predicted_classes)

# Find the teams predicted as winners (Winner = 1)
predicted_winner_teams = data[data['Predicted_Winner'] == 1]['Team'].tolist()

print("Predicted Winning Teams:", predicted_winner_teams)
