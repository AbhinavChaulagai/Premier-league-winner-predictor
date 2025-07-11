# Project: Premier League Winner Predictor Model (AI Powered with TensorFlow)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the latest dataset
data = pd.read_csv('Data.csv')

# Preprocess categorical features
data['Avg Rating'] = data['Avg Rating'].map({'Very High': 4, 'High': 3, 'Good': 2, 'Moderate': 1, 'Low': 0})
data['Injuries'] = data['Injuries'].map({'Low': 0, 'Moderate': 1, 'High': 2, 'Very High': 3})
data['2024–25 Pos.'] = data['2024–25 Pos.'].str.replace('st|nd|rd|th', '', regex=True).astype(int)

# Define features and target
feature_columns = ['Avg Goals/Game', 'Market Value (€M)', 'Avg Rating', 'Injuries', '2024–25 Pos.']
X = data[feature_columns]
y = data['2024–25 Pos.']  # Predicting the final position (lower is better)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build TensorFlow regression model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Regression output
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='mean_squared_error',
              metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=1)

# Predict the final positions for the full dataset
X_scaled_full = scaler.transform(X)
predicted_positions = model.predict(X_scaled_full).flatten()

data['Predicted_Pos'] = predicted_positions

# Get the team with the lowest predicted final position (1st place)
predicted_winner_team = data.loc[data['Predicted_Pos'].idxmin(), 'Team']

print("Predicted Winner Team:", predicted_winner_team)

# Optional: print the full predicted table
# print(data[['Team', 'Predicted_Pos']].sort_values(by='Predicted_Pos'))
