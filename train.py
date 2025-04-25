import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('diabetes_dataset_with_notes.csv')

# Convert target column to 0 and 1
df['diabetes'] = df['diabetes'].replace({'Yes': 1, 'No': 0})

# Select only the 4 features you're using in your Flask form
selected_features = ['age', 'bmi', 'blood_glucose_level', 'hbA1c_level']
X = df[selected_features]
y = df['diabetes']

# Handle missing values
X.fillna(X.median(), inplace=True)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate it
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model and scaler
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
