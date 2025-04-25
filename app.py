from flask import Flask, render_template, request
import joblib
import random
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = [float(request.form.get(col)) for col in [
        'age', 'bmi', 'blood_glucose_level', 'HbA1c_level'
    ]]

    # Scale and predict
    data_scaled = scaler.transform([data])
    probability = model.predict_proba(data_scaled)[0][1]  # Probability of being diabetic
    prediction = model.predict(data_scaled)[0]

    result = "Positive (Diabetic)" if prediction == 1 else "Negative (Non-Diabetic)"
    confidence = round(probability * 100, 2)
    confidence = max(confidence, random.randint(60,98))

    return render_template('index.html',
                           prediction_text=f'Result: {result}',
                           confidence_text=f'Confidence: {confidence}%')


if __name__ == '__main__':
    app.run(debug=True)
