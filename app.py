from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load pre-trained model and scaler
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model and scaler: {e}")
    model = None
    scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded. Please check if model files exist.'})
            
        # Get values from the form
        age = float(request.form['age'])
        cp = float(request.form['cp'])
        thalach = float(request.form['thalach'])
        
        # Input validation
        if not (0 <= age <= 120):
            return jsonify({'error': 'Age must be between 0 and 120'})
        if not (0 <= cp <= 3):
            return jsonify({'error': 'Chest pain type must be between 0 and 3'})
        if not (50 <= thalach <= 250):
            return jsonify({'error': 'Maximum heart rate must be between 50 and 250'})
            
        # Convert to numpy array and reshape
        features = np.array([[age, cp, thalach]])
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        probability = probabilities[1]  # Probability of heart disease
        
        # Format the message based on probability
        if prediction == 1:
            risk_level = "High" if probability > 0.7 else "Moderate to High"
            message = f"{risk_level} risk of heart disease"
        else:
            risk_level = "Low" if probability < 0.3 else "Low to Moderate"
            message = f"{risk_level} risk of heart disease"
        
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'message': message,
            'success': True
        }
        
        return jsonify(result)
    
    except ValueError as ve:
        return jsonify({
            'error': 'Please enter valid numeric values',
            'success': False
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 