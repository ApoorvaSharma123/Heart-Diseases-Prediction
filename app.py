from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Create a folder for uploaded files

# Create uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load data and train model
try:
    # Load data
    df = pd.read_csv('heart_disease.csv')
    
    # Feature and target selection
    X = df[['age', 'cp', 'thalach']]
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    print("Model trained successfully!")
except Exception as e:
    print(f"Error loading data or training model: {e}")
    model = None
    scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not trained. Please check the data file.'})
            
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

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file and file.filename.endswith('.csv'):
            # Save the file
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            
            # Read and return the first few rows of the CSV
            try:
                df = pd.read_csv(filename)
                preview = df.head().to_dict('records')
                return jsonify({
                    'message': 'File uploaded successfully',
                    'preview': preview
                })
            except Exception as e:
                return jsonify({'error': f'Error reading CSV: {str(e)}'})
        
        return jsonify({'error': 'Please upload a CSV file'})
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True) 