# ML_PROJECT-Heart-Diseases-
# Heart Disease Prediction Project with Chatbot

## Overview
This project leverages machine learning techniques to predict the likelihood of heart disease based on various health parameters. The goal is to provide a tool that assists in early diagnosis and better healthcare management.This project includes a chatbot integrated, allowing users to interact with an AI-powered assistant directly on the site. The chatbot can answer queries, guide users, and improve overall user experience.

## Features
- Predicts the presence or absence of heart disease.
- Uses a dataset with features such as age, blood pressure, cholesterol levels, and more.
- Employs machine learning algorithms like Logistic Regression, Random Forest, or Neural Networks.

## Dataset
The dataset used for this project is sourced from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease) or other publicly available repositories. It contains the following key features:

- **Age**: Age of the individual.
- **Sex**: Gender (1 = male; 0 = female).
- **Chest Pain Type**: Types of chest pain experienced (4 categories).
- **Resting Blood Pressure**: Blood pressure in mmHg.
- **Cholesterol**: Serum cholesterol in mg/dl.
- **Fasting Blood Sugar**: Whether fasting blood sugar > 120 mg/dl.
- **Rest ECG**: Resting electrocardiographic results (0, 1, 2).
- **Maximum Heart Rate Achieved**: Maximum heart rate achieved.
- **Exercise-Induced Angina**: Induced angina (1 = yes; 0 = no).
- **ST Depression**: Depression induced by exercise.
- **Slope of Peak Exercise ST Segment**: (0, 1, 2).
- **Number of Major Vessels**: (0-3).
- **Thalassemia**: (3 = normal; 6 = fixed defect; 7 = reversible defect).

## Installation

To run the project locally:

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/heart-disease-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd heart-disease-prediction
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure the dataset is placed in the appropriate folder (e.g., `data/heart_disease.csv`).
2. Run the training script:
   ```bash
   python train.py
   ```
3. To test the model, use:
   ```bash
   python predict.py --input sample_input.json
   ```

## Key Steps from Notebook
- **Data Preprocessing**: Null value handling, feature scaling, and encoding categorical variables.
- **Model Training**: Algorithms like Logistic Regression and Random Forest were trained and evaluated.
- **Evaluation**:
  - Confusion Matrix: Visualized to evaluate true/false positives and negatives.
  - ROC Curve: Demonstrates model performance.

### Example Code Snippet
Below is an example of model training from the notebook:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

## Results
The model achieves the following performance metrics on the test set:
- **Accuracy**: 80%
- **Precision**: 0.77 (No Disease), 0.83 (Disease)
- **Recall**: 0.83 (No Disease), 0.78 (Disease)
- **F1 Score**: 0.80

## Key Files
- `train.py`: Script for training the model.
- `predict.py`: Script for making predictions.
- `data/`: Directory containing the dataset.
- `models/`: Saved models.
- `notebooks/`: Jupyter notebooks for EDA and model development.

## Future Work
- Integration with a web application for user-friendly predictions.
- Use of more advanced algorithms like Gradient Boosting or Deep Learning.
- Incorporate real-time data collection using IoT devices.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease) for the dataset.
- The open-source community for libraries and tools.
# Flask Web Application

This is a Flask web application that can be deployed on Render.

## Local Development

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
venv\Scripts\activate
```
- Unix/MacOS:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

## Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure the service:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
4. Deploy!

## Environment Variables

The following environment variables can be configured in Render:
- `FLASK_ENV`: Set to 'production' for deployment
- `FLASK_APP`: Set to 'app.py' 
