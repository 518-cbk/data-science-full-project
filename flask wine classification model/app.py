from flask import Flask, render_template, request, jsonify
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Load the saved model
try:
    with open('savedmodel.sav', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Warning: savedmodel.sav not found. Please ensure the model file is in the same directory.")
    model = None

# Wine feature names (as per sklearn wine dataset)
FEATURE_NAMES = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
    'color_intensity', 'hue', 'od280_od315_of_diluted_wines', 'proline'
]

# Feature descriptions for better UX
FEATURE_DESCRIPTIONS = {
    'alcohol': 'Alcohol content (%)',
    'malic_acid': 'Malic acid (g/L)',
    'ash': 'Ash content (g/L)',
    'alcalinity_of_ash': 'Alkalinity of ash (mEq/L)',
    'magnesium': 'Magnesium (mg/L)',
    'total_phenols': 'Total phenols (mg/L)',
    'flavanoids': 'Flavanoids (mg/L)',
    'nonflavanoid_phenols': 'Non-flavanoid phenols (mg/L)',
    'proanthocyanins': 'Proanthocyanins (mg/L)',
    'color_intensity': 'Color intensity (0-10 scale)',
    'hue': 'Hue (0-2 scale)',
    'od280_od315_of_diluted_wines': 'OD280/OD315 ratio',
    'proline': 'Proline (mg/L)'
}

# Typical ranges for validation
FEATURE_RANGES = {
    'alcohol': (11.0, 15.0),
    'malic_acid': (0.74, 5.80),
    'ash': (1.36, 3.23),
    'alcalinity_of_ash': (10.6, 30.0),
    'magnesium': (70, 162),
    'total_phenols': (0.98, 3.88),
    'flavanoids': (0.34, 5.08),
    'nonflavanoid_phenols': (0.13, 0.66),
    'proanthocyanins': (0.41, 3.58),
    'color_intensity': (1.28, 13.0),
    'hue': (0.48, 1.71),
    'od280_od315_of_diluted_wines': (1.27, 4.00),
    'proline': (278, 1680)
}

# Wine cultivar names
CULTIVAR_NAMES = {
    0: 'Cultivar 1 (Class 0)',
    1: 'Cultivar 2 (Class 1)', 
    2: 'Cultivar 3 (Class 2)'
}

@app.route('/')
def index():
    return render_template('index.html', 
                         features=FEATURE_NAMES,
                         descriptions=FEATURE_DESCRIPTIONS,
                         ranges=FEATURE_RANGES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check if savedmodel.sav exists.'})
        
        # Get form data
        features = []
        for feature_name in FEATURE_NAMES:
            value = request.form.get(feature_name)
            if value is None or value == '':
                return jsonify({'error': f'Missing value for {feature_name}'})
            
            try:
                features.append(float(value))
            except ValueError:
                return jsonify({'error': f'Invalid value for {feature_name}. Please enter a number.'})
        
        # Validate ranges
        validation_errors = []
        for i, (feature_name, value) in enumerate(zip(FEATURE_NAMES, features)):
            min_val, max_val = FEATURE_RANGES[feature_name]
            if value < min_val * 0.5 or value > max_val * 2:  # Allow some flexibility
                validation_errors.append(f'{feature_name}: {value} seems outside typical range ({min_val}-{max_val})')
        
        # Create input array
        input_data = np.array([features])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Get prediction probabilities if available
        try:
            probabilities = model.predict_proba(input_data)[0]
            confidence = max(probabilities) * 100
            prob_dict = {f'Cultivar {i}': prob * 100 for i, prob in enumerate(probabilities)}
        except:
            confidence = 85.0  # Default confidence if probabilities not available
            prob_dict = {}
        
        result = {
            'prediction': int(prediction),
            'cultivar_name': CULTIVAR_NAMES[prediction],
            'confidence': round(confidence, 1),
            'probabilities': prob_dict,
            'validation_warnings': validation_errors
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})



# Load the CSV file once at startup
df = pd.read_csv('wine.csv')
@app.route('/example_data')

def example_data():
    try:
        # Number of samples to return
        num_samples = 3

        # Sample rows from DataFrame
        sampled_df = df.sample(n=num_samples).reset_index(drop=True)

        examples = {}
        for idx, row in sampled_df.iterrows():
            example = {
                'alcohol': row['alcohol'],
                'malic_acid': row['malic_acid'],
                'ash': row['ash'],
                'alcalinity_of_ash': row['alcalinity_of_ash'],
                'magnesium': row['magnesium'],
                'total_phenols': row['total_phenols'],
                'flavanoids': row['flavanoids'],
                'nonflavanoid_phenols': row['nonflavanoid_phenols'],
                'proanthocyanins': row['proanthocyanins'],
                'color_intensity': row['color_intensity'],
                'hue': row['hue'],
                'od280_od315_of_diluted_wines': row['od280_od315_of_diluted_wines'],
                'proline': row['proline']
            }
            examples[f'cultivar_{idx}'] = example

        return jsonify(examples)
    except Exception as e:
        # Return error message for debugging
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)