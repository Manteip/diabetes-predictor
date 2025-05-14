from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

# Load the pre-trained models
try:
    with open('naive_bayes_model.pkl', 'rb') as f:
        naive_bayes_model = pickle.load(f)
    with open('perceptron_model.pkl', 'rb') as f:
        perceptron_model = pickle.load(f)
    print("Models loaded successfully.")
except FileNotFoundError:
    print("Model files not found. Please ensure naive_bayes_model.pkl and perceptron_model.pkl are in the same directory.")

# Route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_type = data.get('model_type', 'naive_bayes')
    input_features = np.array([[data['age'], data['glucose'], data['insulin'], data['bmi']]])

    # Select and apply the chosen model
    if model_type == 'naive_bayes':
        prediction = naive_bayes_model.predict(input_features)
    elif model_type == 'perceptron':
        prediction = perceptron_model.predict(input_features)
    else:
        return jsonify({'error': 'Invalid model type'}), 400

    return jsonify({'diabetes_type': int(prediction[0])})

# Start the Flask app
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render sets this
    app.run(host="0.0.0.0", port=port, debug=True)





