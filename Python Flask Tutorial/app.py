# app.py
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the machine learning model
with open('model/iris_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.get_json(force=True)
    
    # Extract input features from the data
    input_data = np.array(data['input_data']).reshape(1, -1)

    # Make predictions using the loaded model
    prediction = loaded_model.predict(input_data)

    # Map the predicted class to the corresponding species
    species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    predicted_species = species_mapping[prediction[0]]

    # Return the prediction as JSON
    return jsonify({'prediction': predicted_species})

if __name__ == '__main__':
    app.run(debug=True)