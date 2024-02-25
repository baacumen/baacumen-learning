import pickle
import numpy as np

# Load the saved model
with open('model/iris_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Input your own data for prediction
sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Replace this with your own input data

# Make predictions using the loaded model
prediction = loaded_model.predict(sample_data)

# Map the predicted class to the corresponding species
species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
predicted_species = species_mapping[prediction[0]]

print(f"{predicted_species}")