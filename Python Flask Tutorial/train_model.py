import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target variable (species)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model on the training set
knn.fit(X_train, y_train)

# Save the trained model to a file using pickle
with open('model/iris_model.pkl', 'wb') as model_file:
    pickle.dump(knn, model_file)

# Load the saved model
with open('model/iris_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Make predictions on the testing set using the loaded model
y_pred = loaded_model.predict(X_test)

# Evaluate the accuracy of the loaded model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of loaded model: {accuracy * 100:.2f}%")