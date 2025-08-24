🌲 Random Forest Classifier on the Iris Dataset
📌 Overview

This project demonstrates how to use a Random Forest Classifier (from Scikit-Learn) to classify flowers from the famous Iris dataset.

The Iris dataset contains measurements of flower petals and sepals for three different Iris species. Using these measurements, the goal is to correctly predict the flower’s species.

Random Forest is a powerful ensemble machine learning algorithm that combines multiple decision trees to improve accuracy and prevent overfitting.

⚙️ Project Workflow
1. Libraries Used
from sklearn.ensemble import RandomForestClassifier   # Random Forest Model
from sklearn.model_selection import train_test_split   # Train/Test split
from sklearn.datasets import load_iris                 # Iris dataset
from sklearn.metrics import accuracy_score             # Model evaluation

2. Dataset

We use the Iris dataset built into Scikit-Learn.

Features (X):

sepal length (cm)

sepal width (cm)

petal length (cm)

petal width (cm)

Target (y):

Flower species:

0 = Setosa

1 = Versicolor

2 = Virginica

iris = load_iris()
X, y = iris.data, iris.target

3. Train/Test Split

We divide the dataset into training (80%) and testing (20%) so the model can learn on one set and be evaluated on unseen data.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


test_size=0.2 → 20% of data for testing.

random_state=42 → ensures reproducibility (same random split every time).

4. Random Forest Classifier

We train a Random Forest with 100 decision trees.

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


n_estimators=100 → the number of decision trees. More trees = better accuracy (but slower).

Each tree is trained on a random sample of the data and a random subset of features → this reduces overfitting.

5. Predictions & Evaluation

After training, the model makes predictions on the test data:

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


✅ Accuracy shows how well the model performs on unseen data.

6. Feature Importance

Random Forests can tell us which features are most important in making decisions.

for i, importance in enumerate(rf.feature_importances_):
    print(f"{iris.feature_names[i]}: {importance:.3f}")


This helps us understand which measurements (like petal length) are more useful for classification.

7. Predictions vs Actual

We also compare some predictions with actual labels:

print(f"Predicted: {y_pred[:10]}")
print(f"Actual:    {y_test[:10]}")

🌳 What is Random Forest?

A Decision Tree splits data into branches based on conditions (like a flowchart).

Problem: A single decision tree can easily overfit (memorize the training data).

Random Forest = Many Decision Trees

Each tree sees a random sample of data and features.

The forest makes predictions by majority vote from all trees.

🔑 Advantages of Random Forest:

Handles large datasets well.

Reduces overfitting (better generalization).

Provides feature importance.

Works for both classification & regression.

📊 Example Output
Accuracy: 1.00

Feature Importance:
sepal length (cm): 0.095
sepal width (cm): 0.025
petal length (cm): 0.430
petal width (cm): 0.450

Sample predictions vs actual:
Predicted: [1 0 2 1 1 0 2 1 2 0]
Actual:    [1 0 2 1 1 0 2 1 2 0]

🚀 How to Run

Clone this repo:

git clone <repo-url>
cd ML-Foundations


Install dependencies (inside your environment):

pip install scikit-learn pandas matplotlib


Run the script:

python randomForest.py

✅ Key Takeaways

Random Forest uses multiple decision trees for stronger, more reliable predictions.

It prevents overfitting and improves accuracy compared to a single tree.

Feature importance helps us interpret the model.