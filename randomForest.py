# Libraries to Import
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make Predictions
y_pred = rf.predict(X_test)

# Evaluate Model    
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

#Display feature importance
print("\nFeature Importance:")
feature_names = iris.feature_names
for i, importance in enumerate(rf.feature_importances_):
    print(f"{feature_names[i]}: {importance:.3f}")

#Display some predictions vs actual
print(f"\nSample predictions vs actual:")
print(f"Predicted: {y_pred[:10]}")
print(f"Actual:    {y_test[:10]}")