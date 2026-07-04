import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# -----------------------------
# Create folders if not present
# -----------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
file_path = r"D:\heart disease system 2\new model\heart.csv"
df = pd.read_csv(file_path)


print("Dataset Shape:", df.shape)

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------
# Random Forest Model
# -----------------------------
rf = RandomForestClassifier(random_state=42)

# -----------------------------
# Hyperparameter Tuning
# -----------------------------
param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

print("Training Model...")

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters")
print(grid.best_params_)

# -----------------------------
# Prediction
# -----------------------------
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(best_model, "models/heart_model.pkl")

print("\nModel Saved Successfully!")

# -----------------------------
# Confusion Matrix
# -----------------------------
disp = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred)
)

disp.plot()

plt.savefig("images/confusion_matrix.png")
plt.close()

# -----------------------------
# ROC Curve
# -----------------------------
RocCurveDisplay.from_estimator(
    best_model,
    X_test,
    y_test
)

plt.savefig("images/roc_curve.png")
plt.close()

# -----------------------------
# Feature Importance
# -----------------------------
importance = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values()

plt.figure(figsize=(8,6))
importance.plot(kind="barh")

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("images/feature_importance.png")

plt.close()

print("\nImages Saved Successfully!")

