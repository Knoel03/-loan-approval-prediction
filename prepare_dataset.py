import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Load the original dataset
file_path = "data/raw/default of credit card clients.xls"

df = pd.read_excel(file_path, header=1)

print("Original dataset shape:", df.shape)

print("\nFirst 5 records:")
print(df.head())

# Separate target from input features

target_column = "default payment next month"

X = df.drop(columns=[target_column])
y = df[target_column]

print("\nFEATURES SHAPE:", X.shape)
print("TARGET SHAPE:", y.shape)

print("\nTARGET DISTRIBUTION:")
print(y.value_counts())

# Remove customer ID from input features

X = X.drop(columns=["ID"])

print("\nFEATURES AFTER REMOVING ID:")
print(X.shape)

print("\nREMAINING FEATURES:")
print(X.columns.tolist())

# Remove demographic features from the initial model

demographic_columns = [
    "SEX",
    "AGE",
    "EDUCATION",
    "MARRIAGE"
]

X = X.drop(columns=demographic_columns)

print("\nFEATURES AFTER REMOVING DEMOGRAPHIC COLUMNS:")
print(X.shape)

print("\nREMAINING FEATURES:")
print(X.columns.tolist())

# Identify repayment status columns

categorical_columns = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6"
]

print("\nREPAYMENT STATUS DATA TYPES:")
print(X[categorical_columns].dtypes)

print("\nUNIQUE REPAYMENT STATUS VALUES:")
for column in categorical_columns:
    print(f"{column}: {sorted(X[column].unique())}")

# Convert repayment status columns to categorical data

for column in categorical_columns:
    X[column] = X[column].astype("category")

print("\nUPDATED DATA TYPES:")
print(X[categorical_columns].dtypes)

# Identify numerical features

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nNUMERICAL FEATURES:")
print(numerical_columns)

print("\nMISSING VALUES IN NUMERICAL FEATURES:")
print(X[numerical_columns].isnull().sum())

print("\nNUMERICAL FEATURE SUMMARY:")
print(X[numerical_columns].describe().round(2))

# Detect potential outliers using the IQR method

print("\nPOTENTIAL OUTLIER COUNTS")

for column in numerical_columns:
    Q1 = X[column].quantile(0.25)
    Q3 = X[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    outliers = X[
        (X[column] < lower_bound) |
        (X[column] > upper_bound)
    ]

    print(f"{column}: {len(outliers)}")

    # Validate target variable

print("\nTARGET VALIDATION")
print("Unique target values:", sorted(y.unique()))
print("Missing target values:", y.isnull().sum())

print("\nTARGET DATA TYPE:")
print(y.dtype)

print("\nTARGET CLASS DISTRIBUTION:")
print(y.value_counts(normalize=True).mul(100).round(2))

from sklearn.model_selection import train_test_split

# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDATASET SPLIT")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTRAINING TARGET DISTRIBUTION:")
print(y_train.value_counts(normalize=True).mul(100).round(2))

print("\nTESTING TARGET DISTRIBUTION:")
print(y_test.value_counts(normalize=True).mul(100).round(2))

# Define preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", RobustScaler(), numerical_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
)

print("\nPREPROCESSING PIPELINE CREATED")
print("Numerical features:", len(numerical_columns))
print("Categorical features:", len(categorical_columns))

# Fit preprocessing on training data and transform it
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using the training-fitted preprocessor
X_test_processed = preprocessor.transform(X_test)

print("\nPREPROCESSED DATA")
print("Training data shape:", X_train_processed.shape)
print("Testing data shape:", X_test_processed.shape)

# Establish a baseline using the majority class
baseline_model = DummyClassifier(strategy="most_frequent")

baseline_model.fit(X_train_processed, y_train)

# Make predictions on test data
baseline_predictions = baseline_model.predict(X_test_processed)

# Evaluate baseline
baseline_accuracy = accuracy_score(y_test, baseline_predictions)

print("\nBASELINE MODEL RESULTS")
print("Accuracy:", round(baseline_accuracy * 100, 2), "%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, baseline_predictions, zero_division=0))

# Train Logistic Regression model
logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

logistic_model.fit(X_train_processed, y_train)

# Make predictions
logistic_predictions = logistic_model.predict(X_test_processed)

# Evaluate model
logistic_accuracy = accuracy_score(y_test, logistic_predictions)

print("\nLOGISTIC REGRESSION RESULTS")
print("Accuracy:", round(logistic_accuracy * 100, 2), "%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, logistic_predictions, zero_division=0))

# Generate confusion matrix
cm = confusion_matrix(y_test, logistic_predictions)

print("\nLOGISTIC REGRESSION CONFUSION MATRIX")
print(cm)

# Display confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Default", "Default"]
)

display.plot()
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

# Evaluate probability-based model performance
logistic_probabilities = logistic_model.predict_proba(X_test_processed)[:, 1]

roc_auc = roc_auc_score(y_test, logistic_probabilities)
pr_auc = average_precision_score(y_test, logistic_probabilities)

print("\nLOGISTIC REGRESSION ADVANCED METRICS")
print("ROC-AUC:", round(roc_auc, 4))
print("PR-AUC:", round(pr_auc, 4))

# Train Decision Tree model
tree_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=50,
    class_weight="balanced",
    random_state=42
)

tree_model.fit(X_train_processed, y_train)

# Make predictions
tree_predictions = tree_model.predict(X_test_processed)

# Evaluate model
tree_accuracy = accuracy_score(y_test, tree_predictions)

print("\nDECISION TREE RESULTS")
print("Accuracy:", round(tree_accuracy * 100, 2), "%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, tree_predictions, zero_division=0))

# Calculate advanced metrics
tree_probabilities = tree_model.predict_proba(X_test_processed)[:, 1]

tree_roc_auc = roc_auc_score(y_test, tree_probabilities)
tree_pr_auc = average_precision_score(y_test, tree_probabilities)

print("\nDECISION TREE ADVANCED METRICS")
print("ROC-AUC:", round(tree_roc_auc, 4))
print("Average Precision:", round(tree_pr_auc, 4))

# Train Random Forest model
forest_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

forest_model.fit(X_train_processed, y_train)

# Make predictions
forest_predictions = forest_model.predict(X_test_processed)

# Evaluate model
forest_accuracy = accuracy_score(y_test, forest_predictions)

print("\nRANDOM FOREST RESULTS")
print("Accuracy:", round(forest_accuracy * 100, 2), "%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    y_test,
    forest_predictions,
    zero_division=0
))

# Calculate advanced metrics
forest_probabilities = forest_model.predict_proba(X_test_processed)[:, 1]

forest_roc_auc = roc_auc_score(y_test, forest_probabilities)
forest_pr_auc = average_precision_score(y_test, forest_probabilities)

print("\nRANDOM FOREST ADVANCED METRICS")
print("ROC-AUC:", round(forest_roc_auc, 4))
print("Average Precision:", round(forest_pr_auc, 4))

# Random Forest Feature Importance

feature_names = preprocessor.get_feature_names_out()

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": forest_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 15 MOST IMPORTANT FEATURES")
print(feature_importance.head(15).to_string(index=False))

# Visualize feature importance
top_features = feature_importance.head(15).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 7))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.title("Random Forest - Top 15 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# Check Random Forest for Overfitting

train_predictions = forest_model.predict(X_train_processed)
test_predictions = forest_model.predict(X_test_processed)

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print("\nRANDOM FOREST OVERFITTING CHECK")
print("Training Accuracy:", round(train_accuracy * 100, 2), "%")
print("Testing Accuracy:", round(test_accuracy * 100, 2), "%")
print("Accuracy Difference:", round(
    (train_accuracy - test_accuracy) * 100, 2
), "percentage points")

import joblib
import os

# Create models directory
os.makedirs("models", exist_ok=True)

# Save trained model and preprocessing pipeline
joblib.dump(forest_model, "models/random_forest_model.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")

print("\nMODEL SAVING")
print("Random Forest model saved successfully.")
print("Preprocessor saved successfully.")

# PREDICTION PROBABILITY ANALYSIS

y_prob = forest_model.predict_proba(X_test_processed)[:, 1]

print("\nPREDICTION PROBABILITY ANALYSIS")
print("Minimum probability:", y_prob.min())
print("Maximum probability:", y_prob.max())
print("Average probability:", y_prob.mean())

# THRESHOLD COMPARISON

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

thresholds = [0.30, 0.40, 0.50, 0.60]

print("\nTHRESHOLD COMPARISON")

for threshold in thresholds:
    y_pred_threshold = (y_prob >= threshold).astype(int)

    print(f"\nThreshold: {threshold}")
    print("Precision:", precision_score(y_test, y_pred_threshold))
    print("Recall:", recall_score(y_test, y_pred_threshold))
    print("F1 Score:", f1_score(y_test, y_pred_threshold))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_threshold))

    # PRECISION-RECALL VS THRESHOLD GRAPH

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

precision_values, recall_values, threshold_values = precision_recall_curve(
    y_test, y_prob
)

plt.figure(figsize=(10, 6))

plt.plot(threshold_values, precision_values[:-1], label="Precision")
plt.plot(threshold_values, recall_values[:-1], label="Recall")

plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall at Different Thresholds")
plt.legend()
plt.grid(True)

plt.savefig("threshold_analysis.png")
plt.show()

# MODEL CALIBRATION ANALYSIS

from sklearn.calibration import calibration_curve

fraction_positive, mean_predicted = calibration_curve(
    y_test,
    y_prob,
    n_bins=10,
    strategy="uniform"
)

plt.figure(figsize=(8, 6))

plt.plot(mean_predicted, fraction_positive, marker="o", label="Random Forest")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Actual Default Rate")
plt.title("Random Forest Calibration Curve")
plt.legend()
plt.grid(True)

plt.savefig("calibration_curve.png")
plt.show()

print("\nCALIBRATION ANALYSIS")
print("Mean predicted probability:", y_prob.mean())
print("Actual default rate:", y_test.mean())

# BRIER SCORE EVALUATION

from sklearn.metrics import brier_score_loss

brier_score = brier_score_loss(y_test, y_prob)

print("\nBRIER SCORE")
print("Random Forest Brier Score:", brier_score)

baseline_probability = y_train.mean()

baseline_brier = brier_score_loss(
    y_test,
    [baseline_probability] * len(y_test)
)

print("Baseline Brier Score:", baseline_brier)

# RANDOM FOREST PROBABILITY CALIBRATION

from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(
    forest_model,
    method="sigmoid",
    cv=5
)

calibrated_model.fit(X_train_processed, y_train)

calibrated_probabilities = calibrated_model.predict_proba(
    X_test_processed
)[:, 1]

calibrated_brier = brier_score_loss(
    y_test,
    calibrated_probabilities
)

print("\nCALIBRATED RANDOM FOREST")
print("Original Brier Score:", brier_score)
print("Calibrated Brier Score:", calibrated_brier)
print("Mean calibrated probability:", calibrated_probabilities.mean())

# CALIBRATED MODEL CLASSIFICATION PERFORMANCE

calibrated_predictions = calibrated_model.predict(X_test_processed)

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score
)

print("\nCALIBRATED MODEL PERFORMANCE")

print("\nClassification Report:")
print(classification_report(y_test, calibrated_predictions))

print("ROC-AUC:",
      roc_auc_score(y_test, calibrated_probabilities))

print("Average Precision:",
      average_precision_score(y_test, calibrated_probabilities))

print("Brier Score:",
      brier_score_loss(y_test, calibrated_probabilities))

# SAVE CALIBRATED MODEL

joblib.dump(
    calibrated_model,
    "models/calibrated_random_forest.pkl"
)

print("\nCalibrated model saved successfully.")