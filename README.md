
# Loan Approval Prediction & Assessment Platform

A machine learning and financial assessment platform designed to explore borrower risk, loan affordability, and repayment prediction.

## Project Overview

This project aims to develop a platform that combines transparent financial assessment calculations with machine learning to support preliminary loan assessments.

The platform is being developed with two distinct assessment components:

1. **Financial Affordability Assessment:** Evaluates income, expenses, existing debt, and estimated repayment obligations.
2. **Machine Learning Risk Assessment:** Uses historical financial data to investigate patterns associated with repayment default.

The long-term objective is to support separate assessment workflows for personal and business loan applicants.

## Current Development Stage

The current development stage focuses on data exploration, preprocessing, and machine learning experimentation.

A Taiwanese credit-card default dataset from the UCI Machine Learning Repository is being used to develop and evaluate the initial classification pipeline.

**Important:** This dataset predicts credit-card default in the following month. It does not predict Nigerian loan approval or directly measure the repayment performance of new loan applicants.

## Dataset

**Source:** UCI Machine Learning Repository

**Dataset:** Default of Credit Card Clients

- Records: 30,000
- Features: 23 explanatory variables
- Target: Default payment next month
- Target classes: Default and No Default

## Machine Learning Models

Three classification models have been evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

A Dummy Classifier was also used as a baseline.

### Current Results

| Model               | Accuracy | ROC-AUC | Average Precision |
| ------------------- | -------: | ------: | ----------------: |
| Dummy Classifier    |   77.88% |      — |                — |
| Logistic Regression |   77.75% |  0.7547 |            0.5188 |
| Decision Tree       |       — |  0.7452 |            0.4902 |
| Random Forest       |   78.03% |  0.7757 |            0.5494 |

Random Forest currently has the strongest ROC-AUC and Average Precision among the evaluated models.

### Random Forest Evaluation

- Training accuracy: 80.67%
- Testing accuracy: 78.03%
- Accuracy difference: 2.64 percentage points
- Default-class precision: 0.50
- Default-class recall: 0.59
- Default-class F1-score: 0.54

## Technologies

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Git and GitHub

## Project Structure

```text
loan-approval-prediction/
│
├── data/
│   └── raw/
│
├── models/
│   ├── random_forest_model.pkl
│   └── preprocessor.pkl
│
├── app.py
├── inspect_dataset.py
├── prepare_dataset.py
├── requirements.txt
└── README.md
```

## Limitations

- The current dataset represents Taiwanese credit-card customers, not Nigerian loan applicants.
- The target represents next-month credit-card default, not loan approval.
- Current model results are experimental and do not establish real-world lending performance.
- The model is not ready for actual lending decisions.

## Planned Development

1. Improve model evaluation and validation.
2. Develop a transparent financial affordability engine.
3. Research suitable Nigerian personal and business financial datasets.
4. Develop separate personal and business assessment workflows.
5. Build a Flask backend.
6. Implement user registration, authentication, and application history.
7. Develop applicant and administrator dashboards.
8. Test and deploy the application.

## Disclaimer

This project is an educational and experimental software engineering project. Its predictions are not financial advice, loan offers, or guarantees of loan approval.
