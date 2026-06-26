# Customer Churn Prediction using XGBoost

## Overview
This project predicts whether a telecom customer is likely to churn (cancel their subscription) based on account and service usage data. It uses XGBoost, a gradient boosting algorithm, along with data preprocessing, feature scaling, and hyperparameter tuning to build an accurate classification model.

## Dataset
- **Source:** [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- ~7,000 customer records with features like tenure, contract type, monthly charges, payment method, and internet service.
- Target variable: `Churn` (Yes/No)

## Approach
1. **Data Cleaning** — Removed non-predictive identifiers, handled missing/blank values in `TotalCharges`, encoded the target variable.
2. **Feature Engineering** — One-hot encoded categorical variables (contract type, payment method, internet service, etc.).
3. **Feature Scaling** — Standardized numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler`.
4. **Model Training** — Trained an XGBoost classifier (`XGBClassifier`) on the processed dataset.
5. **Hyperparameter Tuning** — Used `GridSearchCV` to tune `max_depth`, `n_estimators`, and `learning_rate`, optimizing for F1-score.
6. **Evaluation** — Assessed model performance using precision, recall, F1-score, and accuracy on a held-out test set.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost

## Results
| Metric | Score |
|--------|-------|
| Accuracy | XX% |
| Precision | XX% |
| Recall | XX% |
| F1-Score | XX% |

*(Replace with your actual classification_report output from Step 6)*

## How to Run
1. Clone this repo
2. Install dependencies: `pip install pandas numpy scikit-learn xgboost`
3. Download the dataset from Kaggle and place it in the project folder
4. Run the notebook: `customer_churn_xgboost.ipynb`

## Key Learnings
- Handling missing/blank numeric values disguised as strings (`TotalCharges`)
- Importance of feature scaling for gradient boosting stability
- Using GridSearchCV to systematically tune hyperparameters rather than guessing
