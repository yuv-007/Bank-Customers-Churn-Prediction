# ChurnAnalytics

A machine learning project for predicting credit card customer churn using behavioral and transaction data.

## Overview

Customer churn is a major concern for banks because acquiring new customers is often more expensive than retaining existing ones. In this project, I built a machine learning pipeline to identify customers who are likely to churn and analyzed the factors that influence customer attrition.

The project covers the complete data science workflow including data cleaning, exploratory data analysis, feature engineering, model training, evaluation, threshold optimization, and model explainability using SHAP.

## Dataset

The project uses the **BankChurners (Credit Card Customers)** dataset available on Kaggle.

**Note:** The dataset is not included in this repository.

To run this project:

1. Download the BankChurners dataset from Kaggle.
2. Place the CSV file in the project directory.
3. Update the dataset path in the notebook if required.
4. Run the notebook from top to bottom.

Dataset file used:

```text
BankChurners.csv
```

## Project Workflow

### 1. Data Cleaning & Preprocessing

* Removed unnecessary columns
* Encoded categorical variables
* Scaled numerical features where required
* Created train-test split

### 2. Exploratory Data Analysis

* Customer distribution analysis
* Churn pattern analysis
* Correlation analysis
* Business KPI dashboard

### 3. Feature Engineering

Created additional features to capture customer behavior, including:

* Average transaction value
* Customer inactivity indicators
* Relationship density
* Utilization-related features
* Other behavioral metrics

### 4. Model Training

The following models were trained and compared:

* Logistic Regression
* XGBoost
* LightGBM

### 5. Model Evaluation

Models were evaluated using:

* ROC-AUC
* PR-AUC
* Precision
* Recall
* F1 Score
* Confusion Matrix

### 6. Threshold Optimization

Instead of using the default threshold of 0.5, different thresholds were tested to improve the balance between precision and recall.

### 7. Explainable AI (SHAP)

To understand model predictions, SHAP was used for:

* Global feature importance
* Customer-level explanations
* High-risk customer analysis
* Low-risk customer analysis

## Results

### Best Model: XGBoost

| Metric    | Score |
| --------- | ----- |
| ROC-AUC   | 0.992 |
| PR-AUC    | 0.965 |
| Precision | 0.943 |
| Recall    | 0.868 |
| F1 Score  | 0.904 |

## Key Insights

* Transaction count was the strongest predictor of churn.
* Customers with lower transaction activity were more likely to leave.
* Inactivity and declining engagement were strong warning signs.
* SHAP analysis helped explain both global model behavior and individual customer predictions.

## Project Outputs

The repository includes:

* KPI Dashboard
* Behavioral EDA Visualizations
* Correlation Analysis
* Model Comparison Results
* Threshold Optimization Analysis
* SHAP Summary Plot
* SHAP Feature Importance Plot
* Individual Customer Explanations
* Confusion Matrix
* Saved Model Files

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost
* LightGBM
* SHAP
* Joblib

## Repository Structure

```
ChurnAnalytics/
│
├── ChurnAnalytics.ipynb
├── README.md
│
├── outputs/
│   ├── KPI Dashboard
│   ├── EDA Plots
│   ├── SHAP Visualizations
│   └── Confusion Matrix
│
└── model/
    ├── churn_model.pkl
    ├── scaler.pkl
    ├── encoders.pkl
    ├── feature_columns.pkl
    └── optimal_threshold.pkl
```

## Future Work

* Deploy the model using Streamlit
* Add real-time churn prediction
* Create a simple user interface for customer scoring

## Author

**Yuvraj Gupta**
B.Tech, IIT (BHU) Varanasi

## Streamlit App (Interactive Deployment)

An interactive Streamlit dashboard is included to run the saved model on user-provided inputs or CSV files. The app expects the following pickle files to be present in the project root or `model/` folder:

- `churn_model.pkl`
- `feature_columns.pkl`
- `encoders.pkl`
- `scaler.pkl` (optional)
- `optimal_threshold.pkl` (optional)

Run the dashboard locally with:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Notes:

- If `feature_columns.pkl` or `encoders.pkl` are missing the app will fall back to a reasonable default feature set and simple inputs — however predictions will be most accurate when the saved artifacts from the notebook are present.
- Place the pickle files in the `model/` directory or project root so the app can find them automatically.

