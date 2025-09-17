# Customer-Churn-Prediction-Project
Customer churn prediction using ML with feature engineering, class imbalance handling, hyperparameter tuning, and model interpretability (SHAP). Compared multiple models including Gradient Boosting and XGBoost.

Project Overview

Customer churn is one of the most critical challenges for businesses. Identifying customers at risk of leaving helps companies take proactive actions and improve retention.

This project builds a machine learning pipeline to predict customer churn using the Churn_Modelling dataset
.

The project covers the full lifecycle:

Exploratory Data Analysis (EDA)

Feature Engineering

Multiple ML Models & Hyperparameter Tuning

Handling Class Imbalance

Model Evaluation & Comparison

Interpretability with SHAP

Dataset

Source: Kaggle (Churn Modelling dataset)

Size: 10,000 customers

Features:

Demographic: Age, Gender, Geography

Banking: CreditScore, Balance, EstimatedSalary

Account: Tenure, NumOfProducts, HasCrCard, IsActiveMember

Target: Exited → 1 = churned, 0 = retained

Workflow
1. Exploratory Data Analysis (EDA)

Checked missing values & duplicates

Visualized churn distribution

Plotted churn by gender and geography

Bar charts for churn rates across age groups, tenure, and product holdings

2. Feature Engineering

BalanceToSalaryRatio = Balance ÷ EstimatedSalary

Grouped categorical features (AgeGroup, TenureGroup)

One-Hot encoding for Gender and Geography

3. Modeling

Tested multiple classifiers:

Logistic Regression

Random Forest

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Gradient Boosting

XGBoost

4. Hyperparameter Tuning

GridSearchCV and RandomizedSearchCV

Focused on F1-score (due to class imbalance)

5. Handling Imbalance

SMOTE oversampling inside pipelines

class_weight / scale_pos_weight for Gradient Boosting & XGBoost

6. Model Evaluation

Metrics used:

Accuracy

Precision, Recall, F1 (especially for churn class = 1)

ROC-AUC & PR-AUC

Created comparison tables for all models.

7. Model Interpretability

Feature importance from tree models

SHAP analysis:

Global feature importance (summary plot, bar plot)

Local explanations for individual customers

Results

Best Performing Models: Gradient Boosting & XGBoost

Accuracy: ~86%

F1 (Churn class): ~0.59 (improved after imbalance handling)

Recall (Churn class): ~0.50 → model captures half of churners

Key drivers of churn: Age, Balance, CreditScore, Tenure, Number of Products

Key Insights

Older customers with fewer products and lower tenure are more likely to churn.

Higher balance and higher salary reduce churn risk.

Customers in some geographies (e.g., Germany) show higher churn tendencies.

Tech Stack

Python

Pandas, NumPy

Scikit-learn

XGBoost

Imbalanced-learn (SMOTE)

Matplotlib, Seaborn

SHAP

✨ Conclusion

This project demonstrates the end-to-end process of solving a churn prediction problem: from EDA and feature engineering to model training, hyperparameter tuning, handling imbalance, and model explainability.

The workflow can be extended for real business applications by integrating deployment pipelines and cost-sensitive evaluation metrics.
Deploy pipeline using Streamlit or Flask for interactive churn predictions.
