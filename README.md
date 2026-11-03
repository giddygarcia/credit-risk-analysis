# 💴 Credit Risk Modeling: Loan Default Prediction

## Overview

This project focuses on developing a binary classification model to predict whether a customer will default on their existing loan, using a dataset of approximately 32,000 loan customers.

The goal is to build a reliable credit risk model that can distinguish between defaulting and non-defaulting borrowers, identify key factors associated with loan defaults, and help financial institutions prioritize customers who may require early intervention.

Model interpretability is explored using SHAP, while predicted default probabilities are used to rank customers by their estimated risk.

[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data) is taken from Kaggle and has 32k+ rows of data, 12 features.

### 🎯 Objectives
- Perform exploratory data analysis (EDA) and data quality checks to understand the data
- Train a classification model to predict whether a client will default on an existing loan.
  - Evaluate and compare different classification models.
  - Perform hyperparameter tuning to optimize model performance.
- Identify and interpret the variables that contribute most to loan default risk using SHAP.
- Identify and rank the top-N customers at highest risk of defaulting to help the bank prioritize high-risk accounts and support early intervention strategies.

### 📌 Key Findings
* 3 boosted models, LightGBM, XGBoost & CatBoost, performed very similarly across all metrics, the final model selection will primarily depend on the business objective.
* LightGBM was selected for its highest ROC-AUC score, competitive recall, and strong overall balance between precision and recall, while being a lightweight yet well-performing model 
* The final LightGBM model achieved a ROC-AUC of 0.95 on the held-out test set, demonstrating strong classification performance and good generalization.
    * Its test performance is close to validation results, the model generalizes well to unseen data without substantial performance degradation.
* Model Explainability - top predictive features:
    Top Predictive Features for loan defaulting:
    1. Income
    2. Loan-to-income ratio
    3. Loan intent
    4. Loan grade
    5. Ownership type
    6. Interest rate
    * **Overall: Income, loan-to-income ratio, loan grade, and interest rate appear to be the most influential predictive features.** 

* Highest risk customers: tend to exhibit a combination of low income, high loan-to-income ratios, and rental ownership.

#### 💡 Expected Outcomes

- A classification model capable of identifying customers at risk of loan default.
- Insights into the financial and demographic factors associated with default risk.
- A ranked list of high-risk customers to support proactive account monitoring and early intervention.
- A more data-driven approach to credit risk assessment and management.

#### 🛣️ Feature Roadmap / Future Work
- [ ] **Productionization & Deployment** – Develop a production-ready repository, package the final model, and implement a deployment pipeline for live predictions.
- [ ] **Threshold Tuning** – Explore different classification thresholds to optimize the precision-recall trade-off.
- [ ] **Specialized Hyperparameter Tuning** – Further refine model-specific search spaces and optimization strategies to explore potential performance improvements.
- [ ] **Metric-Specific Optimization** – Experiment with optimizing specifically for recall or other evaluation metrics based on business objectives.
- [ ] **Credit Risk Grading** – Develop risk categories (e.g., Low, Medium, High) based on predicted default probabilities.


## 🛠️ Technologies & Libraries

- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, LightGBM, XGBoost, CatBoost, Imbalanced-learn
- **Model Interpretability:** SHAP
- **Environment & Dependency Management:** uv

### 🔎 Viewing / Installation

- *Viewing Option:* For the complete analysis and demonstration, simply view the notebook file `CreditRiskModeling.ipynb`.

- *Full Installation Option*: To develop on all the code firsthand, clone this repo

     ```bash
     git clone https://github.com/giddygarcia/credit-risk-analysis.git
     cd credit-risk
     uv sync
     ```

## ✉️ Author and Contact Information
Developed by: Christine Garcia 

Have questions? Feel free to:
* email me at cavgarcia22@gmail.com 
* connect on [LinkedIn](www.linkedin.com/in/cavgarcia) 
* or [view more projects](https://github.com/giddygarcia) that I enjoyed making