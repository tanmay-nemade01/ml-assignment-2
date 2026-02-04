# Loan Approval Prediction - ML Assignment 2

**Live App:** https://2025aa05004-ml-assignment-2.streamlit.app/

---

## 1. Problem Statement

The objective of this project is to build a machine learning classification system that predicts whether a loan application will be **approved (1)** or **rejected (0)** based on various applicant and loan features. This is a **binary classification problem** that helps financial institutions automate and improve their loan approval decision-making process.

---

## 2. Dataset Description

**Source:** Kaggle - Loan Approval Dataset  
**Type:** Binary Classification  

### Dataset Statistics:
- **Total Instances:** 30,000+
- **Total Features:** 14 (13 input features + 1 target)
- **Target Variable:** `loan_status` (0 = Rejected, 1 = Approved)

### Feature Description:

| Feature | Description | Type |
|---------|-------------|------|
| `person_age` | Age of the applicant | Numerical |
| `person_gender` | Gender of the applicant (male/female) | Categorical |
| `person_education` | Education level (High School, Bachelor, Master, Associate, etc.) | Categorical |
| `person_income` | Annual income of the applicant | Numerical |
| `person_emp_exp` | Employment experience in years | Numerical |
| `person_home_ownership` | Home ownership status (RENT, OWN, MORTGAGE, OTHER) | Categorical |
| `loan_amnt` | Loan amount requested | Numerical |
| `loan_intent` | Purpose of loan (EDUCATION, MEDICAL, VENTURE, PERSONAL, DEBTCONSOLIDATION, HOMEIMPROVEMENT) | Categorical |
| `loan_int_rate` | Interest rate of the loan | Numerical |
| `loan_percent_income` | Loan amount as percentage of annual income | Numerical |
| `cb_person_cred_hist_length` | Length of credit history in years | Numerical |
| `credit_score` | Credit score of the applicant | Numerical |
| `previous_loan_defaults_on_file` | Whether applicant has previous loan defaults (Yes/No) | Categorical |

---

## 3. Models Implemented

All 6 required classification models have been implemented:

1. **Logistic Regression** - Linear model for binary classification
2. **Decision Tree Classifier** - Tree-based model with interpretable rules
3. **K-Nearest Neighbors (KNN)** - Instance-based lazy learning algorithm
4. **Naive Bayes (Gaussian)** - Probabilistic classifier based on Bayes theorem
5. **Random Forest (Ensemble)** - Ensemble of decision trees using bagging
6. **XGBoost (Ensemble)** - Gradient boosting ensemble method

---

## 4. Model Comparison - Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8595 | 0.9580 | 0.6264 | 0.9390 | 0.7515 | 0.6852 |
| Decision Tree | 0.8975 | 0.8482 | 0.7818 | 0.7583 | 0.7699 | 0.7041 |
| KNN | 0.8878 | 0.9190 | 0.8069 | 0.6625 | 0.7276 | 0.6629 |
| Naive Bayes | 0.7459 | 0.9450 | 0.4707 | 0.9888 | 0.6378 | 0.5561 |
| Random Forest (Ensemble) | 0.9291 | 0.9743 | 0.9289 | 0.7436 | 0.8260 | 0.7898 |
| XGBoost (Ensemble) | 0.9327 | 0.9785 | 0.9005 | 0.7899 | 0.8416 | 0.8019 |

> **Note:** Run each model in the Streamlit app to obtain the actual metric values for the table above.

---

## 5. Model Performance Observations

---

## 6. Project Structure

```
project-folder/
│── app.py                    # Streamlit web application
│── requirements.txt          # Python dependencies
│── README.md                 # Project documentation
│── dataset/
│   └── loan_data.csv         # Training dataset
│── test_dataset/
│   └── loan_data_test.csv    # Test dataset
│── model/
│   ├── __init__.py           # Module exports
│   ├── base_model.py         # Base model class with common functionality
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── knn.py
│   ├── naive_bayes.py
│   ├── random_forest.py
│   └── xgboost_model.py
```

---

## 7. How to Run Locally

```bash
# Clone the repository
git clone <repository-url>
cd ml-assignment-2

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## 8. Streamlit App Features

- ✅ **Dataset Upload Option** - Upload custom CSV test data or use bundled test dataset
- ✅ **Model Selection Dropdown** - Choose from 6 different ML models
- ✅ **Evaluation Metrics Display** - Shows all 6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- ✅ **Confusion Matrix** - Visual confusion matrix with counts
- ✅ **Classification Report** - Detailed precision, recall, F1 per class
- ✅ **Feature Importance** - For models that support it (tree-based models)

---

## 9. Technologies Used

- Python 3.x
- Streamlit (Web UI)
- Scikit-learn (ML Models)
- XGBoost (Gradient Boosting)
- Pandas & NumPy (Data Processing)
- Matplotlib & Seaborn (Visualization)

---

**Author:** [Your Name]  
**Course:** M.Tech (AIML/DSE) - Machine Learning  
**Assignment:** Assignment 2
