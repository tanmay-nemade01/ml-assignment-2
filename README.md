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
| Naive Bayes | 0.7335 | 0.9410 | 0.4590 | 0.9976 | 0.6288 | 0.5473 |
| Random Forest (Ensemble) | 0.9291 | 0.9743 | 0.9289 | 0.7436 | 0.8260 | 0.7898 |
| XGBoost (Ensemble) | 0.9327 | 0.9785 | 0.9005 | 0.7899 | 0.8416 | 0.8019 |

> **Note:** Run each model in the Streamlit app to obtain the actual metric values for the table above.

---

## 5. Model Performance Observations

After training all six models on the loan approval dataset, some clear patterns emerge in how different algorithms handle this classification task.

### The Standout Performers

**XGBoost takes the crown** with 93.27% accuracy and the best overall balance. It achieves excellent precision (90.05%) while maintaining solid recall (78.99%), which translates to the highest F1 score (0.8416) and MCC (0.8019). The AUC of 0.9785 shows it's exceptional at ranking loan applicants by risk. This model would be my go-to choice for production deployment.

**Random Forest comes in a close second** at 92.91% accuracy. What really stands out is its precision of 92.89%—the highest among all models. This means when Random Forest says "approve," it's almost always right. The trade-off is slightly lower recall (74.36%), so it might be a bit more conservative. If your priority is avoiding bad loans even if it means rejecting some good applicants, this is your model.

### The Balanced Middle Ground

**Logistic Regression** delivers solid 85.95% accuracy with an interesting characteristic: extremely high recall (93.90%). It catches almost all the legitimate loan applicants, though at the cost of lower precision (62.64%). This means it approves more loans overall, including some that might default. The excellent AUC (0.9580) suggests it's great for probability estimates, making it useful when you want to set custom decision thresholds.

**Decision Tree** hits 89.75% accuracy with a nice balance—precision of 78.18% and recall of 75.83%. It's interpretable and performs decently across the board. The lower AUC (0.8482) compared to other models suggests it's not as confident in its probability estimates, but it's still a respectable baseline.

**KNN** achieves 88.78% accuracy with strong precision (80.69%) but weaker recall (66.25%). It tends to be conservative like Random Forest but doesn't quite reach the same performance level. The good AUC (0.9190) shows it has decent ranking ability, though it requires careful feature scaling and can be computationally expensive on large datasets.

### The Special Case

**Naive Bayes** is the outlier here. At 73.35% accuracy, it's clearly the weakest in terms of overall correctness. However, it has the most extreme recall (99.76%)—it catches virtually every single approved loan! The downside? Precision is only 45.90%, meaning more than half of the loans it approves might actually default. 

This isn't necessarily a bad model; it just has a very specific use case. If you're in a competitive lending market where missing a good customer is worse than approving a few risky ones, Naive Bayes could be strategically valuable. Its strong AUC (0.9410) also means you could tune the decision threshold to find a better balance.

### What This Means for Real-World Use

**If you're risk-averse** (like a conservative bank): Use Random Forest or XGBoost. Their high precision means fewer bad loans slip through.

**If you're competing for customers** (like a fintech startup): Consider Logistic Regression or even Naive Bayes with a tuned threshold. You'll approve more loans and capture more market share, accepting slightly higher default risk.

**If you need the best overall system**: Go with XGBoost. It threads the needle between catching good applicants and avoiding bad ones.

**If you need interpretability**: Decision Tree is your friend. You can literally trace the decision path and explain to regulators or customers why a loan was approved or denied.

### The Class Imbalance Challenge

All models are dealing with an imbalanced dataset (78% rejections vs 22% approvals). The ensemble methods (Random Forest and XGBoost) handle this best through their class weighting strategies. Naive Bayes struggles the most, which explains its tendency to over-predict approvals. The MCC scores confirm this—XGBoost (0.8019) and Random Forest (0.7898) show strong correlation with actual outcomes even accounting for class imbalance.

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

**Author:** Tanmay Nemade
