import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from io import StringIO

from model import (
    ProcessData,
    LibLogisticRegression,
    LoanDecisionTree,
    LoanKNN,
    LoanNaiveBayes,
    LoanRandomForest,
    LoanXGBoost,
)

# Page configuration
st.set_page_config(page_title="Loan Prediction Model Comparison", layout="wide")

st.title("🏦 Loan Prediction Model Comparison")
st.markdown("By default the bundled test dataset is used. Check the option below to upload your own CSV test dataset.")

# Sidebar for model selection
st.sidebar.header("Model Configuration")

model_options = {
    "Logistic Regression": LibLogisticRegression,
    "Decision Tree": LoanDecisionTree,
    "K-Nearest Neighbors": LoanKNN,
    "Naive Bayes (Gaussian)": LoanNaiveBayes,
    "Random Forest": LoanRandomForest,
    "XGBoost": LoanXGBoost,
}

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    options=list(model_options.keys()),
    help="Choose a classification model to evaluate"
)

# File upload section
st.header("📁 Dataset")

# Checkbox to allow user to upload their own dataset. Default: use bundled test dataset.
use_custom = st.checkbox(
    "Upload my own test dataset",
    value=False,
    help="Check to upload a CSV file to use as the test dataset instead of the bundled test dataset."
)

test_data = None
data_error = None

if use_custom:
    uploaded_file = st.file_uploader(
        "Upload your test dataset (CSV)",
        type=["csv"],
        help="Upload a CSV file containing loan data with the same features as the training set"
    )

    if uploaded_file is not None:
        try:
            test_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset uploaded successfully! Shape: {test_data.shape}")
            with st.expander("📊 Preview Dataset"):
                st.dataframe(test_data.head(10))
                st.write(f"**Columns:** {', '.join(test_data.columns.tolist())}")
        except Exception as e:
            data_error = f"Error reading uploaded file: {str(e)}"
            st.error(data_error)

else:
    # Load bundled test dataset by default
    st.info("Using bundled test dataset: `test_dataset/loan_data_test.csv`")
    try:
        test_data = pd.read_csv("test_dataset/loan_data_test.csv")
        st.success(f"✅ Loaded bundled test dataset. Shape: {test_data.shape}")
        with st.expander("📊 Preview Bundled Test Dataset"):
            st.dataframe(test_data.head(10))
            st.write(f"**Columns:** {', '.join(test_data.columns.tolist())}")
    except Exception as e:
        data_error = f"Error loading bundled test dataset: {str(e)}"
        st.error(data_error)

# If we have test_data, validate and preprocess
if test_data is not None:
    if 'loan_status' not in test_data.columns:
        st.error("❌ The dataset must contain a 'loan_status' column for evaluation.")
    else:
        try:
            with st.spinner("Processing data..."):
                X_test, y_test = ProcessData.preprocess_data(test_data)

            st.info(f"✓ Data preprocessed. Features shape: {X_test.shape}, Target shape: {y_test.shape}")

            if st.button("🚀 Train Model and Evaluate", type="primary"):
                with st.spinner(f"Training {selected_model_name} model..."):
                    raw_data = ProcessData.get_data()
                    X_full, y_full = ProcessData.preprocess_data(raw_data)

                    ModelClass = model_options[selected_model_name]
                    classifier = ModelClass()
                    classifier.fit(X_full, y_full)

                    metrics = classifier.evaluate(X_test, y_test)
                    predictions = classifier.predict(X_test)

                st.success(f"✅ {selected_model_name} trained and evaluated successfully!")

                # Display metrics
                st.header("📈 Evaluation Metrics")
                
                # Row 1: Accuracy, AUC, Precision
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                with col2:
                    if isinstance(metrics['roc_auc'], float):
                        st.metric("AUC Score", f"{metrics['roc_auc']:.4f}")
                    else:
                        st.metric("AUC Score", "N/A")
                with col3:
                    st.metric("Precision", f"{metrics['precision']:.4f}")

                # Row 2: Recall, F1, MCC
                col4, col5, col6 = st.columns(3)
                with col4:
                    st.metric("Recall", f"{metrics['recall']:.4f}")
                with col5:
                    st.metric("F1 Score", f"{metrics['f1']:.4f}")
                with col6:
                    st.metric("MCC Score", f"{metrics['mcc']:.4f}")

                # Summary table of all metrics
                st.subheader("📊 Metrics Summary Table")
                metrics_summary = pd.DataFrame({
                    'Metric': ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                    'Value': [
                        f"{metrics['accuracy']:.4f}",
                        f"{metrics['roc_auc']:.4f}" if isinstance(metrics['roc_auc'], float) else "N/A",
                        f"{metrics['precision']:.4f}",
                        f"{metrics['recall']:.4f}",
                        f"{metrics['f1']:.4f}",
                        f"{metrics['mcc']:.4f}"
                    ]
                })
                st.dataframe(metrics_summary, hide_index=True, use_container_width=True)

                # Confusion Matrix
                st.header("🎯 Confusion Matrix")
                cm = confusion_matrix(y_test, predictions)
                fig, ax = plt.subplots(figsize=(8, 6))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Approved", "Approved"])
                disp.plot(ax=ax, cmap='Blues', values_format='d')
                ax.set_title(f'Confusion Matrix - {selected_model_name}')
                st.pyplot(fig)

                # Additional metrics breakdown
                st.header("📊 Detailed Metrics Breakdown")
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Confusion Matrix Values")
                    tn, fp, fn, tp = cm.ravel()
                    metrics_df = pd.DataFrame({
                        'Metric': ['True Negatives', 'False Positives', 'False Negatives', 'True Positives'],
                        'Count': [tn, fp, fn, tp]
                    })
                    st.dataframe(metrics_df, hide_index=True)

                with col2:
                    st.subheader("Class Distribution")
                    class_dist = pd.DataFrame({
                        'Class': ['Not Approved (0)', 'Approved (1)'],
                        'Count': [len(y_test[y_test == 0]), len(y_test[y_test == 1])]
                    })
                    st.dataframe(class_dist, hide_index=True)

                # Feature importance (if available)
                try:
                    importance_df = classifier.feature_importance()
                    if not importance_df.empty:
                        st.header("🔍 Feature Importance")
                        st.dataframe(importance_df.head(15), hide_index=True)
                except:
                    pass

        except Exception as e:
            st.error(f"❌ Error processing dataset: {str(e)}")
            st.exception(e)

else:
    # No dataset available (either upload not checked or upload failed)
    if data_error is None:
        st.info("👆 No dataset selected. Check 'Upload my own test dataset' to upload a CSV, or leave it unchecked to use the bundled test dataset.")

    # Show expected format as guidance
    st.header("📝 Expected Dataset Format")
    st.markdown("""
    Your CSV file should contain the following columns:
    - `person_age`: Age of the person
    - `person_gender`: Gender (male/female)
    - `person_education`: Education level
    - `person_income`: Annual income
    - `person_emp_exp`: Employment experience (years)
    - `person_home_ownership`: Home ownership status
    - `loan_amnt`: Loan amount
    - `loan_intent`: Purpose of loan
    - `loan_int_rate`: Interest rate
    - `loan_percent_income`: Loan as percentage of income
    - `cb_person_cred_hist_length`: Credit history length
    - `credit_score`: Credit score
    - `previous_loan_defaults_on_file`: Previous defaults (Yes/No)
    - `loan_status`: Target variable (0 or 1)
    """)