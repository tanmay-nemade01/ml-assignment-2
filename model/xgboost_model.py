from xgboost import XGBClassifier
from .base_model import BaseModel
from .data_loader import ProcessData

class LoanXGBoost(BaseModel):
    def __init__(self, test_size: float = 0.2, random_state: int = 42, n_estimators=100) -> None:
        model = XGBClassifier(
            n_estimators=n_estimators,
            random_state=random_state, 
            scale_pos_weight=1, # Adjust if imbalance is severe
            eval_metric='logloss'
        )
        super().__init__(model, test_size, random_state, scale=False)

if __name__ == "__main__":
    raw_data = ProcessData.get_data()
    X, y = ProcessData.preprocess_data(raw_data)

    classifier = LoanXGBoost()
    X_train, X_test, y_train, y_test = classifier.split_data(X, y)

    classifier.fit(X_train, y_train)
    metrics = classifier.evaluate(X_test, y_test)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("Classification report:\n", metrics["classification_report"])
