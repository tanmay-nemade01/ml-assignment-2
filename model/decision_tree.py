from sklearn.tree import DecisionTreeClassifier
from .base_model import BaseModel
from .data_loader import ProcessData

class LoanDecisionTree(BaseModel):
    def __init__(self, test_size: float = 0.2, random_state: int = 42, max_depth=None) -> None:
        model = DecisionTreeClassifier(
            random_state=random_state, 
            class_weight="balanced",
            max_depth=max_depth
        )
        # Decision Trees don't strictly require scaling, but we keep the structure consistent. 
        # Set scale=False if you want raw features.
        super().__init__(model, test_size, random_state, scale=False)

if __name__ == "__main__":
    raw_data = ProcessData.get_data()
    X, y = ProcessData.preprocess_data(raw_data)

    classifier = LoanDecisionTree(max_depth=5)
    X_train, X_test, y_train, y_test = classifier.split_data(X, y)

    classifier.fit(X_train, y_train)
    metrics = classifier.evaluate(X_test, y_test)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("Classification report:\n", metrics["classification_report"])
