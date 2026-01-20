from sklearn.neighbors import KNeighborsClassifier
from .base_model import BaseModel
from .data_loader import ProcessData

class LoanKNN(BaseModel):
    def __init__(self, test_size: float = 0.2, random_state: int = 42, n_neighbors=5) -> None:
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        # KNN requires scaling
        super().__init__(model, test_size, random_state, scale=True)

if __name__ == "__main__":
    raw_data = ProcessData.get_data()
    X, y = ProcessData.preprocess_data(raw_data)

    classifier = LoanKNN(n_neighbors=5)
    X_train, X_test, y_train, y_test = classifier.split_data(X, y)

    classifier.fit(X_train, y_train)
    metrics = classifier.evaluate(X_test, y_test)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("Classification report:\n", metrics["classification_report"])
