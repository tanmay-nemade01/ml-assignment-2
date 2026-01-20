from sklearn.naive_bayes import GaussianNB, MultinomialNB
from .base_model import BaseModel
from .data_loader import ProcessData

class LoanNaiveBayes(BaseModel):
    def __init__(self, test_size: float = 0.2, random_state: int = 42, method='gaussian') -> None:
        if method == 'gaussian':
            model = GaussianNB()
            scale = True # Gaussian typically benefits from scaling if features vary largely
        elif method == 'multinomial':
            model = MultinomialNB()
            scale = False # Multinomial expects non-negative counts, standard scaler produces negatives
        else:
            raise ValueError("Method must be 'gaussian' or 'multinomial'")
            
        super().__init__(model, test_size, random_state, scale=scale)

if __name__ == "__main__":
    raw_data = ProcessData.get_data()
    X, y = ProcessData.preprocess_data(raw_data)

    # Note: For MultinomialNB, ensure features are non-negative. 
    # Our data has continuous scaled features, so Gaussian is safer or use MinMax scaling.
    # The BaseModel uses StandardScaler by default which can produce negatives.
    # We default to Gaussian here.
    classifier = LoanNaiveBayes(method='gaussian')
    X_train, X_test, y_train, y_test = classifier.split_data(X, y)

    classifier.fit(X_train, y_train)
    metrics = classifier.evaluate(X_test, y_test)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("Classification report:\n", metrics["classification_report"])
