from typing import Tuple

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class BaseModel:
    def __init__(self, model: BaseEstimator, test_size: float = 0.2, random_state: int = 42, scale: bool = True) -> None:
        self.model = model
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler() if scale else None
        self.feature_names = []

    def split_data(
        self, features: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return train_test_split(
            features,
            target,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=target,
        )

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "BaseModel":
        if self.scaler:
            scaled_features = self.scaler.fit_transform(features)
        else:
            scaled_features = features
        
        self.model.fit(scaled_features, target)
        self.feature_names = list(features.columns)
        return self

    def _transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.scaler:
            return self.scaler.transform(features)
        return features

    def predict(self, features: pd.DataFrame) -> pd.Series:
        scaled_features = self._transform(features)
        return pd.Series(self.model.predict(scaled_features), index=features.index)

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        scaled_features = self._transform(features)
        if hasattr(self.model, "predict_proba"):
            return pd.Series(
                self.model.predict_proba(scaled_features)[:, 1], index=features.index
            )
        else: # Fallback for models without probability output, though most requested have it
            return self.predict(features) 

    def evaluate(
        self, features: pd.DataFrame, target: pd.Series
    ) -> dict[str, object]:
        predictions = self.predict(features)
        # Try to get probabilities, handle exception if model doesn't support it
        try:
            probabilities = self.predict_proba(features)
            auc = roc_auc_score(target, probabilities)
        except (AttributeError, NotImplementedError):
            auc = "N/A"

        return {
            "accuracy": accuracy_score(target, predictions),
            "roc_auc": auc,
            "precision": precision_score(target, predictions, zero_division=0),
            "recall": recall_score(target, predictions, zero_division=0),
            "f1": f1_score(target, predictions, zero_division=0),
            "mcc": matthews_corrcoef(target, predictions),
            "classification_report": classification_report(
                target, predictions, zero_division=0
            ),
        }

    def feature_importance(self) -> pd.DataFrame:
        if not self.feature_names:
             raise ValueError("Model must be trained before retrieving feature importance.")
        
        importance = None
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
             importance = self.model.coef_[0]
        
        if importance is not None:
             return pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": importance,
                }
            ).sort_values(by="importance", key=lambda x: x.abs(), ascending=False)
        return pd.DataFrame()
