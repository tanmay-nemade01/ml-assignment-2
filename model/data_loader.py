from pathlib import Path
from typing import Tuple

import pandas as pd


class ProcessData:
    @staticmethod
    def get_data() -> pd.DataFrame:
        DATA_PATH = (
            Path(__file__).resolve().parents[1] / "dataset" / "loan_data.csv"
        )
        return pd.read_csv(DATA_PATH)

    @staticmethod
    def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        data = data.copy()
        data = data.dropna()

        defaults_map = {"Yes": 1, "No": 0}
        data["previous_loan_defaults_on_file"] = (
            data["previous_loan_defaults_on_file"].map(defaults_map).fillna(0).astype(int)
        )

        categorical_cols = [
            "person_gender",
            "person_education",
            "person_home_ownership",
            "loan_intent",
        ]
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

        target = data["loan_status"].astype(int)
        features = data.drop(columns=["loan_status"])
        return features, target
