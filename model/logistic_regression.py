import pandas as pd
from pathlib import Path

def get_data():
    # Locate dataset relative to project root (two levels up from this file)
    DATA_PATH = Path(__file__).resolve().parents[1] / "dataset" / "loan_data.csv"

    dataset = pd.read_csv(DATA_PATH)
    return dataset

