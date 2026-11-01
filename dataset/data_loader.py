"""
Data Loading and Preprocessing Module for Early Diabetes Prediction.
Dataset: Pima Indians Diabetes Dataset (National Institute of Diabetes and Digestive and Kidney Diseases).
"""

from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


FEATURE_COLUMNS: List[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET_COLUMN: str = "Outcome"

PHYSIOLOGICAL_ZERO_COLS: List[str] = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def load_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load the Pima Indians Diabetes CSV file into a pandas DataFrame."""
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent / "diabetes.csv"
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Diabetes dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def preprocess_data(
    df: pd.DataFrame,
    impute_missing_zeros: bool = True
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Cleans physiological zero values and standardizes features.
    
    In the Pima dataset, values of 0 for Glucose, BloodPressure, SkinThickness,
    Insulin, and BMI are clinically invalid and represent unrecorded/missing data.
    """
    clean_df = df.copy()

    if impute_missing_zeros:
        for col in PHYSIOLOGICAL_ZERO_COLS:
            if col in clean_df.columns:
                clean_df[col] = clean_df[col].replace(0, np.nan)

        imputer = SimpleImputer(strategy="median")
        clean_df[FEATURE_COLUMNS] = imputer.fit_transform(clean_df[FEATURE_COLUMNS])

    X = clean_df[FEATURE_COLUMNS].values
    y = clean_df[TARGET_COLUMN].values.astype(int)

    return X, y, clean_df


def get_scaled_splits(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.30,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Splits dataset into stratified train and test sets (default: 70/30 as in paper)
    and standardizes feature values with StandardScaler.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
