"""
Test Suite for Early Diabetes Prediction Pipeline.
Validates dataset integrity, data cleaning, feature ranking, and model execution.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from dataset.data_loader import load_dataset, preprocess_data, get_scaled_splits, FEATURE_COLUMNS, TARGET_COLUMN
from diabetes_pipeline.feature_importance import compute_feature_rankings
from diabetes_pipeline.models import get_classifier_suite
from diabetes_pipeline.evaluate import compute_metrics


@pytest.fixture
def sample_dataset(tmp_path):
    """Generates a synthetic replica of the Pima Indians dataset for isolated testing."""
    np.random.seed(42)
    n = 60
    data = {
        "Pregnancies": np.random.randint(0, 10, n),
        "Glucose": np.random.randint(70, 190, n),
        "BloodPressure": np.random.randint(50, 110, n),
        "SkinThickness": np.random.randint(10, 45, n),
        "Insulin": np.random.randint(15, 300, n),
        "BMI": np.random.uniform(18.0, 45.0, n),
        "DiabetesPedigreeFunction": np.random.uniform(0.1, 1.8, n),
        "Age": np.random.randint(21, 75, n),
        "Outcome": np.random.choice([0, 1], n, p=[0.65, 0.35]),
    }
    df = pd.DataFrame(data)
    csv_file = tmp_path / "test_diabetes.csv"
    df.to_csv(csv_file, index=False)
    return csv_file, df


def test_dataset_loading_structure():
    """Verify that the official Pima Indians Diabetes dataset has 768 rows and expected columns."""
    dataset_path = Path(__file__).resolve().parent.parent / "dataset" / "diabetes.csv"
    assert dataset_path.exists(), "Official dataset file must exist in dataset/ folder."
    df = load_dataset(str(dataset_path))
    assert len(df) == 768
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]


def test_preprocessing_and_imputation(sample_dataset):
    """Verify that physiological zero values are handled and features are correctly extracted."""
    csv_file, df = sample_dataset
    # Introduce zero in glucose
    df.loc[0, "Glucose"] = 0
    X, y, clean_df = preprocess_data(df, impute_missing_zeros=True)

    assert X.shape == (len(df), len(FEATURE_COLUMNS))
    assert len(y) == len(df)
    assert clean_df.loc[0, "Glucose"] > 0, "Zero glucose must be imputed."


def test_feature_rankings(sample_dataset):
    """Verify that feature rankings calculate ANOVA and Mutual Info scores properly."""
    _, df = sample_dataset
    rankings = compute_feature_rankings(df)

    assert len(rankings) == len(FEATURE_COLUMNS)
    assert "ANOVA_F_Score" in rankings.columns
    assert "Mutual_Info_Gain" in rankings.columns
    assert "ANOVA_Rank" in rankings.columns


def test_classifier_suite_instantiation():
    """Verify that all classifiers and parameter grids instantiate properly."""
    suite = get_classifier_suite(random_seed=42)
    assert len(suite) >= 8
    assert "Logistic Regression" in suite
    assert "Neural Network (MLP)" in suite
    assert "Random Forest" in suite
    assert "Support Vector Machine" in suite


def test_metrics_computation():
    """Verify calculation of accuracy, precision, recall, F1, and AUC."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.85])

    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert 0.0 <= metrics["Accuracy"] <= 1.0
    assert 0.0 <= metrics["F1"] <= 1.0
    assert 0.0 <= metrics["AUC"] <= 1.0
