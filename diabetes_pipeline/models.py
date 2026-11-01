"""
Model definitions and hyperparameter configuration for Diabetes Early Prediction.
Includes all classifiers investigated in Alzboon et al. (2023) plus Gradient Boosting.
"""

from typing import Dict, Any, Tuple
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def get_classifier_suite(random_seed: int = 42) -> Dict[str, Tuple[BaseEstimator, Dict[str, Any]]]:
    """
    Returns a dictionary of classifiers along with corresponding hyperparameter tuning grids.
    """
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1500, random_state=random_seed),
            {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs"]}
        ),
        "Neural Network (MLP)": (
            MLPClassifier(max_iter=1500, random_state=random_seed, early_stopping=True),
            {"hidden_layer_sizes": [(64, 32), (50,), (30, 20)], "alpha": [0.0001, 0.001, 0.01]}
        ),
        "Support Vector Machine": (
            SVC(probability=True, random_state=random_seed),
            {"C": [0.5, 1.0, 5.0], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]}
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=random_seed),
            {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10], "criterion": ["gini", "entropy"]}
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"var_smoothing": [1e-9, 1e-8, 1e-7]}
        ),
        "k-Nearest Neighbors": (
            KNeighborsClassifier(),
            {"n_neighbors": [5, 7, 9, 11], "weights": ["uniform", "distance"]}
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=random_seed),
            {"max_depth": [3, 5, 8], "criterion": ["gini", "entropy"]}
        ),
        "SGD Classifier": (
            SGDClassifier(loss="log_loss", max_iter=2000, random_state=random_seed),
            {"alpha": [0.0001, 0.001, 0.01], "penalty": ["l2", "l1"]}
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=random_seed),
            {"n_estimators": [50, 100], "learning_rate": [0.1, 0.5, 1.0]}
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=random_seed),
            {"n_estimators": [100, 150], "learning_rate": [0.05, 0.1], "max_depth": [3, 4]}
        ),
    }

    return models
