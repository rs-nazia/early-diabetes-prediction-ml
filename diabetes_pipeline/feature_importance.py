"""
Feature Importance and Statistical Ranking Module.
Reproduces feature ranking methodologies (ANOVA F-statistic, Mutual Information / Information Gain)
described in Table I of Alzboon et al. (2023).
"""

from typing import Dict
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from dataset.data_loader import FEATURE_COLUMNS, TARGET_COLUMN


def compute_feature_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes ANOVA F-statistic, p-values, and Mutual Information (Info Gain)
    for all 8 clinical and demographic features against diabetic outcome.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    f_scores, p_values = f_classif(X, y)
    mi_scores = mutual_info_classif(X, y, random_state=42)

    rankings_df = pd.DataFrame({
        "Feature_Code": [f"Fe{i+1}" for i in range(len(FEATURE_COLUMNS))],
        "Feature_Name": FEATURE_COLUMNS,
        "ANOVA_F_Score": f_scores.round(3),
        "P_Value": p_values.round(6),
        "Mutual_Info_Gain": mi_scores.round(4),
    })

    rankings_df["ANOVA_Rank"] = rankings_df["ANOVA_F_Score"].rank(ascending=False, method="min").astype(int)
    rankings_df["Info_Gain_Rank"] = rankings_df["Mutual_Info_Gain"].rank(ascending=False, method="min").astype(int)

    return rankings_df.sort_values(by="ANOVA_F_Score", ascending=False).reset_index(drop=True)
