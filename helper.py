import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
)
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Fill NaNs in `target` with the median of its `group` (learned on train).
    Falls back to the global median for missing or unseen groups.
    Adds a `<target>_missing` flag column."""

    def __init__(self, target, group):
        self.target = target
        self.group = group

    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(self.group)[self.target].median().to_dict()
        self.global_median_ = X[self.target].median()
        return self

    def transform(self, X):
        X = X.copy()
        X[f"{self.target}_missing"] = X[self.target].isna().astype(int)
        fill = X[self.group].map(self.group_medians_).fillna(self.global_median_)
        X[self.target] = X[self.target].fillna(fill)
        return X

def score_predictions(
    proba_train,
    proba_val,
    y_train: pd.Series,
    y_val: pd.Series,
    model_name: str,
    params: dict,
    results: pd.DataFrame = None,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score classification probabilities on train and validation and append to a tracked df.

    Returns:
        results: unformatted dataframe with all scores appended.
        new_results: rounded copy sorted by validation PR-AUC (descending).
    """
    cols = ["Model", "Val PR-AUC", "Val ROC-AUC", "Train ROC-AUC",
            "F1", "Precision", "Recall", "Threshold", "Params"]
    if results is None:
        results = pd.DataFrame(columns=cols)

    pred_val = (proba_val >= threshold).astype(int)

    row = {
        "Model": model_name,
        "Val PR-AUC": average_precision_score(y_val, proba_val),
        "Val ROC-AUC": roc_auc_score(y_val, proba_val),
        "Train ROC-AUC": roc_auc_score(y_train, proba_train),
        "F1": f1_score(y_val, pred_val, zero_division=0),
        "Precision": precision_score(y_val, pred_val, zero_division=0),
        "Recall": recall_score(y_val, pred_val, zero_division=0),
        "Threshold": threshold,
        "Params": str(params),
    }
    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)

    metric_cols = ["Val PR-AUC", "Val ROC-AUC", "Train ROC-AUC", "F1", "Precision", "Recall"]
    new_results = results.sort_values(by="Val PR-AUC", ascending=False).reset_index(drop=True)
    new_results[metric_cols] = new_results[metric_cols].astype(float).round(3)

    return results, new_results