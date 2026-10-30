import warnings

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgbm
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from optuna.pruners import MedianPruner
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder

# silence LightGBM's eval_set deprecation noise and Optuna's per-trial logs
warnings.filterwarnings("ignore", message=".*eval_set.*")
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


search_spaces = {
    "LightGBM": lambda trial: {
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "class_weight": "balanced",
        "num_leaves": trial.suggest_int("num_leaves", 15, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "n_jobs": -1,
        "verbose": -1,
    },
    "XGBoost": lambda trial: {
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "scale_pos_weight": 1.0,  # overridden in objective() from y_train
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "tree_method": "hist",
        "eval_metric": "auc",
        "n_jobs": -1,
        "verbosity": 0,
    },
    "CatBoost": lambda trial: {
        "iterations": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "auto_class_weights": "Balanced",
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 20, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 5),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 3),
        "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        "eval_metric": "AUC",
        "verbose": 0,
        "thread_count": -1,
    },
}

model_classes = {
    "LightGBM": LGBMClassifier,
    "XGBoost": XGBClassifier,
    "CatBoost": CatBoostClassifier,
}


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score classification probabilities on validation and append to a tracked df.

    Returns:
        results: unformatted dataframe with all scores appended.
        new_results: rounded copy sorted by validation PR-AUC (descending).
    """
    cols = ["Model", "ROC-AUC", "Recall", "F1", "Precision", "PR-AUC", "Params"]
    if results is None:
        results = pd.DataFrame(columns=cols)

    pred_val = (np.asarray(proba_val) >= 0.5).astype(int)

    row = {
        "Model": model_name,
        "ROC-AUC": roc_auc_score(y_val, proba_val),
        "Recall": recall_score(y_val, pred_val, zero_division=0),
        "F1": f1_score(y_val, pred_val, zero_division=0),
        "Precision": precision_score(y_val, pred_val, zero_division=0),
        "PR-AUC": average_precision_score(y_val, proba_val),
        "Params": str(params),
    }
    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)

    metric_cols = ["ROC-AUC", "Recall", "F1", "Precision", "PR-AUC"]
    new_results = results.sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    new_results[metric_cols] = new_results[metric_cols].astype(float).round(3)

    return results, new_results


def objective(trial, name, X_train, y_train, preprocessor):
    """Mean ROC-AUC over 3 stratified folds. Preprocessor is cloned and fit per fold."""
    params = search_spaces[name](trial)

    if name == "XGBoost":
        params["scale_pos_weight"] = (y_train == 0).sum() / (y_train == 1).sum()

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs, best_iters = [], []

    for fold_num, (tr_idx, v_idx) in enumerate(skf.split(X_train, y_train)):
        prep = clone(preprocessor)
        X_tr, X_v = X_train.iloc[tr_idx], X_train.iloc[v_idx]
        y_tr, y_v = y_train.iloc[tr_idx], y_train.iloc[v_idx]

        X_tr_t = prep.fit_transform(X_tr, y_tr)
        X_v_t = prep.transform(X_v)

        model = model_classes[name](**params, random_state=42)

        if name == "LightGBM":
            model.fit(
                X_tr_t, y_tr,
                eval_set=[(X_v_t, y_v)],
                eval_metric="auc",
                callbacks=[lgbm.early_stopping(25, verbose=False)],
            )
            best_iters.append(model.best_iteration_)
        elif name == "XGBoost":
            model.set_params(early_stopping_rounds=25)
            model.fit(X_tr_t, y_tr, eval_set=[(X_v_t, y_v)], verbose=False)
            best_iters.append(model.best_iteration + 1)
        else:  # CatBoost
            model.fit(X_tr_t, y_tr, eval_set=(X_v_t, y_v), early_stopping_rounds=25)
            best_iters.append(model.get_best_iteration() + 1)

        aucs.append(roc_auc_score(y_v, model.predict_proba(X_v_t)[:, 1]))

        trial.report(float(np.mean(aucs)), step=fold_num)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    trial.set_user_attr("best_n_estimators", int(np.mean(best_iters)))
    return float(np.mean(aucs))


def early_stopping(study, trial, patience=20, min_delta=0.0005):
    """Stop if the best ROC-AUC hasn't improved by min_delta in the last `patience` trials."""
    completed = [t for t in study.trials if t.value is not None]
    if len(completed) < patience:
        return
    best = study.best_value
    if all(t.value < best - min_delta for t in completed[-patience:]):
        print(f"🛑 Early stopping: no {min_delta} AUC improvement in {patience} trials")
        study.stop()


def tune_models(names, X_train, y_train, preprocessor, n_trials=50):
    best_params_dict, studies = {}, {}

    for name in names:
        try:
            study = optuna.create_study(
                direction="maximize",
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
                sampler=optuna.samplers.TPESampler(seed=42),
            )
            study.optimize(
                lambda trial, n=name: objective(trial, n, X_train, y_train, preprocessor),
                n_trials=n_trials,
                n_jobs=1,
                show_progress_bar=True,
                callbacks=[early_stopping],
            )
            params = dict(study.best_params)
            n_est = study.best_trial.user_attrs["best_n_estimators"]
            params["iterations" if name == "CatBoost" else "n_estimators"] = n_est

            best_params_dict[name] = params
            studies[name] = study
            print(f"✅ {name}: best CV ROC-AUC = {study.best_value:.4f}")
        except Exception as e:
            print(f"🛑 Failed to tune {name}: {e}")

    return best_params_dict, studies