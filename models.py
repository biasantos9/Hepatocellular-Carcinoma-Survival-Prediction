import os
import random
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd

from utils import (CleanNumeric, CorrFeatureSelector, CVConfig, scorers)

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    RandomizedSearchCV, cross_validate, StratifiedKFold
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from joblib import Memory

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense
    from scikeras.wrappers import KerasClassifier
    HAS_NN = True
except ImportError as e:
    HAS_NN = False

random.seed(42)
np.random.seed(42)
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

def build_pipeline(estimator, k_top: Optional[int]):
    return Pipeline(
        memory=Memory(location=CACHE_DIR, verbose=0),
        steps=[
            ("clean", CleanNumeric()),
            ("impute", KNNImputer(n_neighbors=5)), 
            ("scale", MinMaxScaler()),
            ("select", CorrFeatureSelector(k=k_top)),
            ("clf", estimator),
        ]
    )

def create_nn_model(meta, neurons: int = 1, learning_rate: float = 0.001):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_dim=meta["n_features_in_"]))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def get_model_space() -> Dict[str, Tuple[object, dict]]:
    space: Dict[str, Tuple[object, dict]] = {
        "LogReg": (
            LogisticRegression(max_iter=1000, solver="liblinear"),
            {"clf__C": [0.001, 0.01, 0.1, 1, 10]},
        ),
        "RandomForest": (
            RandomForestClassifier(n_jobs=1, random_state=42),
            {"clf__n_estimators": [100, 300, 500],
             "clf__max_depth": [None, 5, 10]},
        ),
        "SVM": (
            SVC(),
            {"clf__C": [0.001, 0.01, 0.1, 1, 10],
             "clf__gamma": ["scale", "auto"]},
        ),
    }
    if HAS_LGB:
        space["LightGBM"] = (
            lgb.LGBMClassifier(verbose=-1, n_jobs=1, random_state=42),
            {"clf__num_leaves": [20, 30, 40],
             "clf__learning_rate": [0.01, 0.05, 0.1],
             "clf__n_estimators": [200, 400]},
        )
    if HAS_NN:
        space["NeuralNetwork"] = (
            KerasClassifier(model=create_nn_model, verbose=0, random_state=42, metrics=["accuracy"]),
            {
                "clf__model__neurons": [5, 10, 25, 50],
                "clf__epochs": [100, 150],
                "clf__batch_size": [10, 20],
                "clf__model__learning_rate": [0.001, 0.01],
            },
        )
    return space

def run_experiments(
    df: pd.DataFrame,
    target_col: str = "Class",
    topk_options: List[Optional[int]] = (None, 12),
    cv_cfg: CVConfig = CVConfig(),
    n_iter: int = 12,
    parallel_models: int = 1
) -> pd.DataFrame:
    y = df[target_col].astype(int).values
    X = df.drop(columns=[target_col])

    rows = []
    inner = StratifiedKFold(n_splits=3, shuffle=cv_cfg.shuffle, random_state=cv_cfg.random_state)
    outer = StratifiedKFold(n_splits=cv_cfg.n_splits, shuffle=cv_cfg.shuffle, random_state=cv_cfg.random_state)

    for k_top in topk_options:
        feature_set = "all" if k_top is None else f"top_{k_top}"
        for name, (estimator, dist) in get_model_space().items():
            pipe = build_pipeline(estimator, k_top=k_top)

            rs = RandomizedSearchCV(estimator=pipe, param_distributions=dist, n_iter=n_iter, cv=inner, scoring="f1",
                                     refit=True, n_jobs=1, pre_dispatch="1*n_jobs", random_state=42, verbose=0)
            
            scores = cross_validate(rs, X, y, scoring=scorers, cv=outer, n_jobs=parallel_models,
                                     pre_dispatch="1*n_jobs", return_estimator=False, error_score="raise")

            rows.append({
                "feature_set": feature_set,
                "model": name,
                "mean_accuracy": np.mean(scores["test_accuracy"]),
                "mean_f1": np.mean(scores["test_f1"]),
                "mean_recall": np.mean(scores["test_recall"]),
                "mean_precision": np.mean(scores["test_precision"]),
            })

    res = (
        pd.DataFrame(rows)
        .sort_values(["feature_set", "model", "mean_f1"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return res

def print_scores_per_model(results: pd.DataFrame, fs_top: str = "top_12", fs_all: str = "all", decimals: int = 4):
    fmt = lambda x: f"{x:.{decimals}f}"
    for fs, title in [(fs_top, "Scores with 12 most important features"),
                      (fs_all, "Scores with all features")]:
        sub = results[results["feature_set"] == fs].copy()
        if sub.empty:
            continue
        print(title)
        for _, row in sub.sort_values("model").iterrows():
            print(f"Model = {row['model']}")
            print("Accuracy =", fmt(row["mean_accuracy"]))
            print("F1 =", fmt(row["mean_f1"]))
            print("Recall =", fmt(row["mean_recall"]))
            print("Precision =", fmt(row["mean_precision"]))
            print()

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "hcc.csv"

def main():
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df["Class"] = pd.to_numeric(df["Class"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Class"]).copy()
    df["Class"] = df["Class"].astype(int)

    cfg = CVConfig(n_splits=5, shuffle=True, random_state=42)

    results = run_experiments(df=df, target_col="Class", topk_options=[None, 12], cv_cfg=cfg, n_iter=12, parallel_models=1)
    print_scores_per_model(results, fs_top="top_12", fs_all="all", decimals=4)

if __name__ == "__main__":
    main()