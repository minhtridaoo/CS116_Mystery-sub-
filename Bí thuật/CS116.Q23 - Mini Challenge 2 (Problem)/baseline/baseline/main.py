import argparse
import os
import sys

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LABEL_COLUMN = "two_year_recid"


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X):
    categorical_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric_features = [column for column in X.columns if column not in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_candidates(X):
    preprocessor = build_preprocessor(X)
    return {
        "logistic": Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
                (
                    "model",
                    LogisticRegression(
                        C=0.75,
                        class_weight="balanced",
                        max_iter=3000,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
                (
                    "model",
                    GradientBoostingClassifier(
                        learning_rate=0.045,
                        max_depth=2,
                        min_samples_leaf=18,
                        n_estimators=220,
                        random_state=42,
                        subsample=0.9,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        l2_regularization=0.08,
                        learning_rate=0.035,
                        max_iter=260,
                        max_leaf_nodes=12,
                        min_samples_leaf=24,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
                (
                    "model",
                    ExtraTreesClassifier(
                        class_weight="balanced_subsample",
                        max_depth=7,
                        min_samples_leaf=14,
                        n_estimators=420,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def best_threshold(y_true, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    thresholds = np.unique(np.quantile(probabilities, np.linspace(0.05, 0.95, 181)))

    best_score = -1.0
    best_value = 0.5
    for threshold in thresholds:
        score = f1_score(y_true, (probabilities >= threshold).astype(int))
        if score > best_score:
            best_score = score
            best_value = float(threshold)
    return best_value, best_score


def cross_validated_selection(X, y):
    n_splits = 5 if len(y) >= 1000 else 3
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    candidates = build_candidates(X)
    best_name = None
    best_threshold_value = 0.5
    best_score = -1.0

    for name, pipeline in candidates.items():
        out_of_fold = np.zeros(len(y), dtype=float)
        for train_idx, valid_idx in splitter.split(X, y):
            model = clone(pipeline)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            out_of_fold[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]

        threshold, score = best_threshold(y, out_of_fold)
        if score > best_score:
            best_name = name
            best_threshold_value = threshold
            best_score = score

    return best_name, best_threshold_value, best_score


def run_train(public_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)

    train_path = os.path.join(public_dir, "train_data", "train.json")
    df = pd.read_json(train_path, lines=True)
    X = df.drop(columns=[LABEL_COLUMN])
    y = df[LABEL_COLUMN].astype(int)

    best_name, threshold, _ = cross_validated_selection(X, y)
    model = build_candidates(X)[best_name]
    model.fit(X, y)

    dump({"model": model, "threshold": threshold}, os.path.join(model_dir, "trained_model.joblib"))


def run_predict(model_dir, test_input_dir, output_path):
    bundle = load(os.path.join(model_dir, "trained_model.joblib"))
    model = bundle["model"]
    threshold = bundle["threshold"]

    test_path = os.path.join(test_input_dir, "test.json")
    df_test = pd.read_json(test_path, lines=True)
    probabilities = model.predict_proba(df_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    pd.DataFrame({LABEL_COLUMN: predictions}).to_json(output_path, orient="records", lines=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--public_dir", type=str, required=True)
    parser_train.add_argument("--model_dir", type=str, required=True)

    parser_predict = subparsers.add_parser("predict")
    parser_predict.add_argument("--model_dir", type=str, required=True)
    parser_predict.add_argument("--test_input_dir", type=str, required=True)
    parser_predict.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()

    if args.command == "train":
        run_train(args.public_dir, args.model_dir)
    elif args.command == "predict":
        run_predict(args.model_dir, args.test_input_dir, args.output_path)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
