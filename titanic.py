from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..common import TaskResult, ensure_output_dir, save_figure
from ..data import make_titanic_data


def run(output_dir: str | Path) -> TaskResult:
    output_path = ensure_output_dir(output_dir)
    df = make_titanic_data()
    features = df.drop(columns=["survived"])
    target = df["survived"]
    numeric_features = ["age", "sibsp", "parch", "fare", "family_size", "pclass"]
    categorical_features = ["sex", "embarked"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42, stratify=target
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }

    metrics = {}
    for name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics[name] = {
            "accuracy": round(accuracy_score(y_test, predictions), 4),
            "precision": round(precision_score(y_test, predictions), 4),
            "recall": round(recall_score(y_test, predictions), 4),
            "f1": round(f1_score(y_test, predictions), 4),
        }

    comparison = pd.DataFrame(metrics).T
    comparison.plot(kind="bar", figsize=(10, 5), rot=0)
    plt.title("Titanic Model Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    save_figure(output_path / "titanic_model_comparison.png")

    return TaskResult(
        name="Task 1",
        metrics=metrics,
        artifact_paths=[str(output_path / "titanic_model_comparison.png")],
    )
