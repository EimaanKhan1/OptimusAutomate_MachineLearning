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
from ..data import make_churn_data


def run(output_dir: str | Path) -> TaskResult:
    output_path = ensure_output_dir(output_dir)
    df = make_churn_data()
    features = df.drop(columns=["churn"])
    target = df["churn"]
    numeric_features = ["tenure", "monthly_charges", "total_charges", "support_tickets", "senior_citizen", "auto_pay"]
    categorical_features = ["contract", "internet_service", "payment_method", "tech_support", "paperless_billing"]
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
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample"),
    }

    metrics = {}
    feature_importance_artifact = output_path / "churn_feature_importance.png"
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
        if name == "random_forest":
            transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            importance = pd.Series(pipeline.named_steps["model"].feature_importances_, index=transformed_names)
            top_importance = importance.sort_values(ascending=False).head(12)
            plt.figure(figsize=(9, 5))
            top_importance.sort_values().plot(kind="barh")
            plt.title("Customer Churn: Top Feature Importances")
            plt.xlabel("Importance")
            save_figure(feature_importance_artifact)

    return TaskResult(
        name="Task 3",
        metrics=metrics,
        artifact_paths=[str(feature_importance_artifact)],
    )
