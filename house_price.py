from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..common import TaskResult, ensure_output_dir, save_figure
from ..data import make_house_price_data


class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, x, y=None):
        data = pd.DataFrame(x).copy()
        self.lower_bounds_ = data.quantile(0.25) - self.factor * (data.quantile(0.75) - data.quantile(0.25))
        self.upper_bounds_ = data.quantile(0.75) + self.factor * (data.quantile(0.75) - data.quantile(0.25))
        return self

    def transform(self, x):
        data = pd.DataFrame(x).copy()
        for column in data.columns:
            data[column] = data[column].clip(self.lower_bounds_[column], self.upper_bounds_[column])
        return data


def run(output_dir: str | Path) -> TaskResult:
    output_path = ensure_output_dir(output_dir)
    df = make_house_price_data()
    features = df.drop(columns=["price"])
    target = df["price"]
    numeric_features = ["bedrooms", "bathrooms", "sqft", "lot_size", "age", "distance_to_city", "school_score", "garage"]
    categorical_features = ["neighborhood", "property_type", "condition"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", IQRClipper()),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    ridge_model = Pipeline(
        steps=[
            ("preprocessor", preprocessing),
            ("select", SelectKBest(score_func=f_regression, k=10)),
            ("model", Ridge(alpha=2.0)),
        ]
    )
    forest_model = Pipeline(
        steps=[
            ("preprocessor", preprocessing),
            ("model", RandomForestRegressor(n_estimators=300, random_state=42)),
        ]
    )

    ridge_model.fit(x_train, y_train)
    forest_model.fit(x_train, y_train)

    ridge_predictions = ridge_model.predict(x_test)
    forest_predictions = forest_model.predict(x_test)

    ridge_metrics = {
        "r2": round(r2_score(y_test, ridge_predictions), 4),
        "mae": round(mean_absolute_error(y_test, ridge_predictions), 2),
    }
    forest_metrics = {
        "r2": round(r2_score(y_test, forest_predictions), 4),
        "mae": round(mean_absolute_error(y_test, forest_predictions), 2),
    }

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, forest_predictions, alpha=0.45)
    min_value = min(y_test.min(), forest_predictions.min())
    max_value = max(y_test.max(), forest_predictions.max())
    plt.plot([min_value, max_value], [min_value, max_value], color="black", linestyle="--")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("House Price: Actual vs Predicted")
    save_figure(output_path / "house_price_predictions.png")

    return TaskResult(
        name="Task 2",
        metrics={"ridge": ridge_metrics, "random_forest": forest_metrics},
        artifact_paths=[str(output_path / "house_price_predictions.png")],
    )
