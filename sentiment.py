from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ..common import TaskResult, ensure_output_dir, save_figure
from ..data import make_sentiment_data


def run(output_dir: str | Path) -> TaskResult:
    output_path = ensure_output_dir(output_dir)
    df = make_sentiment_data()
    distribution = df["sentiment"].value_counts().reindex(["positive", "negative", "neutral"])
    distribution.plot(kind="bar", color=["#3CB371", "#CD5C5C", "#808080"], figsize=(7, 4))
    plt.title("Sentiment Distribution")
    plt.xlabel("Class")
    plt.ylabel("Review Count")
    save_figure(output_path / "sentiment_distribution.png")

    x_train, x_test, y_train, y_test = train_test_split(
        df["review"], df["sentiment"], test_size=0.3, random_state=42, stratify=df["sentiment"]
    )

    models = {
        "naive_bayes": MultinomialNB(alpha=0.5),
        "linear_svm": LinearSVC(class_weight="balanced"),
    }

    metrics = {}
    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics[name] = {
            "accuracy": round(accuracy_score(y_test, predictions), 4),
            "precision": round(precision_score(y_test, predictions, average="macro", zero_division=0), 4),
            "recall": round(recall_score(y_test, predictions, average="macro", zero_division=0), 4),
            "f1": round(f1_score(y_test, predictions, average="macro", zero_division=0), 4),
        }

    return TaskResult(
        name="Task 4",
        metrics=metrics,
        artifact_paths=[str(output_path / "sentiment_distribution.png")],
    )
