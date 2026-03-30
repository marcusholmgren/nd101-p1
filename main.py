#!/usr/bin/env python3
"""
Project 1: Your First Neural Network - Bike Sharing Prediction
--------------------------------------------------------------
This script demonstrates a complete Machine Learning pipeline using scikit-learn
to predict bike-sharing demand. It serves as a modern, production-style counterpart
to the manual neural network implementation found in the project notebook.

Key Concepts Demonstrated:
1. Data Loading & Cleaning: Handling time-series-like data.
2. Feature Engineering: Creating dummy variables for categorical data.
3. Pipelines: Automating preprocessing (scaling) and model training.
4. Neural Networks: Using MLPRegressor (Multi-layer Perceptron).
5. Evaluation: Understanding MSE, RMSE, and R2 scores.

Usage:
    uv run main.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Locate the dataset relative to this script
DEFAULT_CSV = Path(__file__).resolve().parent / "Bike-Sharing-Dataset" / "hour.csv"


def load_and_prepare_data(path: Path) -> pd.DataFrame:
    """
    Loads the bike-sharing dataset and prepares it for modeling.

    In ML, we often need to convert categorical text/ID data (like 'season')
    into 'dummy' or 'one-hot' variables so the math-based model can understand them.
    """
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at {path}")

    df = pd.read_csv(path)

    # 1. Feature Engineering: Dummy Variables
    # We convert categories like 'season' (1, 2, 3, 4) into multiple columns (season_1, season_2, etc.)
    dummy_fields = ["season", "weathersit", "mnth", "hr", "weekday"]
    for field in dummy_fields:
        dummies = pd.get_dummies(df[field], prefix=field, drop_first=False)
        df = pd.concat([df, dummies], axis=1)

    # 2. Drop unnecessary columns
    # We drop columns that are:
    # - IDs (instant)
    # - Redundant (season, weathersit... we have dummies now)
    # - Leakage (casual, registered... they sum up to our target 'cnt')
    fields_to_drop = [
        "instant",
        "dteday",
        "season",
        "weathersit",
        "weekday",
        "atemp",
        "mnth",
        "workingday",
        "hr",
    ]
    df = df.drop(
        columns=[c for c in fields_to_drop if c in df.columns], errors="ignore"
    )

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separates the dataset into input features (X) and the target variable (y)."""
    target_col = "cnt"

    # 'casual' and 'registered' must be removed because they directly calculate 'cnt'.
    # If we kept them, the model would just learn 'cnt = casual + registered'
    # and fail to generalize to real-world scenarios where we don't know those yet.
    leakage_cols = ["casual", "registered"]

    X = df.drop(
        columns=[target_col] + [c for c in leakage_cols if c in df.columns],
        errors="ignore",
    )
    y = df[target_col].astype(float)

    return X, y


def get_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Creates a Scikit-Learn Pipeline.

    Pipelines ensure that our preprocessing (like scaling) is applied
    consistently to both training and testing data, preventing 'data leakage'.
    """
    # Identify numeric columns to scale. Binary (dummy) columns don't need scaling.
    # Scalers help Neural Networks converge faster by keeping inputs in a similar range.
    numeric_features = ["temp", "hum", "windspeed"]
    # All other columns are assumed to be our dummies
    dummy_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", "passthrough", dummy_features),
        ]
    )

    # MLPRegressor is a Multi-layer Perceptron (a basic Neural Network).
    # hidden_layer_sizes=(24, 12) means:
    # - Layer 1: 24 neurons
    # - Layer 2: 12 neurons
    model = MLPRegressor(
        hidden_layer_sizes=(24, 12),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )

    return Pipeline([("preprocessor", preprocessor), ("regressor", model)])


def plot_results(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    """Simple visualization to see how well our predictions track reality."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:100], label="Actual (cnt)", alpha=0.7)
    plt.plot(y_pred[:100], label="Predicted", linestyle="--")
    plt.title(f"First 100 Hours: {title}")
    plt.xlabel("Time (Hours)")
    plt.ylabel("Number of Bikes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # We don't call plt.show() here to avoid blocking;
    # in a real script we might save to file.
    plt.savefig(f"results_{title.lower().replace(' ', '_')}.png")
    print(f"Plot saved as results_{title.lower().replace(' ', '_')}.png")


def main():
    parser = argparse.ArgumentParser(description="Bike-Sharing Neural Network Pipeline")
    parser.add_argument("--data", type=str, default=str(DEFAULT_CSV))
    parser.add_argument(
        "--save", type=str, help="Path to save the model (e.g. model.joblib)"
    )
    args = parser.parse_args()

    print("--- 1. Loading Data ---")
    df = load_and_prepare_data(Path(args.data))
    print(f"Dataset loaded. Shape: {df.shape}")

    # Following the notebook: reserve the last 21 days for testing
    test_hours = 21 * 24
    train_data = df.iloc[:-test_hours]
    test_data = df.iloc[-test_hours:]

    X_train, y_train = split_features_target(train_data)
    X_test, y_test = split_features_target(test_data)

    # Further split training for validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print("\n--- 2. Building & Training Model ---")
    print(f"Training on {len(X_train_final)} samples, validating on {len(X_val)}")
    pipeline = get_pipeline(X_train_final)
    pipeline.fit(X_train_final, y_train_final)

    print("\n--- 3. Evaluation ---")
    for name, X_set, y_set in [
        ("Train", X_train_final, y_train_final),
        ("Val", X_val, y_val),
        ("Test", X_test, y_test),
    ]:
        preds = pipeline.predict(X_set)
        mse = mean_squared_error(y_set, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_set, preds)
        print(f"{name:5} | RMSE: {rmse:7.2f} (avg error in bikes) | R2: {r2:6.4f}")

        if name == "Test":
            plot_results(y_set.values, preds, "Test Set Predictions")

    if args.save:
        joblib.dump(pipeline, args.save)
        print(f"\nModel saved to {args.save}")


if __name__ == "__main__":
    main()
