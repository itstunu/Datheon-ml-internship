# Task2_Housing_Regression/task2_regression.py
"""Task 2: Regression on the California Housing dataset.

Pipeline: load -> feature engineering -> correlation heatmap -> split ->
scale -> OLS / Ridge / Lasso (GridSearchCV for alpha) -> metrics -> plots.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-GUI backend.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0]


def load_and_engineer_features():
    """Load California Housing and add two engineered features.

    Returns a DataFrame that includes the target column 'MedHouseVal'.
    """
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    df["Rooms_per_Person"] = df["AveRooms"] / df["AveOccup"]
    df["Bedrooms_per_Room"] = df["AveBedrms"] / df["AveRooms"]
    return df


def plot_correlation_heatmap(df):
    """Save a correlation heatmap of all features (and target) to correlation_heatmap.png."""
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("California Housing - Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def compute_metrics(y_true, y_pred):
    """Return a dict with RMSE, MAE and R2."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_models(X_train, y_train):
    """Fit OLS, Ridge and Lasso (alpha tuned via GridSearchCV).

    Each model is wrapped in a Pipeline with StandardScaler, so scaling is fit
    on the training folds only. Returns {name: fitted_estimator}.
    """
    fitted = {}

    ols = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    ols.fit(X_train, y_train)
    fitted["Linear Regression"] = ols

    for name, estimator in [("Ridge", Ridge(random_state=RANDOM_STATE)),
                            ("Lasso", Lasso(max_iter=10000, random_state=RANDOM_STATE))]:
        pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        grid = GridSearchCV(pipe, {"model__alpha": ALPHAS}, cv=5,
                            scoring="neg_root_mean_squared_error", n_jobs=-1)
        grid.fit(X_train, y_train)
        print(f"{name}: best alpha = {grid.best_params_['model__alpha']} "
              f"(CV RMSE = {-grid.best_score_:.4f})")
        fitted[name] = grid.best_estimator_

    return fitted


def plot_predicted_vs_actual(y_true, y_pred, model_name):
    """Save a predicted-vs-actual scatter plot to predicted_vs_actual.png."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.3, s=12, edgecolor="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual median house value ($100k)")
    ax.set_ylabel("Predicted median house value ($100k)")
    ax.set_title(f"Predicted vs. Actual - {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)


def plot_residuals(y_true, y_pred, model_name):
    """Save a residual histogram (with KDE curve) to residuals.png."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, bins=50, kde=True, ax=ax)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Residual Distribution - {model_name}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "residuals.png", dpi=150)
    plt.close(fig)


def main():
    """Run the full Task 2 pipeline."""
    df = load_and_engineer_features()
    print(f"Loaded California Housing: {df.shape[0]} rows, {df.shape[1]} columns "
          "(including 2 engineered features and the target)\n")

    plot_correlation_heatmap(df)
    print("Saved: correlation_heatmap.png\n")

    X = df.drop(columns="MedHouseVal")
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print("Tuning models (GridSearchCV, 5-fold)...")
    models = build_models(X_train, y_train)

    print("\n" + "=" * 60)
    print("TEST SET METRICS")
    print("=" * 60)
    print(f"{'Model':<20}{'RMSE':>10}{'MAE':>10}{'R2':>10}")
    print("-" * 50)

    results, predictions = {}, {}
    for name, model in models.items():
        preds = model.predict(X_test)
        predictions[name] = preds
        results[name] = compute_metrics(y_test, preds)
        m = results[name]
        print(f"{name:<20}{m['RMSE']:>10.4f}{m['MAE']:>10.4f}{m['R2']:>10.4f}")

    best_name = min(results, key=lambda k: results[k]["RMSE"])
    print(f"\nBest model (lowest RMSE): {best_name}")

    plot_predicted_vs_actual(y_test.to_numpy(), predictions[best_name], best_name)
    plot_residuals(y_test.to_numpy(), predictions[best_name], best_name)
    print("Saved: predicted_vs_actual.png, residuals.png")


if __name__ == "__main__":
    main()