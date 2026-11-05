# Task1_Iris_Classification/task1_classification.py
"""Task 1: Multi-class classification on the Iris dataset.

Pipeline: load data -> EDA pairplot -> stratified 80/20 split -> scaling ->
train Logistic Regression and Decision Tree -> evaluate -> compare.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-GUI backend: safe for servers / CI.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

OUTPUT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42


def load_data():
    """Load the Iris dataset and return (DataFrame, X, y, class_names)."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris.data, iris.target, list(iris.target_names)


def plot_pairplot(df):
    """Save an exploratory seaborn pairplot colored by species to pairplot.png."""
    grid = sns.pairplot(df, hue="species", diag_kind="hist", corner=False)
    grid.figure.suptitle("Iris Dataset - Pairplot", y=1.02)
    grid.savefig(OUTPUT_DIR / "pairplot.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Save a confusion-matrix heatmap as confusion_matrix_<model_name>.png."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix - {model_name.replace('_', ' ').title()}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"confusion_matrix_{model_name}.png", dpi=150)
    plt.close(fig)


def evaluate_model(name, model, X_test, y_test, class_names):
    """Predict on the test set, print a classification report, save the
    confusion matrix, and return the accuracy."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=" * 60)
    print(f"MODEL: {name.replace('_', ' ').title()}")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    plot_confusion_matrix(y_test, y_pred, class_names, name)
    return acc


def main():
    """Run the full Task 1 pipeline."""
    df, X, y, class_names = load_data()
    print(f"Loaded Iris dataset: {df.shape[0]} samples, {X.shape[1]} features, "
          f"{len(class_names)} classes\n")

    plot_pairplot(df)
    print("Saved: pairplot.png\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Fit the scaler on training data only to avoid data leakage.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        results[name] = evaluate_model(name, model, X_test_s, y_test, class_names)

    print("=" * 60)
    print("FINAL COMPARISON (Test Accuracy)")
    print("=" * 60)
    print(f"{'Model':<25}{'Accuracy':>10}")
    print("-" * 35)
    for name, acc in results.items():
        print(f"{name.replace('_', ' ').title():<25}{acc:>10.4f}")
    print("\nSaved: confusion_matrix_logistic_regression.png, "
          "confusion_matrix_decision_tree.png")


if __name__ == "__main__":
    main()