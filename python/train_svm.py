"""
train_svm.py
------------
Phase 2: Trains a separate SVM classifier for each patient group.

Reads:
    features/group1_features.csv   (EMG features, Stroke group)
    features/group2_features.csv   (IMU features, Parkinson's group)

Outputs:
    models/group1_svm.joblib       (scaler + SVM for Group 1)
    models/group2_svm.joblib       (scaler + SVM for Group 2)

Usage:
    python train_svm.py
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, works without a display
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_DIR = "features"
MODELS_DIR   = "models"

# ── Feature columns per group ─────────────────────────────────────────────────
GROUP1_FEATURES = [
    "emg_rms",
    "median_freq",
    "fatigue_index",
    "tremor_band_power",
    "peak_tremor_freq",
]

GROUP2_FEATURES = [
    "tremor_power",
    "tremor_frequency",
    "orientation_stability",
    "range_of_motion",
    "hold_stability",
    "drift_rate",
]

# ── SVM hyperparameter grid ───────────────────────────────────────────────────
PARAM_GRID = {
    "svm__C":     [0.1, 1, 10, 100],
    "svm__gamma": [0.001, 0.01, 0.1, "scale"],
}


def load_data(csv_path: str, feature_cols: list) -> tuple:
    """
    Load CSV and return X (features), y (labels), groups (subject IDs).
    Drops rows with any NaN in the feature columns.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=feature_cols)

    X      = df[feature_cols].values
    y      = df["label"].values.astype(int)
    groups = df["subject_id"].values  # used to split by subject, not by row

    print(f"  Loaded {len(df)} rows, {len(set(groups))} subjects")
    print(f"  Label distribution: { {int(k): int(v) for k,v in zip(*np.unique(y, return_counts=True))} }")
    return X, y, groups


def train_group(csv_path: str, feature_cols: list, group_name: str) -> Pipeline:
    """
    Full training pipeline for one patient group:
      1. Load data
      2. Subject-level train/test split (80/20)
      3. Normalize + train SVM with GridSearchCV
      4. Evaluate on test set
      5. Return fitted pipeline
    """
    print(f"\n── {group_name} ──────────────────────────────────────────")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    X, y, groups = load_data(csv_path, feature_cols)

    # ── 2. Subject-level split ────────────────────────────────────────────────
    # GroupShuffleSplit ensures all rows from one subject stay in the same set.
    # This prevents data leakage where the model sees the same patient in
    # both training and testing.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # ── 3. Build pipeline (scaler + SVM) ──────────────────────────────────────
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(
            kernel="rbf",
            class_weight="balanced",  # handles class imbalance
            probability=True,         # enables predict_proba for confidence scores
        )),
    ])

    # GridSearchCV tries every combination of C and gamma,
    # picks the one with highest macro F1 score using 5-fold cross-validation.
    # groups_train is passed so CV also splits by subject within training set.
    groups_train = groups[train_idx]
    cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    search = GridSearchCV(
        pipeline,
        PARAM_GRID,
        cv=cv.split(X_train, y_train, groups_train),
        scoring="f1_macro",
        n_jobs=-1,       # use all CPU cores
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV F1:  {search.best_score_:.3f}")

    # ── 4. Evaluate on held-out test set ──────────────────────────────────────
    y_pred = best_model.predict(X_test)

    print(f"\n  Classification report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Healthy", "Mild", "Severe"],
        zero_division=0,
    ))

    # Save confusion matrix as an image
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Healthy", "Mild", "Severe"]
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"{group_name} — Confusion Matrix")
    fig.tight_layout()
    cm_path = os.path.join(MODELS_DIR, f"{group_name.lower().replace(' ', '_')}_confusion_matrix.png")
    fig.savefig(cm_path)
    plt.close(fig)
    print(f"  Confusion matrix saved → {cm_path}")

    return best_model


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Group 1: Stroke / EMG ─────────────────────────────────────────────────
    csv1 = os.path.join(FEATURES_DIR, "group1_features.csv")
    if os.path.exists(csv1):
        model1 = train_group(csv1, GROUP1_FEATURES, "Group 1 Stroke")
        out1 = os.path.join(MODELS_DIR, "group1_svm.joblib")
        joblib.dump(model1, out1)
        print(f"\n  ✓ Group 1 model saved → {out1}")
    else:
        print(f"\n[SKIP] Group 1: {csv1} not found. Run build_dataset.py first.")

    # ── Group 2: Parkinson's / IMU ────────────────────────────────────────────
    csv2 = os.path.join(FEATURES_DIR, "group2_features.csv")
    if os.path.exists(csv2):
        model2 = train_group(csv2, GROUP2_FEATURES, "Group 2 Parkinsons")
        out2 = os.path.join(MODELS_DIR, "group2_svm.joblib")
        joblib.dump(model2, out2)
        print(f"\n  ✓ Group 2 model saved → {out2}")
    else:
        print(f"\n[SKIP] Group 2: {csv2} not found. Run build_dataset.py first.")

    print("\n✓ Phase 2 complete. Models saved in models/")
    print("  Hand these .joblib files to your lab partner for the FastAPI backend.")


if __name__ == "__main__":
    main()
