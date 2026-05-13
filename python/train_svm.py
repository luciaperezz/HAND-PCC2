"""
train_svm.py
------------
Phase 2: Trains a separate SVM classifier for each patient group.

Group 1 (Stroke/EMG):
    - Binary SVM: Healthy (0) vs Stroke (1)
    - Also computes and saves healthy centroid for k-NN progress tracker

Group 2 (Parkinson's/IMU):
    - Binary SVM: Healthy (0) vs Parkinson's (1)

Reads:
    features/group1_features.csv
    features/group2_features.csv

Outputs:
    models/group1_svm.joblib                (scaler + SVM for Group 1)
    models/group1_healthy_centroid.joblib   (centroid + scaler for k-NN)
    models/group2_svm.joblib                (scaler + SVM for Group 2)
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
                             ConfusionMatrixDisplay, balanced_accuracy_score,
                             f1_score, precision_score, recall_score)
import matplotlib
matplotlib.use("Agg")
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
    "svm__C":     [1, 10],
    "svm__gamma": [0.01, "scale"],
}


def load_data(csv_path: str, feature_cols: list) -> tuple:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=feature_cols)
    X      = df[feature_cols].values
    y      = df["label"].values.astype(int)
    groups = df["subject_id"].values
    print(f"  Loaded {len(df)} rows, {len(set(groups))} subjects")
    print(f"  Label distribution: { {int(k): int(v) for k,v in zip(*np.unique(y, return_counts=True))} }")
    return X, y, groups


def _fit_best_model(pipeline, X_train, y_train, groups_train, verbose=0):
    cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    search = GridSearchCV(
        pipeline,
        PARAM_GRID,
        cv=cv.split(X_train, y_train, groups_train),
        scoring="f1_macro",
        n_jobs=-1,
        verbose=verbose,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def _save_confusion_matrix(cm, labels, title, path):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Confusion matrix saved → {path}")


def train_group1(csv_path: str) -> tuple:
    """
    Group 1: Binary SVM — Healthy (0) vs Stroke (1).
    Also computes healthy centroid for k-NN progress tracker.
    Returns (trained_pipeline, centroid_dict)
    """
    print(f"\n── Group 1 Stroke ───────────────────────────────────────────────")

    X, y, groups = load_data(csv_path, GROUP1_FEATURES)

    # ── Compute healthy centroid for k-NN tracker ──────────────────────────
    # Use only healthy subjects (label=0) to define the reference centroid
    healthy_mask = y == 0
    scaler_knn = StandardScaler()
    X_scaled_all = scaler_knn.fit_transform(X)
    centroid = X_scaled_all[healthy_mask].mean(axis=0)
    print(f"  Healthy centroid computed from {healthy_mask.sum()} healthy recordings.")
    centroid_data = {
        "scaler":       scaler_knn,
        "centroid":     centroid,
        "feature_cols": GROUP1_FEATURES
    }

    # ── Train SVM ──────────────────────────────────────────────────────────
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
        )),
    ])

    eval_splits = 5  # fewer splits since dataset is small (180 rows)
    test_size   = 0.2
    base_seed   = 42

    print(f"  Repeated evaluation: {eval_splits} group splits | test_size={test_size}")

    rows = []
    cm_sum = np.zeros((2, 2), dtype=int)

    for i in range(eval_splits):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                     random_state=base_seed + i)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        model_i, best_params_i, best_cv_f1_i = _fit_best_model(
            pipeline, X_train, y_train, groups_train
        )
        y_pred = model_i.predict(X_test)
        cm_sum += confusion_matrix(y_test, y_pred, labels=[0, 1])

        rows.append({
            "split":                    i,
            "test_f1_macro":            f1_score(y_test, y_pred, average="macro", zero_division=0),
            "test_balanced_accuracy":   balanced_accuracy_score(y_test, y_pred),
            "test_precision_stroke":    precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            "test_recall_stroke":       recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            "innercv_best_f1_macro":    best_cv_f1_i,
            "best_params":              str(best_params_i),
        })

    report = pd.DataFrame(rows)
    report_path = os.path.join(MODELS_DIR, "group1_repeated_eval.csv")
    report.to_csv(report_path, index=False)

    def _mean_std(col):
        return float(report[col].mean()), float(report[col].std(ddof=1))

    f1m,  f1s  = _mean_std("test_f1_macro")
    bam,  bas  = _mean_std("test_balanced_accuracy")
    rpm,  rps  = _mean_std("test_recall_stroke")
    ppm,  pps  = _mean_std("test_precision_stroke")

    print("\n  Repeated-eval summary (mean ± std over splits):")
    print(f"    F1-macro:          {f1m:.3f} ± {f1s:.3f}")
    print(f"    Balanced accuracy: {bam:.3f} ± {bas:.3f}")
    print(f"    Recall (Stroke):   {rpm:.3f} ± {rps:.3f}")
    print(f"    Prec. (Stroke):    {ppm:.3f} ± {pps:.3f}")
    print(f"  Per-split metrics saved → {report_path}")

    _save_confusion_matrix(
        cm_sum, ["Healthy", "Stroke"],
        f"Group 1 Stroke — Confusion Matrix (sum over {eval_splits} splits)",
        os.path.join(MODELS_DIR, "group1_confusion_matrix_sum.png")
    )

    # Train final model on all data
    best_model, best_params, best_cv_f1 = _fit_best_model(
        pipeline, X, y, groups, verbose=1
    )
    print(f"\n  Final model (fit on all data)")
    print(f"  Best params: {best_params}")
    print(f"  Best CV F1:  {best_cv_f1:.3f}")

    return best_model, centroid_data


def train_group2(csv_path: str) -> Pipeline:
    """
    Group 2: Binary SVM — Healthy (0) vs Parkinson's (1).
    """
    print(f"\n── Group 2 Parkinson's ──────────────────────────────────────────")

    X, y, groups = load_data(csv_path, GROUP2_FEATURES)
    y = np.where(y > 0, 1, 0)
    print(f"  Labels after merging: { {int(k): int(v) for k,v in zip(*np.unique(y, return_counts=True))} }")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
        )),
    ])

    eval_splits = int(os.environ.get("GROUP2_EVAL_SPLITS", "5"))  # reduced from 30
    test_size   = float(os.environ.get("GROUP2_TEST_SIZE", "0.2"))  # reduced from 0.2
    base_seed   = int(os.environ.get("GROUP2_BASE_SEED", "42"))

    print(f"  Repeated evaluation: {eval_splits} group splits | test_size={test_size}")

    rows = []
    cm_sum = np.zeros((2, 2), dtype=int)

    for i in range(eval_splits):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                     random_state=base_seed + i)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        model_i, best_params_i, best_cv_f1_i = _fit_best_model(
            pipeline, X_train, y_train, groups_train
        )
        y_pred = model_i.predict(X_test)
        cm_sum += confusion_matrix(y_test, y_pred, labels=[0, 1])

        rows.append({
            "split":                        i,
            "test_f1_macro":                f1_score(y_test, y_pred, average="macro", zero_division=0),
            "test_balanced_accuracy":       balanced_accuracy_score(y_test, y_pred),
            "test_precision_parkinsons":    precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            "test_recall_parkinsons":       recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            "innercv_best_f1_macro":        best_cv_f1_i,
            "best_params":                  str(best_params_i),
        })

    report = pd.DataFrame(rows)
    report_path = os.path.join(MODELS_DIR, "group2_repeated_eval.csv")
    report.to_csv(report_path, index=False)

    def _mean_std(col):
        return float(report[col].mean()), float(report[col].std(ddof=1))

    f1m,  f1s  = _mean_std("test_f1_macro")
    bam,  bas  = _mean_std("test_balanced_accuracy")
    rpm,  rps  = _mean_std("test_recall_parkinsons")
    ppm,  pps  = _mean_std("test_precision_parkinsons")

    print("\n  Repeated-eval summary (mean ± std over splits):")
    print(f"    F1-macro:          {f1m:.3f} ± {f1s:.3f}")
    print(f"    Balanced accuracy: {bam:.3f} ± {bas:.3f}")
    print(f"    Recall (Parkinson):{rpm:.3f} ± {rps:.3f}")
    print(f"    Prec. (Parkinson): {ppm:.3f} ± {pps:.3f}")
    print(f"  Per-split metrics saved → {report_path}")

    _save_confusion_matrix(
        cm_sum, ["Healthy", "Parkinson's"],
        f"Group 2 Parkinson's — Confusion Matrix (sum over {eval_splits} splits)",
        os.path.join(MODELS_DIR, "group2_confusion_matrix_sum.png")
    )

    # Train final model on all data
    best_model, best_params, best_cv_f1 = _fit_best_model(
        pipeline, X, y, groups, verbose=1
    )
    print(f"\n  Final model (fit on all data)")
    print(f"  Best params: {best_params}")
    print(f"  Best CV F1:  {best_cv_f1:.3f}")

    return best_model


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Group 1: train SVM + compute healthy centroid ─────────────────────────
    csv1 = os.path.join(FEATURES_DIR, "group1_features.csv")
    if os.path.exists(csv1):
        model1, centroid_data = train_group1(csv1)

        out1_svm = os.path.join(MODELS_DIR, "group1_svm.joblib")
        joblib.dump(model1, out1_svm)
        print(f"\n  ✓ Group 1 SVM saved → {out1_svm}")

        out1_centroid = os.path.join(MODELS_DIR, "group1_healthy_centroid.joblib")
        joblib.dump(centroid_data, out1_centroid)
        print(f"  ✓ Group 1 centroid saved → {out1_centroid}")
    else:
        print(f"\n[SKIP] Group 1: {csv1} not found.")

    # ── Group 2: train SVM ────────────────────────────────────────────────────
    csv2 = os.path.join(FEATURES_DIR, "group2_features.csv")
    if os.path.exists(csv2):
        model2 = train_group2(csv2)
        out2 = os.path.join(MODELS_DIR, "group2_svm.joblib")
        joblib.dump(model2, out2)
        print(f"\n  ✓ Group 2 SVM saved → {out2}")
    else:
        print(f"\n[SKIP] Group 2: {csv2} not found.")

    print("\n✓ Phase 2 complete. Models saved in models/")
    print("  Files ready for the FastAPI backend:")
    print("    - models/group1_svm.joblib")
    print("    - models/group1_healthy_centroid.joblib")
    print("    - models/group2_svm.joblib")


if __name__ == "__main__":
    main()